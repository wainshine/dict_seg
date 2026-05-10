"""Tests for dict_seg — batch Chinese word segmentation + word frequency counting."""

import collections
import io
import os
import tempfile

import pytest
from click.testing import CliRunner

from dict_seg.config import (
    TEXT_EXTENSIONS, MIN_FREQ, CHUNK_LINES, CHUNKS_PER_BATCH, WORKERS,
    OUTPUT_FILE_SUFFIX, OUTPUT_FILE_SUFFIX_POS,
)
from dict_seg.segment import _is_garbage, cut_and_count, cut_and_count_pos
from dict_seg.segment import cut_and_count_text, cut_and_count_text_pos
from dict_seg.pipeline import (
    _collect_files,
    _make_output_path,
    _detect_file_encoding,
    _is_single_line_file,
    _merge_sorted_word_counts,
    _has_chinese,
    _write_counter_to_file,
    _make_sort_cmd,
    _filter_and_sort_final,
)
from dict_seg.__main__ import main, merge


# ── segment.py tests ──────────────────────────────────────────────

class TestIsGarbage:
    def test_pure_digits_long(self):
        assert _is_garbage("12345")               # 5 digits
        assert _is_garbage("35190107271")          # 11 digits

    def test_pure_digits_short(self):
        assert not _is_garbage("1234")             # 4 digits
        assert not _is_garbage("1")

    def test_mixed_digits(self):
        assert not _is_garbage("abc12345")         # not pure digits
        assert not _is_garbage("T恤12345")

    def test_pure_ascii_long(self):
        assert _is_garbage("abcdefghijkl")           # 12 chars
        assert _is_garbage("T1Z0eVXy8iXXXXXXXX")     # mixed alnum 19

    def test_pure_ascii_short(self):
        assert not _is_garbage("abc")               # 3 chars
        assert not _is_garbage("item_url")          # 8 chars

    def test_chinese(self):
        assert not _is_garbage("品牌")
        assert not _is_garbage("T恤")
        assert not _is_garbage("4S店")


class TestCutAndCount:
    def test_basic(self):
        counter = cut_and_count(["我来到北京清华大学"])
        assert counter["来到"] >= 1
        assert counter["北京"] >= 1
        assert counter["清华大学"] >= 1

    def test_empty_lines(self):
        counter = cut_and_count([""])
        assert len(counter) == 0

    def test_garbage_filtered(self):
        counter = cut_and_count(["item_id 35190107271 T1Z0eVXy8iXXXXXXXX"])
        assert "35190107271" not in counter
        assert "T1Z0eVXy8iXXXXXXXX" not in counter
        # jieba splits "item_id" into "item" / "_" / "id"
        assert "item" in counter or "id" in counter

    def test_chinese_kept(self):
        counter = cut_and_count(["品牌 颜色 材质"])
        assert "品牌" in counter
        assert "颜色" in counter
        assert "材质" in counter


class TestCutAndCountText:
    def test_single_string(self):
        counter = cut_and_count_text("我来到北京清华大学")
        assert counter["来到"] >= 1
        assert counter["北京"] >= 1

    def test_with_html_strip(self):
        counter = cut_and_count_text("<p>你好世界</p><script>var x=1;</script>", strip_html=True)
        assert "你好世界" in counter or "你好" in counter
        assert "var" not in counter


class TestPosCutAndCount:
    def test_pos_mode(self):
        counter = cut_and_count_pos(["我来到北京清华大学"])
        keys = list(counter.keys())
        assert len(keys) > 0
        assert isinstance(keys[0], tuple)
        assert len(keys[0]) == 2  # (word, pos)


# ── pipeline.py: utility function tests ───────────────────────────

class TestCollectFiles:
    def test_single_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert _collect_files(str(f)) == [str(f)]

    def test_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.csv").write_text("b")
        (tmp_path / "img.jpg").write_text("jpg")  # not in TEXT_EXTENSIONS
        result = _collect_files(str(tmp_path))
        assert len(result) == 2
        for p in result:
            assert p.endswith(".txt") or p.endswith(".csv")

    def test_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            _collect_files("/nonexistent/path/xyz")

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert _collect_files(str(d)) == []


class TestHasChinese:
    def test_pure_chinese(self):
        assert _has_chinese("品牌")
        assert _has_chinese("清华大学")

    def test_mixed(self):
        assert _has_chinese("T恤")
        assert _has_chinese("4S店")
        assert _has_chinese("PH值")

    def test_no_chinese(self):
        assert not _has_chinese("abc")
        assert not _has_chinese("123")
        assert not _has_chinese("item_id")


class TestMakeOutputPath:
    def test_file_input(self, tmp_path):
        f = tmp_path / "corpus.txt"
        f.write_text("dummy")
        path = _make_output_path(str(f))
        assert "corpus_" in path
        assert path.endswith("wordfreq.txt")

    def test_directory_input(self):
        path = _make_output_path("/data/mydir/")
        assert "mydir_" in path
        assert path.endswith("wordfreq.txt")


class TestDetectEncoding:
    def test_utf8_clean(self, tmp_path):
        f = tmp_path / "utf8.txt"
        f.write_text("你好世界", encoding="utf-8")
        assert _detect_file_encoding(str(f)) == "utf-8"

    def test_utf8_bom(self, tmp_path):
        f = tmp_path / "bom.txt"
        f.write_bytes(b'\xef\xbb\xbf' + "你好".encode("utf-8"))
        enc = _detect_file_encoding(str(f))
        assert enc == "utf-8"


class TestIsSingleLineFile:
    def test_small_file(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("line1\n" * 100)
        assert not _is_single_line_file(str(f))

    def test_file_under_100mb(self, tmp_path):
        # Any file under 100 MB is not single-line
        f = tmp_path / "tiny.txt"
        f.write_text("single line")
        assert not _is_single_line_file(str(f))


class TestMergeSortedWordCounts:
    def test_basic_merge(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("品牌\t3\n颜色\t2\n颜色\t1\n材质\t5\n", encoding="utf-8")
        count = _merge_sorted_word_counts(str(inp), str(out), use_pos=False)
        assert count == 3
        result = out.read_text(encoding="utf-8").strip().split("\n")
        lines = {l.split("\t")[0]: int(l.split("\t")[1]) for l in result}
        assert lines["品牌"] == 3
        assert lines["颜色"] == 3  # 2+1
        assert lines["材质"] == 5

    def test_merge_pos(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("品牌\tn\t3\n品牌\tn\t2\n颜色\tn\t5\n", encoding="utf-8")
        count = _merge_sorted_word_counts(str(inp), str(out), use_pos=True)
        assert count == 2
        # Output format: word\tpos\tfreq
        result = out.read_text(encoding="utf-8")
        assert "品牌\tn\t5" in result
        assert "颜色\tn\t5" in result

    def test_empty_input(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("", encoding="utf-8")
        count = _merge_sorted_word_counts(str(inp), str(out), use_pos=False)
        assert count == 0

    def test_blank_lines_skipped(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("\n\n品牌\t3\n\n颜色\t2\n", encoding="utf-8")
        count = _merge_sorted_word_counts(str(inp), str(out), use_pos=False)
        assert count == 2

    def test_malformed_line_skipped(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("品牌\t3\nbadline\n颜色\t2\n", encoding="utf-8")
        # len(parts) guard prevents IndexError; malformed line silently skipped
        count = _merge_sorted_word_counts(str(inp), str(out), use_pos=False)
        assert count == 2
        result = out.read_text(encoding="utf-8")
        assert "badline" not in result


class TestWriteCounterToFile:
    def test_no_pos(self, tmp_path):
        counter = collections.Counter({"品牌": 5, "颜色": 3})
        p = tmp_path / "out.txt"
        _write_counter_to_file(counter, str(p), use_pos=False)
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert "\t" in lines[0]

    def test_pos_mode(self, tmp_path):
        counter = collections.Counter({("品牌", "n"): 5, ("颜色", "n"): 3})
        p = tmp_path / "out.txt"
        _write_counter_to_file(counter, str(p), use_pos=True)
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        # Intermediate format: word\tpos\tfreq
        assert lines[0].count("\t") == 2
        assert "\t" in lines[0]

    def test_tab_escaping(self, tmp_path):
        counter = collections.Counter({"wo\trd": 1})
        p = tmp_path / "out.txt"
        _write_counter_to_file(counter, str(p), use_pos=False)
        line = p.read_text(encoding="utf-8").strip()
        assert line.count("\t") == 1  # only the separator, word tab replaced
        assert "wo rd" in line


class TestMakeSortCmd:
    def test_gnu_sort(self, monkeypatch):
        monkeypatch.setattr("dict_seg.pipeline._SORT_IS_GNU", True)
        cmd = _make_sort_cmd(1024, 8, "-k1", "-o", "out", "in")
        assert "-S" in cmd
        assert "--parallel=8" in cmd
        assert cmd[0] != "gsort"  # unchanged by monkeypatch

    def test_bsd_sort(self, monkeypatch):
        monkeypatch.setattr("dict_seg.pipeline._SORT_IS_GNU", False)
        cmd = _make_sort_cmd(1024, 8, "-k1", "-o", "out", "in")
        assert "-S" not in cmd
        assert "--parallel" not in cmd


# ── Integration tests ─────────────────────────────────────────────

class TestEndToEnd:
    def test_small_corpus(self, tmp_path):
        from dict_seg.pipeline import run_pipeline

        f = tmp_path / "corpus.txt"
        f.write_text(
            "他来到了网易杭研大厦\n"
            "小明硕士毕业于中国科学院计算所\n"
            "我来到北京清华大学\n"
            "乒乓球拍卖完了\n"
            "中国科学技术大学\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.txt"
        result = run_pipeline(str(f), output_path=str(out), min_freq=1)
        assert result == str(out)
        assert out.exists()
        lines = out.read_text(encoding="utf-8").split("\n")
        assert len(lines) > 5

    def test_min_freq_filter(self, tmp_path):
        from dict_seg.pipeline import run_pipeline

        f = tmp_path / "corpus.txt"
        content = "品牌 品牌 品牌\n颜色 颜色\n材质\n"  # 品牌=3, 颜色=2, 材质=1
        f.write_text(content, encoding="utf-8")
        out = tmp_path / "out.txt"
        run_pipeline(str(f), output_path=str(out), min_freq=2)
        lines = out.read_text(encoding="utf-8").split("\n")
        words = {l.split("\t")[0] for l in lines if l.strip()}
        assert "品牌" in words
        assert "颜色" in words
        assert "材质" not in words  # freq=1 < 2

    def test_overwrite_protection(self, tmp_path):
        from dict_seg.pipeline import run_pipeline

        f = tmp_path / "corpus.txt"
        # Repeat to pass min_freq=5
        f.write_text("测试文本 测试文本 测试文本 测试文本 测试文本\n", encoding="utf-8")
        out = tmp_path / "out.txt"
        out.write_text("existing", encoding="utf-8")

        # Without force, should return without overwriting
        run_pipeline(str(f), output_path=str(out), workers=1, force=False)
        content1 = out.read_text(encoding="utf-8")
        assert content1 == "existing"  # not overwritten

        # With force, should overwrite
        run_pipeline(str(f), output_path=str(out), workers=1, force=True)
        content2 = out.read_text(encoding="utf-8")
        assert "测试" in content2

    def test_pos_suffix(self, tmp_path):
        from dict_seg.pipeline import run_pipeline
        from dict_seg.config import OUTPUT_FILE_SUFFIX_POS

        f = tmp_path / "corpus.txt"
        f.write_text("测试测试测试测试测试测试\n" * 10, encoding="utf-8")
        out = run_pipeline(str(f), use_pos=True, workers=1, min_freq=1, force=True)
        assert OUTPUT_FILE_SUFFIX_POS in out

    def test_pos_output_format(self, tmp_path):
        from dict_seg.pipeline import run_pipeline

        f = tmp_path / "corpus.txt"
        f.write_text("测试测试测试测试测试测试\n" * 10, encoding="utf-8")
        out = run_pipeline(str(f), use_pos=True, workers=1, min_freq=1, force=True)
        lines = open(out, encoding="utf-8").readlines()
        assert len(lines) > 0
        # POS format: word\tpos\tfreq (3 columns)
        cols = lines[0].rstrip("\n").split("\t")
        assert len(cols) == 3

    def test_wordfreq_excluded_from_dir(self, tmp_path):
        from dict_seg.pipeline import run_pipeline

        d = tmp_path / "corpus_dir"
        d.mkdir()
        (d / "a.txt").write_text("测试文本\n" * 10, encoding="utf-8")
        (d / "b.txt").write_text("品牌颜色\n" * 10, encoding="utf-8")
        (d / "c_wordfreq.txt").write_text("existing\t100\n", encoding="utf-8")

        out = run_pipeline(str(d), workers=1, min_freq=1, force=True)
        # c_wordfreq.txt should be excluded, only a.txt and b.txt processed
        result = open(out, encoding="utf-8").read()
        assert "existing" not in result


class TestMergeWordfreq:
    def test_merge_non_pos(self, tmp_path):
        from dict_seg.pipeline import merge_wordfreq_files

        d = tmp_path / "freqs"
        d.mkdir()
        (d / "a_wordfreq.txt").write_text("品牌\t5\n颜色\t3\n", encoding="utf-8")
        (d / "b_wordfreq.txt").write_text("品牌\t2\n材质\t4\n", encoding="utf-8")

        out = merge_wordfreq_files(str(d), min_freq=1, force=True)
        result = {}
        for line in open(out, encoding="utf-8"):
            w, f = line.strip().split("\t")
            result[w] = int(f)
        assert result["品牌"] == 7
        assert result["颜色"] == 3
        assert result["材质"] == 4

    def test_merge_pos(self, tmp_path):
        from dict_seg.pipeline import merge_wordfreq_files

        d = tmp_path / "freqs"
        d.mkdir()
        (d / "a_wordfreq.txt").write_text("品牌\tn\t5\n颜色\tn\t3\n", encoding="utf-8")
        (d / "b_wordfreq.txt").write_text("品牌\tn\t2\n材质\tn\t4\n", encoding="utf-8")

        out = merge_wordfreq_files(str(d), use_pos=True, min_freq=1, force=True)
        lines = [l.strip().split("\t") for l in open(out, encoding="utf-8")]
        assert len(lines[0]) == 3  # word\tpos\tfreq

    def test_merge_suffix_pos(self, tmp_path):
        from dict_seg.pipeline import merge_wordfreq_files
        from dict_seg.config import OUTPUT_FILE_SUFFIX_POS

        d = tmp_path / "freqs"
        d.mkdir()
        (d / "a_wordfreq.txt").write_text("品牌\tn\t5\n", encoding="utf-8")
        (d / "b_wordfreq.txt").write_text("品牌\tn\t2\n", encoding="utf-8")

        out = merge_wordfreq_files(str(d), use_pos=True, min_freq=1, force=True)
        assert OUTPUT_FILE_SUFFIX_POS in out

    def test_merge_e2e_pos(self, tmp_path):
        from dict_seg.pipeline import merge_wordfreq_files

        d = tmp_path / "freqs"
        d.mkdir()
        (d / "a_wordfreq.txt").write_text("品牌\tn\t5\n颜色\tn\t3\n", encoding="utf-8")
        (d / "b_wordfreq.txt").write_text("品牌\tn\t2\n材质\tn\t4\n", encoding="utf-8")

        out = merge_wordfreq_files(str(d), use_pos=True, min_freq=1, force=True)
        assert os.path.exists(out)
        for line in open(out, encoding="utf-8"):
            cols = line.strip().split("\t")
            assert len(cols) == 3  # word\tpos\tfreq

    def test_merge_no_overwrite(self, tmp_path):
        from dict_seg.pipeline import merge_wordfreq_files

        d = tmp_path / "freqs"
        d.mkdir()
        (d / "a_wordfreq.txt").write_text("品牌\t5\n", encoding="utf-8")
        (d / "b_wordfreq.txt").write_text("颜色\t3\n", encoding="utf-8")
        existing = d / "merged_wordfreq.txt"
        existing.write_text("preserve", encoding="utf-8")

        merge_wordfreq_files(str(d), output_path=str(existing), force=False)
        assert existing.read_text(encoding="utf-8") == "preserve"

    def test_merge_force_overwrite(self, tmp_path):
        from dict_seg.pipeline import merge_wordfreq_files

        d = tmp_path / "freqs"
        d.mkdir()
        (d / "a_wordfreq.txt").write_text("品牌\t5\n", encoding="utf-8")
        (d / "b_wordfreq.txt").write_text("颜色\t3\n", encoding="utf-8")
        out = d / "merged_wordfreq.txt"
        out.write_text("existing", encoding="utf-8")

        merge_wordfreq_files(str(d), output_path=str(out), force=True)
        content = out.read_text(encoding="utf-8")
        assert "品牌" in content
        assert "existing" not in content


class TestStripHtmlPosIntegration:
    def test_strip_html_pos(self, tmp_path):
        from dict_seg.pipeline import run_pipeline

        f = tmp_path / "corpus.html"
        f.write_text("<html><body><p>品牌品牌品牌品牌品牌</p><script>var x=1;</script></body></html>\n" * 5,
                     encoding="utf-8")
        out = run_pipeline(str(f), use_pos=True, strip_html=True, workers=1,
                           min_freq=1, force=True)
        content = open(out, encoding="utf-8").read()
        assert "品牌" in content
        assert "var" not in content
        assert "script" not in content


class TestFilterAndSortFinal:
    def test_malformed_freq_tolerated(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("品牌\t5\nbadfreq_line\n颜色\t3\n材质\tabc\n", encoding="utf-8")
        _filter_and_sort_final(str(inp), str(out), str(tmp_path),
                               min_freq=1, use_pos=False, mem_mb=64, workers=1)
        result = out.read_text(encoding="utf-8")
        assert "品牌" in result
        assert "颜色" in result
        assert "badfreq" not in result
        assert "abc" not in result

    def test_min_freq_filter_applied(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("品牌\t5\n颜色\t2\n材质\t1\n", encoding="utf-8")
        _filter_and_sort_final(str(inp), str(out), str(tmp_path),
                               min_freq=3, use_pos=False, mem_mb=64, workers=1)
        result = out.read_text(encoding="utf-8")
        assert "品牌" in result
        assert "颜色" not in result
        assert "材质" not in result

    def test_pos_format(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("品牌\tn\t5\n颜色\tn\t3\n", encoding="utf-8")
        _filter_and_sort_final(str(inp), str(out), str(tmp_path),
                               min_freq=1, use_pos=True, mem_mb=64, workers=1)
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        cols = lines[0].split("\t")
        assert len(cols) == 3


class TestIsSingleLineFileCap:
    def test_large_file_0_newlines_capped(self, monkeypatch):
        def fake_open(path, mode):
            return io.BytesIO(b"x" * (500 * 1024 * 1024))
        monkeypatch.setattr("builtins.open", fake_open)
        monkeypatch.setattr("os.path.getsize", lambda p: 800 * 1024 * 1024)
        assert _is_single_line_file("/fake/large.bin")

    def test_large_file_many_newlines_early(self, monkeypatch):
        data = (b"a" * 1000 + b"\n") * 100
        def fake_open(path, mode):
            return io.BytesIO(data)
        monkeypatch.setattr("builtins.open", fake_open)
        monkeypatch.setattr("os.path.getsize", lambda p: 800 * 1024 * 1024)
        assert not _is_single_line_file("/fake/large.bin")


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "1.2.0" in result.output

    def test_merge_version(self):
        runner = CliRunner()
        result = runner.invoke(merge, ["--version"])
        assert result.exit_code == 0
        assert "1.2.0" in result.output

    def test_user_dict(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("自然语言处理很有趣\n" * 50, encoding="utf-8")
        user_dict = tmp_path / "user_dict.txt"
        user_dict.write_text("自然语言处理 100\n", encoding="utf-8")
        out = tmp_path / "out.txt"
        runner = CliRunner()
        result = runner.invoke(main, [
            str(corpus), "-o", str(out), "-w", "1", "--min-freq", "1",
            "--user-dict", str(user_dict), "--force",
        ])
        assert result.exit_code == 0
        content = out.read_text(encoding="utf-8")
        assert "自然语言处理" in content

    def test_user_dict_pos(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("自然语言处理很有趣\n" * 50, encoding="utf-8")
        user_dict = tmp_path / "user_dict.txt"
        user_dict.write_text("自然语言处理 100\n", encoding="utf-8")
        out = tmp_path / "out.txt"
        runner = CliRunner()
        result = runner.invoke(main, [
            str(corpus), "-o", str(out), "-w", "1", "--min-freq", "1",
            "--user-dict", str(user_dict), "--pos", "--force",
        ])
        assert result.exit_code == 0
        content = out.read_text(encoding="utf-8")
        assert "自然语言处理" in content
