"""Tests for dict_seg — batch Chinese word segmentation + word frequency counting."""

import collections
import os
import tempfile

import pytest

from dict_seg.config import (
    TEXT_EXTENSIONS, MIN_FREQ, CHUNK_LINES, CHUNKS_PER_BATCH, WORKERS,
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
)


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
        # BOM may cause U+FEFF to appear but the encoding should resolve
        assert enc in ("utf-8", "gb18030")  # either is acceptable


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
        count = _merge_sorted_word_counts(str(inp), str(out), no_pos=True)
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
        count = _merge_sorted_word_counts(str(inp), str(out), no_pos=False)
        assert count == 2

    def test_empty_input(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("", encoding="utf-8")
        count = _merge_sorted_word_counts(str(inp), str(out), no_pos=True)
        assert count == 0

    def test_blank_lines_skipped(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("\n\n品牌\t3\n\n颜色\t2\n", encoding="utf-8")
        count = _merge_sorted_word_counts(str(inp), str(out), no_pos=True)
        assert count == 2

    def test_malformed_line_skipped(self, tmp_path):
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("品牌\t3\nbadline\n颜色\t2\n", encoding="utf-8")
        # len(parts) guard prevents IndexError; malformed line silently skipped
        count = _merge_sorted_word_counts(str(inp), str(out), no_pos=True)
        assert count == 2
        result = out.read_text(encoding="utf-8")
        assert "badline" not in result


class TestWriteCounterToFile:
    def test_no_pos(self, tmp_path):
        counter = collections.Counter({"品牌": 5, "颜色": 3})
        p = tmp_path / "out.txt"
        _write_counter_to_file(counter, str(p), no_pos=True)
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert "\t" in lines[0]

    def test_pos_mode(self, tmp_path):
        counter = collections.Counter({("品牌", "n"): 5, ("颜色", "n"): 3})
        p = tmp_path / "out.txt"
        _write_counter_to_file(counter, str(p), no_pos=False)
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert lines[0].count("\t") == 2  # word \t pos \t freq

    def test_tab_escaping(self, tmp_path):
        counter = collections.Counter({"wo\trd": 1})
        p = tmp_path / "out.txt"
        _write_counter_to_file(counter, str(p), no_pos=True)
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
