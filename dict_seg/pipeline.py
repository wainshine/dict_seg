import collections
import os
import re
import shutil
import sys
import time
import threading
import subprocess
import tempfile
from datetime import date
from multiprocessing import Pool, set_start_method

import tqdm

from .config import (
    DEFAULT_MEM_MB, WORKERS, MIN_FREQ, CHUNK_LINES, CHUNKS_PER_BATCH,
    SINGLE_LINE_CHAR_CHUNK, BYTE_BUF, TEXT_EXTENSIONS, OUTPUT_FILE_SUFFIX,
    OUTPUT_FILE_SUFFIX_POS, MAX_CHUNK_CHARS,
)
from .segment import cut_and_count, cut_and_count_pos
from .segment import count_presegmented, count_presegmented_text
from .segment import count_presegmented_pos, count_presegmented_text_pos

_SORT_BIN = "gsort" if shutil.which("gsort") else "sort"
_SORT_IS_GNU = _SORT_BIN == "gsort"
_CHINESE_RE = re.compile(r"[\u4E00-\u9FFF]")
_WORKER_TIMEOUT = 600

_running_sorts: list[subprocess.Popen] = []


def _split_file_to_dir(filepath: str, output_dir: str,
                       approx_bytes: int = 500 * 1024 * 1024) -> list[str]:
    encoding = _detect_file_encoding(filepath)
    char_limit = max(100_000, approx_bytes // 3)
    chunk_paths: list[str] = []
    part = 0
    carryover = b""
    out = None
    acc_chars = 0
    basename = os.path.splitext(os.path.basename(filepath))[0]
    with open(filepath, "rb") as fin:
        while True:
            data = fin.read(BYTE_BUF)
            if not data:
                if carryover and out:
                    text = carryover.decode(encoding, errors="replace")
                    out.write(text.encode("utf-8"))
                break
            segment = carryover + data
            try:
                text = segment.decode(encoding)
                carryover = b""
            except UnicodeDecodeError:
                for cut in range(-1, -5, -1):
                    try:
                        text = segment[:cut].decode(encoding)
                        carryover = segment[cut:]
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    text = segment.decode(encoding, errors="replace")
                    carryover = b""
            for i in range(0, len(text), char_limit):
                piece = text[i:i + char_limit]
                encoded = piece.encode("utf-8")
                if out is None or acc_chars + len(piece) > char_limit:
                    if out:
                        out.close()
                    part += 1
                    path = os.path.join(output_dir, f"{basename}_p{part:04d}.txt")
                    chunk_paths.append(path)
                    out = open(path, "wb")
                    acc_chars = 0
                out.write(encoded)
                acc_chars += len(piece)
    if out:
        out.close()
    return chunk_paths


def _make_sort_cmd(mem_mb: int, workers: int, *args: str) -> list[str]:
    cmd = [_SORT_BIN]
    if _SORT_IS_GNU:
        cmd.extend(["-S", f"{mem_mb}M", f"--parallel={workers}"])
    cmd.extend(args)
    return cmd


def _kill_running_sorts():
    for p in _running_sorts:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
                p.wait(timeout=2)
            except Exception:
                pass
    _running_sorts.clear()


def _monitor_sort_progress(p: subprocess.Popen, output_path: str,
                           input_path: str, label: str = "Sorting") -> None:
    in_size = os.path.getsize(input_path)
    t0 = time.time()
    while p.poll() is None:
        out_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        pct = out_size / in_size * 100 if in_size > 0 else 0
        elapsed = time.time() - t0
        print(f"\r    {label}... {pct:.0f}% ({elapsed:.0f}s)",
              end="", flush=True)
        time.sleep(3)
    elapsed = time.time() - t0
    print(f"\r    {label}... 100% ({elapsed:.0f}s)", flush=True)


def _filter_and_sort_final(merged_path: str, output_path: str,
                           temp_dir: str, min_freq: int, use_pos: bool,
                           mem_mb: int, workers: int) -> None:
    print(f"  Filtering Chinese-only, min_freq >= {min_freq}...")
    filtered_path = os.path.join(temp_dir, "filtered.txt")
    with open(merged_path, "r", encoding="utf-8") as fin, \
         open(filtered_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\n\r").split("\t")
            w = parts[0]
            freq_s = parts[-1]
            try:
                if int(freq_s) < min_freq:
                    continue
            except ValueError:
                continue
            if not _has_chinese(w):
                continue
            fout.write(line)

    print("  Sorting by frequency descending...")
    sort_key = "-k2,2" if not use_pos else "-k3,3"
    subprocess.run(
        _make_sort_cmd(mem_mb, workers,
                       "-t", "\t", sort_key, "-nr",
                       "-o", output_path, filtered_path),
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )


def _segment_worker(args: tuple[list[str], str, bool, bool, str | None, bool]) -> None:
    lines, tmp_path, use_pos, strip_html, user_dict, pre_seg = args
    if user_dict:
        import jieba
        jieba.load_userdict(user_dict)
    if pre_seg:
        counter = count_presegmented_pos(lines) if use_pos else count_presegmented(lines)
    elif not use_pos:
        counter = cut_and_count(lines, strip_html=strip_html)
    else:
        counter = cut_and_count_pos(lines, strip_html=strip_html)
    _write_counter_to_file(counter, tmp_path, use_pos)


# ── File collection ──────────────────────────────────────────────

def _collect_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        result = []
        for root, _dirs, files in os.walk(path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in TEXT_EXTENSIONS:
                    result.append(os.path.join(root, f))
        return sorted(result)
    raise FileNotFoundError(f"Path not found: {path}")


def _make_output_path(input_path: str, suffix: str | None = None) -> str:
    suf = suffix or OUTPUT_FILE_SUFFIX
    today = date.today().strftime("%Y%m%d")
    if os.path.isfile(input_path):
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(
            os.path.dirname(input_path) or ".",
            f"{stem}_{today}_{suf}",
        )
    else:
        name = os.path.basename(input_path.rstrip("/"))
        return os.path.join(input_path, f"{name}_{today}_{suf}")


# ── Encoding detection ───────────────────────────────────────────

def _detect_file_encoding(filepath: str) -> str:
    with open(filepath, "rb") as f:
        sample = f.read(64 * 1024)

    try:
        text = sample.decode("utf-8")
        if "\ufffd" not in text:
            return "utf-8"
    except UnicodeDecodeError:
        text = sample.decode("utf-8", errors="replace")
        ch = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
        if len(text) > 0 and ch > len(text) * 0.005:
            return "utf-8"

    candidates = ["gb18030", "gbk", "big5"]
    best_enc = "utf-8"
    best_ratio = 0.0

    for enc in candidates:
        try:
            text = sample.decode(enc, errors="replace")
            ch = sum(1 for c in text
                 if 0x4E00 <= ord(c) <= 0x9FFF)
            ratio = ch / len(text) if len(text) > 0 else 0
            if ratio > best_ratio:
                best_ratio = ratio
                best_enc = enc
        except (UnicodeDecodeError, LookupError):
            continue

    if best_ratio > 0.01:
        return best_enc

    try:
        import charset_normalizer
        result = charset_normalizer.from_path(
            filepath,
            preemptive_behaviour=False,
        )
        best = result.best()
        if best and best.encoding:
            return best.encoding
    except Exception:
        pass

    return best_enc


# ── Single-line file detection ───────────────────────────────────

def _is_single_line_file(filepath: str) -> bool:
    size = os.path.getsize(filepath)
    if size < 100 * 1024 * 1024:
        return False
    max_probe = 500 * 1024 * 1024
    with open(filepath, "rb") as f:
        count = 0
        bytes_read = 0
        while bytes_read < max_probe:
            data = f.read(4 * 1024 * 1024)
            if not data:
                break
            bytes_read += len(data)
            count += data.count(b"\n")
            if count > 10:
                if bytes_read / count > 65536:
                    return True
                return False
    return count <= 10


# ── Multi-line chunked read ──────────────────────────────────────

def _read_chunks_by_lines(path: str, n: int, encoding: str = "utf-8") -> "collections.abc.Generator[list[str], None, None]":
    chunk: list[str] = []
    chunk_chars = 0
    max_line_chars = SINGLE_LINE_CHAR_CHUNK
    with open(path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            if len(line) > max_line_chars:
                for i in range(0, len(line), max_line_chars):
                    sub = line[i:i + max_line_chars]
                    chunk.append(sub)
                    chunk_chars += len(sub)
                    if len(chunk) >= n or chunk_chars >= MAX_CHUNK_CHARS:
                        yield chunk
                        chunk = []
                        chunk_chars = 0
            else:
                chunk.append(line)
                chunk_chars += len(line)
                if len(chunk) >= n or chunk_chars >= MAX_CHUNK_CHARS:
                    yield chunk
                    chunk = []
                    chunk_chars = 0
        if chunk:
            yield chunk


# ── Temp file I/O ────────────────────────────────────────────────

def _write_counter_to_file(counter: "collections.Counter", path: str, use_pos: bool = True) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if not use_pos:
            for w, c in counter.items():
                w = w.replace("\t", " ").replace("\n", " ").replace("\r", " ")
                f.write(f"{w}\t{c}\n")
        else:
            for (w, pos), c in counter.items():
                w = w.replace("\t", " ").replace("\n", " ").replace("\r", " ")
                pos = pos.replace("\t", " ").replace("\n", " ").replace("\r", " ")
                f.write(f"{w}\t{pos}\t{c}\n")


def _concat_temp_files(paths: list[str], output_path: str) -> None:
    with open(output_path, "wb") as out:
        for tp in paths:
            with open(tp, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(tp)


def _has_chinese(word: str) -> bool:
    return _CHINESE_RE.search(word) is not None


# ── Merge sorted word-count files ────────────────────────────────

def _merge_sorted_word_counts(input_path: str, output_path: str,
                              use_pos: bool = True) -> int:
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        prev_word = None
        total_freq = 0
        pos_counts: dict[str, int] = {}
        for line in fin:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            word = parts[0]
            if not use_pos:
                if len(parts) < 2:
                    continue
                freq = int(parts[1])
                if word == prev_word:
                    total_freq += freq
                else:
                    if prev_word is not None:
                        fout.write(f"{prev_word}\t{total_freq}\n")
                        count += 1
                    prev_word = word
                    total_freq = freq
            else:
                if len(parts) < 3:
                    continue
                pos = parts[1]
                cnt = int(parts[2])
                if word == prev_word:
                    total_freq += cnt
                    pos_counts[pos] = pos_counts.get(pos, 0) + cnt
                else:
                    if prev_word is not None:
                        best_pos = max(pos_counts, key=pos_counts.get) if pos_counts else "?"
                        fout.write(f"{prev_word}\t{best_pos}\t{total_freq}\n")
                        count += 1
                    prev_word = word
                    total_freq = cnt
                    pos_counts = {pos: cnt}
        if prev_word is not None:
            if not use_pos:
                fout.write(f"{prev_word}\t{total_freq}\n")
            else:
                best_pos = max(pos_counts, key=pos_counts.get) if pos_counts else "?"
                fout.write(f"{prev_word}\t{best_pos}\t{total_freq}\n")
            count += 1
    return count


# ── Main pipeline ────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str | None = None,
    mem_mb: int = DEFAULT_MEM_MB,
    workers: int = WORKERS,
    min_freq: int = MIN_FREQ,
    use_pos: bool = False,
    strip_html: bool = False,
    user_dict: str | None = None,
    pre_seg: bool = False,
    force: bool = False,
) -> str:
    # Use spawn on macOS (fork can deadlock with jieba.posseg)
    if sys.platform == "darwin":
        set_start_method("spawn", force=True)
    if user_dict:
        import jieba
        jieba.load_userdict(user_dict)
    txt_files = _collect_files(input_path)
    # Exclude existing wordfreq output so it doesn't feed back in
    txt_files = [f for f in txt_files
                 if "wordfreq" not in os.path.basename(f)]
    total_size = sum(os.path.getsize(fp) for fp in txt_files)
    print(f"Found {len(txt_files)} text file(s) "
          f"({total_size / 1e9:.1f} GB)")

    if output_path is None:
        output_path = _make_output_path(input_path,
                                         OUTPUT_FILE_SUFFIX_POS if use_pos else None)
    if os.path.exists(output_path) and not force:
        print(f"  Output exists: {output_path}. Use --force to overwrite.")
        return output_path

    temp_dir = tempfile.mkdtemp(prefix="dict_seg_")
    chunk_tmp_dir = os.path.join(temp_dir, "chunks")
    os.makedirs(chunk_tmp_dir, exist_ok=True)
    merged_path = os.path.join(temp_dir, "merged.txt")

    try:
        # ── Stage 1: Segment and count ───────────────────────────
        huge_files = [f for f in txt_files if _is_single_line_file(f)]
        normal_files = [f for f in txt_files if f not in huge_files]

        if huge_files:
            split_dir = os.path.join(temp_dir, "split")
            os.makedirs(split_dir, exist_ok=True)
            for hf in huge_files:
                print(f"  Splitting: {os.path.basename(hf)} "
                      f"({os.path.getsize(hf) / 1e9:.1f} GB) into 500MB chunks...")
                chunks = _split_file_to_dir(hf, split_dir)
                normal_files.extend(chunks)
                print(f"    -> {len(chunks)} chunk files")

        tmp_paths: list[str] = []
        batch_id = 0

        if normal_files:
            total_chunks = 0
            file_encodings: dict[str, str] = {}
            failed_files: list[str] = []
            for fp in normal_files:
                try:
                    enc = _detect_file_encoding(fp)
                except Exception:
                    print(f"  WARNING: Cannot detect encoding for {fp}, skipping.")
                    failed_files.append(fp)
                    continue
                file_encodings[fp] = enc
                size = os.path.getsize(fp)
                est_lines = max(1, size // 200)
                try:
                    with open(fp, "rb") as sf:
                        head = sf.read(65536)
                except Exception:
                    head = b""
                if head:
                    nl = head.count(b"\n")
                    if nl >= 5:
                        est_lines = max(1, int(size * nl / len(head)))
                total_chunks += (est_lines + CHUNK_LINES - 1) // CHUNK_LINES

            if failed_files:
                normal_files = [f for f in normal_files if f not in failed_files]

            max_pending = workers * 4

            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(total=total_chunks, desc="  Segmenting",
                                unit="chunk")
                futures: collections.deque = collections.deque()
                chunks_per_task: collections.deque = collections.deque()
                batch_lines: list[str] = []
                batch_chunks: int = 0

                for fp in normal_files:
                    enc = file_encodings[fp]
                    try:
                        chunks_iter = _read_chunks_by_lines(fp, CHUNK_LINES,
                                                            encoding=enc)
                    except Exception:
                        print(f"  WARNING: Cannot read {fp}, skipping.")
                        continue
                    for chunk in chunks_iter:
                        batch_lines.extend(chunk)
                        batch_chunks += 1

                        if batch_chunks >= CHUNKS_PER_BATCH:
                            tmp_path = os.path.join(
                                chunk_tmp_dir, f"chunk_{batch_id:06d}.txt")
                            tmp_paths.append(tmp_path)
                            batch_id += 1

                            if len(futures) >= max_pending:
                                futures.popleft().get(timeout=_WORKER_TIMEOUT)
                                pbar.update(chunks_per_task.popleft())

                            futures.append(pool.apply_async(
                                _segment_worker,
                                ((batch_lines, tmp_path, use_pos,
                                  strip_html, user_dict, pre_seg),),
                            ))
                            chunks_per_task.append(batch_chunks)
                            batch_lines = []
                            batch_chunks = 0

                if batch_lines:
                    tmp_path = os.path.join(
                        chunk_tmp_dir, f"chunk_{batch_id:06d}.txt")
                    tmp_paths.append(tmp_path)
                    batch_id += 1
                    futures.append(pool.apply_async(
                        _segment_worker,
                        ((batch_lines, tmp_path, use_pos, strip_html,
                          user_dict, pre_seg),),
                    ))
                    chunks_per_task.append(batch_chunks)

                for fut, n in zip(futures, chunks_per_task):
                    try:
                        fut.get(timeout=_WORKER_TIMEOUT)
                    except Exception:
                        pool.terminate()
                        raise
                    pbar.update(n)
                pbar.close()

        # ── Stage 2: Concat + sort + merge ───────────────────────
        if not tmp_paths:
            print("  No content to process.")
            with open(output_path, "w", encoding="utf-8") as f:
                if not use_pos:
                    f.write("word\tfreq\n")
                else:
                    f.write("word\tpos\tfreq\n")
            return output_path

        merged_unsorted = os.path.join(temp_dir, "merged_unsorted.txt")
        print(f"  Concatenating {len(tmp_paths)} chunk files...")
        _concat_temp_files(tmp_paths, merged_unsorted)

        total_intermediate = 0
        if os.path.exists(merged_unsorted):
            total_intermediate = os.path.getsize(merged_unsorted)
        print(f"  Intermediate data: {total_intermediate / 1e9:.2f} GB")

        print(f"  Sorting by word (mem={mem_mb}M, workers={workers})...")
        sort_env = {**os.environ, "LC_ALL": "C"}
        p = subprocess.Popen(
            _make_sort_cmd(mem_mb, workers,
                           "-t", "\t", "-k1,1",
                           "-o", merged_path, merged_unsorted),
            stderr=subprocess.PIPE, text=True, env=sort_env)
        _running_sorts.append(p)

        t = threading.Thread(
            target=_monitor_sort_progress,
            args=(p, merged_path, merged_unsorted, "Sorting"),
            daemon=True)
        t.start()

        _, stderr = p.communicate()
        t.join(timeout=1)
        if p.returncode != 0:
            raise RuntimeError(f"Sort failed: {stderr}")

        try:
            os.remove(merged_unsorted)
        except OSError:
            pass

        print("  Merging same-word counts...")
        merged_unsorted2 = os.path.join(temp_dir, "merged_unsorted2.txt")
        word_count = _merge_sorted_word_counts(merged_path, merged_unsorted2,
                                               use_pos)
        print(f"  Unique words after merge: {word_count}")

        _filter_and_sort_final(merged_unsorted2, output_path, temp_dir,
                               min_freq, use_pos, mem_mb, workers)

        print(f"Done: {output_path}")
        return output_path

    finally:
        _kill_running_sorts()
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Wordfreq merge ───────────────────────────────────────────────

def merge_wordfreq_files(
    input_path: str,
    output_path: str | None = None,
    mem_mb: int = DEFAULT_MEM_MB,
    workers: int = WORKERS,
    min_freq: int = MIN_FREQ,
    use_pos: bool = False,
    force: bool = False,
) -> str:
    """Merge multiple _wordfreq.txt files from a directory into one.

    input_path can be a directory containing _wordfreq.txt files,
    or multiple file paths will be auto-collected via _collect_files.
    """
    txt_files = _collect_files(input_path)
    txt_files = [f for f in txt_files
                 if "wordfreq" in os.path.basename(f)
                 and "merged" not in os.path.basename(f)]
    if len(txt_files) < 2:
        print(f"  Need >= 2 wordfreq files, found {len(txt_files)}")
        if txt_files and output_path:
            shutil.copy(txt_files[0], output_path)
        return output_path or ""

    total_size = sum(os.path.getsize(fp) for fp in txt_files)
    print(f"Found {len(txt_files)} wordfreq file(s) "
          f"({total_size / 1e9:.2f} GB)")

    if output_path is None:
        today = date.today().strftime("%Y%m%d")
        suf = OUTPUT_FILE_SUFFIX_POS if use_pos else OUTPUT_FILE_SUFFIX
        output_path = os.path.join(input_path, f"merged_{today}_{suf}")

    if os.path.exists(output_path) and not force:
        print(f"  Output exists: {output_path}. Use --force to overwrite.")
        return output_path

    temp_dir = tempfile.mkdtemp(prefix="dict_seg_merge_")
    merged_path = os.path.join(temp_dir, "merged.txt")

    try:
        merged_unsorted = os.path.join(temp_dir, "merged_unsorted.txt")
        print(f"  Concatenating {len(txt_files)} wordfreq files...")
        with open(merged_unsorted, "wb") as out:
            for tp in txt_files:
                with open(tp, "rb") as f:
                    shutil.copyfileobj(f, out)

        sort_env = {**os.environ, "LC_ALL": "C"}
        p = subprocess.Popen(
            _make_sort_cmd(mem_mb, workers,
                           "-t", "\t", "-k1,1",
                           "-o", merged_path, merged_unsorted),
            stderr=subprocess.PIPE, text=True, env=sort_env)
        _running_sorts.append(p)

        mt = threading.Thread(
            target=_monitor_sort_progress,
            args=(p, merged_path, merged_unsorted, "Sorting"),
            daemon=True)
        mt.start()

        _, stderr = p.communicate()
        mt.join(timeout=1)
        if p.returncode != 0:
            raise RuntimeError(f"Sort failed: {stderr}")

        try:
            os.remove(merged_unsorted)
        except OSError:
            pass

        print("  Merging same-word counts...")
        merged_unsorted2 = os.path.join(temp_dir, "merged_unsorted2.txt")
        word_count = _merge_sorted_word_counts(merged_path, merged_unsorted2,
                                               use_pos)
        print(f"  Unique words after merge: {word_count}")

        _filter_and_sort_final(merged_unsorted2, output_path, temp_dir,
                               min_freq, use_pos, mem_mb, workers)

        print(f"Done: {output_path}")
        return output_path

    finally:
        _kill_running_sorts()
        shutil.rmtree(temp_dir, ignore_errors=True)
