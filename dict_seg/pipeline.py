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
)
from .segment import cut_and_count, cut_and_count_pos
from .segment import cut_and_count_text, cut_and_count_text_pos

_SORT_BIN = "gsort" if shutil.which("gsort") else "sort"
_SORT_IS_GNU = _SORT_BIN == "gsort"
_CHINESE_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")

_running_sorts: list[subprocess.Popen] = []


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


def _segment_worker(args):
    lines, tmp_path, no_pos, strip_html = args
    if no_pos:
        counter = cut_and_count(lines, strip_html=strip_html)
    else:
        counter = cut_and_count_pos(lines, strip_html=strip_html)
    _write_counter_to_file(counter, tmp_path, no_pos)


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


def _make_output_path(input_path: str) -> str:
    today = date.today().strftime("%Y%m%d")
    if os.path.isfile(input_path):
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(
            os.path.dirname(input_path) or ".",
            f"{stem}_{today}_{OUTPUT_FILE_SUFFIX}",
        )
    else:
        name = os.path.basename(input_path.rstrip("/"))
        return os.path.join(input_path, f"{name}_{today}_{OUTPUT_FILE_SUFFIX}")


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
        ch = sum(1 for c in text
                 if any(lo <= ord(c) <= hi
                        for lo, hi in ((0x3400, 0x4DBF),
                                       (0x4E00, 0x9FFF),
                                       (0xF900, 0xFAFF))))
        if len(text) > 0 and ch > len(text) * 0.005:
            return "utf-8"

    candidates = ["gb18030", "gbk", "big5"]
    best_enc = "utf-8"
    best_ratio = 0.0

    for enc in candidates:
        try:
            text = sample.decode(enc, errors="replace")
            ch = sum(1 for c in text
                 if any(lo <= ord(c) <= hi
                        for lo, hi in ((0x3400, 0x4DBF),
                                       (0x4E00, 0x9FFF),
                                       (0xF900, 0xFAFF))))
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
    with open(filepath, "rb") as f:
        count = 0
        while True:
            data = f.read(4 * 1024 * 1024)
            if not data:
                break
            count += data.count(b"\n")
            if count > 10:
                return False
    return count <= 10


# ── Multi-line chunked read ──────────────────────────────────────

def _read_chunks_by_lines(path: str, n: int, encoding: str = "utf-8"):
    chunk: list[str] = []
    with open(path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= n:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


# ── Temp file I/O ────────────────────────────────────────────────

def _write_counter_to_file(counter, path: str, no_pos: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if no_pos:
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
                              no_pos: bool = False) -> int:
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
            if no_pos:
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
                        best_pos = max(pos_counts, key=pos_counts.get)
                        fout.write(f"{prev_word}\t{total_freq}\t{best_pos}\n")
                        count += 1
                    prev_word = word
                    total_freq = cnt
                    pos_counts = {pos: cnt}
        if prev_word is not None:
            if no_pos:
                fout.write(f"{prev_word}\t{total_freq}\n")
            else:
                best_pos = max(pos_counts, key=pos_counts.get)
                fout.write(f"{prev_word}\t{total_freq}\t{best_pos}\n")
            count += 1
    return count


# ── Main pipeline ────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str | None = None,
    mem_mb: int = DEFAULT_MEM_MB,
    workers: int = WORKERS,
    min_freq: int = MIN_FREQ,
    no_pos: bool = True,
    strip_html: bool = False,
    force: bool = False,
) -> str:
    # Ensure fork-based multiprocessing for robustness
    if sys.platform != "win32":
        try:
            set_start_method("fork")
        except RuntimeError:
            pass  # already set
    txt_files = _collect_files(input_path)
    total_size = sum(os.path.getsize(fp) for fp in txt_files)
    print(f"Found {len(txt_files)} text file(s) "
          f"({total_size / 1e9:.1f} GB)")

    if output_path is None:
        output_path = _make_output_path(input_path)
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

        tmp_paths: list[str] = []
        batch_id = 0

        if normal_files:
            total_chunks = 0
            file_encodings: dict[str, str] = {}
            for fp in normal_files:
                enc = _detect_file_encoding(fp)
                file_encodings[fp] = enc
                est_lines = max(1, os.path.getsize(fp) // 100)
                total_chunks += (est_lines + CHUNK_LINES - 1) // CHUNK_LINES

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
                    for chunk in _read_chunks_by_lines(fp, CHUNK_LINES,
                                                       encoding=enc):
                        batch_lines.extend(chunk)
                        batch_chunks += 1

                        if batch_chunks >= CHUNKS_PER_BATCH:
                            tmp_path = os.path.join(
                                chunk_tmp_dir, f"chunk_{batch_id:06d}.txt")
                            tmp_paths.append(tmp_path)
                            batch_id += 1

                            if len(futures) >= max_pending:
                                futures.popleft().get()
                                pbar.update(chunks_per_task.popleft())

                            futures.append(pool.apply_async(
                                _segment_worker,
                                ((batch_lines, tmp_path, no_pos,
                                  strip_html),),
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
                        ((batch_lines, tmp_path, no_pos, strip_html),),
                    ))
                    chunks_per_task.append(batch_chunks)

                for fut, n in zip(futures, chunks_per_task):
                    try:
                        fut.get()
                    except Exception:
                        pool.terminate()
                        raise
                    pbar.update(n)
                pbar.close()

        for hf in huge_files:
            print(f"  Single-line mode: {os.path.basename(hf)} "
                  f"({os.path.getsize(hf) / 1e9:.1f} GB)")
            sl_tmp, batch_id = _process_single_line_file(
                hf, chunk_tmp_dir, batch_id, no_pos, strip_html)
            tmp_paths.extend(sl_tmp)

        # ── Stage 2: Concat + sort + merge ───────────────────────
        if not tmp_paths:
            print("  No content to process.")
            with open(output_path, "w", encoding="utf-8") as f:
                if no_pos:
                    f.write("word\tfreq\n")
                else:
                    f.write("word\tfreq\tpos\n")
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

        in_size = total_intermediate

        def _monitor():
            while p.poll() is None:
                out_size = (
                    os.path.getsize(merged_path)
                    if os.path.exists(merged_path) else 0
                )
                pct = out_size / in_size * 100 if in_size > 0 else 0
                e = time.time() - _monitor.t0
                print(f"\r    Sorting... {pct:.0f}% ({e:.0f}s)",
                      end="", flush=True)
                time.sleep(3)
            e = time.time() - _monitor.t0
            print(f"\r    Sorting... 100% ({e:.0f}s)", flush=True)
        _monitor.t0 = time.time()
        t = threading.Thread(target=_monitor, daemon=True)
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
                                               no_pos)
        print(f"  Unique words after merge: {word_count}")

        # ── Stage 3: Filter + sort by freq desc ──────────────────
        print(f"  Filtering Chinese-only, min_freq >= {min_freq}...")
        filtered_path = os.path.join(temp_dir, "filtered.txt")
        with open(merged_unsorted2, "r", encoding="utf-8") as fin, \
             open(filtered_path, "w", encoding="utf-8") as fout:
            for line in fin:
                parts = line.rstrip("\n\r").split("\t")
                w = parts[0]
                freq_s = parts[1]
                if int(freq_s) < min_freq:
                    continue
                if not _has_chinese(w):
                    continue
                fout.write(line)

        print("  Sorting by frequency descending...")
        subprocess.run(
            _make_sort_cmd(mem_mb, workers,
                           "-t", "\t", "-k2,2", "-nr",
                           "-o", output_path, filtered_path),
            check=True,
            env={**os.environ, "LC_ALL": "C"},
        )

        print(f"Done: {output_path}")
        return output_path

    finally:
        _kill_running_sorts()
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Single-line file processing ──────────────────────────────────

def _process_single_line_file(
    filepath: str,
    chunk_tmp_dir: str,
    start_id: int,
    no_pos: bool,
    strip_html: bool,
) -> tuple[list[str], int]:
    encoding = _detect_file_encoding(filepath)
    total_bytes = os.path.getsize(filepath)
    total_chunks = (
        total_bytes // (SINGLE_LINE_CHAR_CHUNK * 3) + 1
    )

    tmp_paths: list[str] = []
    pbar = tqdm.tqdm(total=total_chunks, desc="  S-line", unit="chunk")

    cut_func = cut_and_count_text if no_pos else cut_and_count_text_pos
    carryover = b""
    batch_id = start_id
    with open(filepath, "rb") as fin:
        while True:
            data = fin.read(BYTE_BUF)
            if not data:
                if carryover:
                    text = carryover.decode(encoding, errors="replace")
                    counter = cut_func(text, strip_html=strip_html)
                    tmp_path = os.path.join(
                        chunk_tmp_dir, f"chunk_{batch_id:06d}.txt")
                    _write_counter_to_file(counter, tmp_path, no_pos)
                    tmp_paths.append(tmp_path)
                    pbar.update(1)
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

            for i in range(0, len(text), SINGLE_LINE_CHAR_CHUNK):
                sub = text[i:i + SINGLE_LINE_CHAR_CHUNK]
                counter = cut_func(sub, strip_html=strip_html)
                tmp_path = os.path.join(
                    chunk_tmp_dir, f"chunk_{batch_id:06d}.txt")
                _write_counter_to_file(counter, tmp_path, no_pos)
                tmp_paths.append(tmp_path)
                batch_id += 1
                pbar.update(1)

        pbar.close()
    return tmp_paths, batch_id
