from multiprocessing import cpu_count
import os


def _detect_cgroup_limit():
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            parts = f.read().strip().split()
            if parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1]) if len(parts) > 1 else 100000
                return max(1, quota // period)
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
            quota = int(f.read().strip())
        if quota > 0:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
                period = int(f.read().strip())
            return max(1, quota // period)
    except Exception:
        pass
    return None

# Output file suffix appended as {name}_{date}_{suffix}
OUTPUT_FILE_SUFFIX = "wordfreq.txt"
OUTPUT_FILE_SUFFIX_POS = "pos_wordfreq.txt"

# Memory budget (MB) for external sort subprocess
DEFAULT_MEM_MB = 1024

# Minimum frequency for final output
MIN_FREQ = 5

# Worker processes (default: half of CPU cores, cgroup-aware)
_RAW_COUNT = cpu_count()
_CGROUP_LIMIT = _detect_cgroup_limit()
if _CGROUP_LIMIT is not None:
    _RAW_COUNT = min(_RAW_COUNT, _CGROUP_LIMIT)
WORKERS = max(1, _RAW_COUNT // 2)

# Lines per raw chunk read from file (~500KB at 100 bytes/line)
CHUNK_LINES = 5_000

# Max characters per chunk (alongside CHUNK_LINES); triggers yield when exceeded
MAX_CHUNK_CHARS = 2_000_000

# Number of raw chunks grouped into one worker batch (~50K lines per batch)
CHUNKS_PER_BATCH = 10

# Max characters per sub-chunk for single-line file processing
SINGLE_LINE_CHAR_CHUNK = 200_000

# Byte read buffer for single-line file streaming
BYTE_BUF = 8 * 1024 * 1024

# Recognised text file extensions
TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".sql", ".md", ".html", ".htm"}
