from multiprocessing import cpu_count

# Output file suffix appended as {name}_{date}_{suffix}
OUTPUT_FILE_SUFFIX = "wordfreq.txt"
OUTPUT_FILE_SUFFIX_POS = "pos_wordfreq.txt"

# Memory budget (MB) for external sort subprocess
DEFAULT_MEM_MB = 1024

# Minimum frequency for final output
MIN_FREQ = 5

# Worker processes (default: half of CPU cores)
WORKERS = max(1, cpu_count() // 2)

# Lines per raw chunk read from file (~500KB at 100 bytes/line)
CHUNK_LINES = 5_000

# Number of raw chunks grouped into one worker batch (~50K lines per batch)
CHUNKS_PER_BATCH = 10

# Max characters per sub-chunk for single-line file processing
SINGLE_LINE_CHAR_CHUNK = 200_000

# Byte read buffer for single-line file streaming
BYTE_BUF = 8 * 1024 * 1024

# Recognised text file extensions
TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".sql", ".md", ".html", ".htm"}
