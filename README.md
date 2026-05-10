# dict-seg v1.1.0 — 批量中文分词 + 词频统计

基于 [jieba](https://github.com/fxsjy/jieba) 的分词工具，对语料批量分词并输出带词频的词典表。支持 20GB 多行文件、10GB 单行文件、200GB 文件夹。

## 安装

```bash
pip install -e .
# macOS 可选: brew install coreutils（获得 gsort，更好的排序性能）
```

## 快速开始

```bash
dict-seg /path/to/corpus.txt           # 单个文件
dict-seg /path/to/dir/                  # 文件夹（递归扫描 .txt/.json 等）
dict-seg /path/to/corpus.txt --pos      # 带词性标注
dict-seg /path/to/corpus.txt --strip-html  # 先去 HTML 标签
dict-seg /path/to/corpus.txt --min-freq 10 -w 16  # 调整参数
dict-seg /path/to/corpus.txt --force    # 覆盖已有输出

# 合并多个词频文件
dict-seg-merge /path/to/wordfreq_dir/
dict-seg-merge /path/to/wordfreq_dir/ --pos
```

## 输出格式

| 模式 | 格式 | 后缀 |
|------|------|------|
| 默认 | `word\tfreq` | `_wordfreq.txt` |
| `--pos` | `word\tpos\tfreq` | `_pos_wordfreq.txt` |

POS 标签使用 jieba 的 ictclas 兼容标记集（n=名词, v=动词, uj=的, ul=了 等）。

文件夹分词时自动排除已有的 `*wordfreq*.txt` 文件，避免循环处理。

## CLI 参数

```
dict-seg [OPTIONS] INPUT_PATH
  -o, --output TEXT     输出路径（默认自动生成）
  -m, --mem INTEGER     sort 内存预算 MB  [default: 1024]
  -w, --workers INTEGER  worker 进程数  [default: cpu_count//2]
  --min-freq INTEGER    最低词频  [default: 5]
  --pos                 启用词性标注
  --strip-html          分词前用 BeautifulSoup 去 HTML 标签
  --force               覆盖已有输出

dict-seg-merge [OPTIONS] INPUT_DIR
  -o, --output TEXT     输出路径
  --pos                 输入为 3 列 POS 格式
  --min-freq INTEGER    最低词频
  --force               覆盖已有输出
```

## 架构

三阶段流水线：

1. **文件准备** — 递归收集、编码检测（UTF-8→GB18030→GBK→Big5→charset_normalizer）、单行大文件检测
2. **并行分词+计数** — 5000 行/chunk，10 chunk/batch，Pool 多进程并行。单行文件走字节级分流
3. **归并+输出** — 拼接 temp 文件 → 系统 sort → 单遍合并同词 → 中文过滤+频次过滤 → 按频降序输出

文件之间用 `spawn` 多进程（macOS 上避免 `fork()` 与 `jieba.posseg` 死锁）。临时文件用 `finally` 统一清理，Ctrl+C 时终止子 sort 进程。

## 过滤设计

| 层 | 位置 | 规则 |
|----|------|------|
| 1 | worker 内 | `^\d{5,}$` + `^[a-zA-Z0-9_]{12,}$` — 不进 Counter |
| 2 | Stage 3 | `_has_chinese(w)` — 仅保留 `\u4E00-\u9FFF` 字符 |
| 3 | Stage 3 | `freq >= min_freq`（默认 5） |

## 项目结构

```
dict_seg/
├── pyproject.toml
├── README.md
├── dict_seg/
│   ├── __init__.py     # API: run_pipeline, merge_wordfreq_files, cut_and_count, cut_and_count_pos
│   ├── __main__.py     # CLI: dict-seg + dict-seg-merge
│   ├── config.py       # 配置常量
│   ├── segment.py      # jieba 分词 + 垃圾过滤 + Counter
│   └── pipeline.py     # 主流水线 + 合并功能
└── tests/
```

## Python API

```python
from dict_seg import run_pipeline, merge_wordfreq_files, cut_and_count, cut_and_count_pos

# 分词
run_pipeline("/path/to/corpus", min_freq=10, workers=8)
run_pipeline("/path/to/corpus", no_pos=False)  # 带词性

# 合并已有的词频文件
merge_wordfreq_files("/path/to/wordfreq_dir/")
merge_wordfreq_files("/path/to/wordfreq_dir/", no_pos=False)  # POS 合并

# Worker 函数
counter = cut_and_count(["我来到北京清华大学"])
counter = cut_and_count_pos(["我来到北京清华大学"])  # 带 POS
```

## 配置常量

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `OUTPUT_FILE_SUFFIX` | `"wordfreq.txt"` | 默认输出后缀 |
| `OUTPUT_FILE_SUFFIX_POS` | `"pos_wordfreq.txt"` | POS 输出后缀 |
| `WORKERS` | `cpu_count()//2` | worker 数 |
| `MIN_FREQ` | 5 | 最低词频 |
| `CHUNK_LINES` | 5000 | 每 chunk 行数 |
| `CHUNKS_PER_BATCH` | 10 | 每 batch chunk 数 |

## 依赖

| 包 | 用途 |
|----|------|
| `jieba >= 0.42` | 分词引擎 |
| `click >= 8.0` | CLI |
| `tqdm >= 4.0` | 进度条 |
| `beautifulsoup4 >= 4.12` | HTML 去标签 |
| `charset-normalizer >= 3.0` | 编码检测 |
| `pytest >= 8.0` (dev) | 测试框架 |

## 测试

```bash
python3 -m pytest tests/ -v
```

45 个用例覆盖 segment.py、pipeline.py、merge、集成测试。

## 已知局限

- POS 模式约比无 POS 慢 6x（jieba 词性标注开销）
- 进度条基于 `file_size // 100` 估算，超长行文件显示不准但不影响结果
- `--strip-html` 会 `''.join(lines)` 整批交 BeautifulSoup，超大 batch 时可能内存偏高

## 修改指南

- **分词参数**：`segment.py::_tokenize` 中的 `cutter(text, HMM=True)`
- **过滤规则**：`segment.py:_is_garbage()` 正则 / `pipeline.py:_CHINESE_RE` 范围 / `--min-freq`
- **文件类型**：`config.py:TEXT_EXTENSIONS`
- **并行参数**：`config.py:CHUNK_LINES` / `CHUNKS_PER_BATCH`
