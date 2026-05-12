# dict-seg v1.3.0 — 批量中文分词 + 词频统计

基于 [jieba](https://github.com/fxsjy/jieba) 的分词工具，对语料批量分词并输出带词频的词典表。支持 20GB 多行文件、10GB 单行文件、200GB 文件夹。支持预分词文件的词频统计和词性标注。

## 安装

```bash
pip install -e .
# macOS 可选: brew install coreutils（获得 gsort，更好的排序性能）
```

## 快速开始

```bash
# 分词
dict-seg /path/to/corpus.txt                     # 单个文件
dict-seg /path/to/dir/                            # 文件夹（递归扫描 .txt/.json 等）
dict-seg /path/to/corpus.txt --pos                # 带词性标注
dict-seg /path/to/corpus.txt --strip-html         # 先去 HTML 标签
dict-seg /path/to/corpus.txt --user-dict my.dict  # 自定义词典
dict-seg /path/to/corpus.txt --min-freq 10 -w 16  # 调整参数
dict-seg /path/to/corpus.txt --force              # 覆盖已有输出
dict-seg --version                                # 查看版本

# 预分词文件词频统计（tab 分隔的已分词文件）
dict-seg /path/to/tokens.txt --pre-seg            # 直接计数，不走 jieba 分词
dict-seg /path/to/tokens.txt --pre-seg --pos      # 词性标注（词典查找，不切分）

# 合并多个词频文件
dict-seg-merge /path/to/wordfreq_dir/
dict-seg-merge /path/to/wordfreq_dir/ --pos --force
```

## 输出格式

| 模式 | 格式 | 后缀 |
|------|------|------|
| 默认 | `word\tfreq` | `_wordfreq.txt` |
| `--pos` | `word\tpos\tfreq` | `_pos_wordfreq.txt` |
| `--pre-seg` | `word\tfreq` | `_wordfreq.txt` |
| `--pre-seg --pos` | `word\tpos\tfreq` | `_pos_wordfreq.txt` |

POS 标签使用 jieba 的 ictclas 兼容标记集（n=名词, v=动词, r=代词 等）。预分词 POS 模式通过词典直接查找词性，词典未收录的标 `x`，不会重新切分 token。

文件夹分词时自动排除已有的 `*wordfreq*` 文件和已合并的 `merged*` 文件，避免循环处理。

## CLI 参数

### dict-seg

```
dict-seg [OPTIONS] INPUT_PATH
  -o, --output TEXT      输出路径（默认自动生成）
  -m, --mem INTEGER      sort 内存预算 MB  [default: 1024]
  -w, --workers INTEGER  worker 进程数  [default: cpu_count//2, cgroup-aware]
  --min-freq INTEGER     最低词频  [default: 5]
  --pos                  启用词性标注
  --strip-html           分词前用 BeautifulSoup 去 HTML 标签
  --user-dict TEXT       自定义 jieba 词典路径
  --pre-seg              输入为预分词文件（tab 分隔 token），直接计数
  --force                覆盖已有输出
  --version              显示版本
```

### dict-seg-merge

```
dict-seg-merge [OPTIONS] INPUT_DIR
  -o, --output TEXT      输出路径
  --pos                  输入为 3 列 POS 格式
  --min-freq INTEGER     最低词频
  --force                覆盖已有输出
  --version              显示版本
```

## 架构

三阶段流水线：

1. **文件准备** — 递归收集、编码检测（UTF-8→GB18030→GBK→Big5→charset_normalizer）、单行/超长行文件检测（平均行 >64KB 触发拆分）。编码检测或读取失败时自动跳过损坏文件
2. **并行分词+计数** — 5000 行/chunk、2M 字符上限/chunk，10 chunk/batch，Pool 多进程并行（macOS 用 `spawn` 避免死锁）。超长行自动切 200K 字符子块。四种模式：jieba 分词、预分词计数、预分词词性标注。每个 batch 带 600s 超时保护。超长行文件先拆分为 500MB 子文件再走正常流水线
3. **归并+输出** — 拼接 temp 文件 → 系统 sort（带进度监控）→ 单遍合并同词 → 中文过滤+频次过滤 → 按频降序输出

## 过滤设计

| 层 | 位置 | 规则 |
|----|------|------|
| 1 | worker 内 | `^\d{5,}$` + `^[a-zA-Z0-9_]{12,}$` — 不进 Counter |
| 2 | Stage 3 | `_has_chinese(w)` — 仅保留 `\u4E00-\u9FFF` 字符 |
| 3 | Stage 3 | `freq >= min_freq`（默认 5），非整数 freq 自动跳过 |

## 项目结构

```
dict_seg/
├── pyproject.toml
├── README.md
├── dict_seg/
│   ├── __init__.py     # API: run_pipeline, merge_wordfreq_files, count_presegmented, count_presegmented_pos 等
│   ├── __main__.py     # CLI: dict-seg + dict-seg-merge (click)
│   ├── config.py       # 配置常量 + cgroup-aware cpu_count
│   ├── segment.py      # jieba 分词 + 预分词计数 + 词性标注 + 垃圾过滤
│   └── pipeline.py     # 主流水线 + merge + 共享 helpers
└── tests/
```

## Python API

```python
from dict_seg import run_pipeline, merge_wordfreq_files
from dict_seg import cut_and_count, cut_and_count_pos
from dict_seg import count_presegmented, count_presegmented_pos

# 分词
run_pipeline("/path/to/corpus", min_freq=10, workers=8)
run_pipeline("/path/to/corpus", use_pos=True)                     # 带词性
run_pipeline("/path/to/corpus", strip_html=True, user_dict="my.dict")
run_pipeline("/path/to/tokens.txt", pre_seg=True)                  # 预分词计数
run_pipeline("/path/to/tokens.txt", pre_seg=True, use_pos=True)    # 预分词词性标注

# 合并已有的词频文件
merge_wordfreq_files("/path/to/wordfreq_dir/")
merge_wordfreq_files("/path/to/wordfreq_dir/", use_pos=True)

# Worker 函数
counter = cut_and_count(["我来到北京清华大学"])
counter = cut_and_count_pos(["我来到北京清华大学"])
counter = count_presegmented(["品牌\t颜色\t材质"])                  # tab 分隔 token 计数
counter = count_presegmented_pos(["品牌\t颜色\t材质"])              # 词典查找词性
```

## 配置常量

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `OUTPUT_FILE_SUFFIX` | `"wordfreq.txt"` | 默认输出后缀 |
| `OUTPUT_FILE_SUFFIX_POS` | `"pos_wordfreq.txt"` | POS 输出后缀 |
| `WORKERS` | `cpu_count()//2` | worker 数（cgroup-aware） |
| `MIN_FREQ` | 5 | 最低词频 |
| `CHUNK_LINES` | 5000 | 每 chunk 行数 |
| `MAX_CHUNK_CHARS` | 2,000,000 | 每 chunk 最大字符数 |
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

73 个用例覆盖 segment.py、pipeline.py、merge、CLI、预分词、超长行处理、集成测试。

## v1.2.0 更新

- `--pre-seg` 预分词文件词频统计（直接 tab 切分计数，不走 jieba，极快）
- `--pre-seg --pos` 预分词词性标注（词典直接查找，不切分 token，未知标 `x`）
- `count_presegmented` / `count_presegmented_pos` Python API
- `--user-dict` 支持自定义 jieba 词典
- `--version` CLI 选项
- `use_pos` 参数命名（替换 `no_pos` 双否定）
- cgroup-aware `cpu_count()`（容器环境自动限并发）
- worker `.get()` 600s 超时保护
- `_is_single_line_file` 超长行检测（平均行 >64KB 自动触发拆分）
- `_split_file_to_dir` 大文件拆分为 500MB 子文件后走正常流水线
- `_read_chunks_by_lines` 字符限流（2M 字符上限）+ 超长行自动切 200K 子块
- 进度条行数估算基于前 64KB 换行符采样
- 损坏文件/编码检测失败自动跳过（非崩溃）
- 过滤阶段 `int(freq_s)` 容错（跳过非整数行）
- 提取共享 helpers：`_filter_and_sort_final`、`_monitor_sort_progress`
- `merge_wordfreq_files` 增加排序进度监控
- POS 合并 guard `pos_counts` 全覆盖
- `--strip-html` 用 `\n` 拼接行保留边界
- 删除死代码 `_process_single_line_file`（~72 行）

## 修改指南

- **分词参数**：`segment.py::_tokenize` 中的 `cutter(text, HMM=True)`
- **预分词处理**：`segment.py::count_presegmented` / `count_presegmented_pos`
- **过滤规则**：`segment.py:_is_garbage()` 正则 / `pipeline.py:_CHINESE_RE` 范围 / `--min-freq`
- **文件类型**：`config.py:TEXT_EXTENSIONS`
- **并行参数**：`config.py:CHUNK_LINES` / `CHUNKS_PER_BATCH`
- **CLI 选项**：`__main__.py` 添加 `@click.option`
