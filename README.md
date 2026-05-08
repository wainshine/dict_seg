# dict-seg v1.0.0 — 批量中文分词 + 词频统计

基于 [jieba](https://github.com/fxsjy/jieba) 的批量中文分词工具，对原始语料进行分词并输出带词频的词典表。支持 20GB 多行文件、10GB 单行文件、200GB 文件夹。

---

## 安装

```bash
pip install -e .
```

macOS 用户可选安装 GNU coreutils 以获得更好的排序性能（自动回退系统 `sort`）：

```bash
brew install coreutils   # 可选，获得 gsort
```

---

## 快速开始

```bash
# 单个文件
dict-seg /path/to/corpus.txt

# 文件夹（递归扫描 .txt/.json/.csv 等）
dict-seg /path/to/dir/

# 自定义输出路径
dict-seg /path/to/corpus.txt -o output.txt

# 带词性标注
dict-seg /path/to/corpus.txt --pos

# HTML 语料先去标签再分词
dict-seg /path/to/weixin.html --strip-html

# 调整最低词频和并行度
dict-seg /path/to/corpus.txt --min-freq 10 --workers 16

# 强制覆盖已有输出
dict-seg /path/to/corpus.txt --force
```

---

## CLI 参数

```
Usage: dict-seg [OPTIONS] INPUT_PATH

Options:
  -o, --output TEXT    输出路径（默认自动生成）
  -m, --mem INTEGER    sort 内存预算 MB  [default: 1024]
  -w, --workers INTEGER  worker 进程数  [default: cpu_count//2]
  --min-freq INTEGER   最低词频  [default: 5]
  --pos                启用词性标注 (word\tfreq\tpos)
  --strip-html         分词前用 BeautifulSoup 去除 HTML 标签
  --force              覆盖已存在的输出文件
  --help               显示帮助信息
```

---

## 输出格式

### 默认（两列，按词频降序）

```
word\tfreq
```

### `--pos`（三列，按词频降序）

```
word\tfreq\tpos
```

POS 标签使用 jieba 的 ictclas 兼容标记集（n=名词, v=动词, uj=的, ul=了 等）。

---

## 项目结构

```
dict_seg/
├── pyproject.toml          # 依赖、构建、entry point
├── README.md
├── .gitignore
├── dict_seg/
│   ├── __init__.py          # 公开 API: run_pipeline, cut_and_count, cut_and_count_pos
│   ├── __main__.py           # CLI 入口 (click)
│   ├── config.py             # 配置常量
│   ├── segment.py            # 分词 worker：jieba 调用 + 过滤 + Counter
│   └── pipeline.py           # 主流程：文件收集 → 并行分词 → 归并 → 输出
└── tests/
```

---

## 架构：三阶段流水线

```
输入路径
  │
  ├─ 阶段 0: 文件准备
  │   ├─ _collect_files()          递归扫描，过滤扩展名
  │   ├─ _detect_file_encoding()   五步编码检测 (UTF-8→GB18030→GBK→Big5→charset_normalizer)
  │   └─ _is_single_line_file()    检测 >100MB 且 <10 个换行符的单行文件
  │
  ├─ 阶段 1: 并行分词 + 计数
  │   │
  │   ├─ 多行文件:
  │   │   _read_chunks_by_lines(file, 5000行/chunk)
  │   │     → 每攒够 10 个 chunk (5万行) 提交一个 batch
  │   │     → pool.apply_async(_segment_worker)
  │   │         ├─ [可选] BeautifulSoup 去 HTML (--strip-html)
  │   │         ├─ jieba.cut(line, HMM=True)
  │   │         │    for token: strip → _is_garbage → Counter
  │   │         └─ _write_counter_to_file → chunk_XXXXXX.txt
  │   │
  │   └─ 单行文件:
  │       _process_single_line_file()
  │         8MB 字节块 → 多字节边界修正 → 每 20万字符一段 → 分词计数
  │
  ├─ 阶段 2: 归并
  │   ├─ _concat_temp_files        拼接所有 chunk_*.txt
  │   ├─ sort -t'\t' -k1,1         按词排序 (LC_ALL=C, 自动适配 GNU/BSD)
  │   └─ _merge_sorted_word_counts 单遍扫描合并同词频次
  │
  └─ 阶段 3: 过滤 + 输出
      ├─ _has_chinese(w)           只保留含中文的词 (CJK 基本+扩展A+兼容区)
      ├─ freq >= min_freq          最小词频过滤
      └─ sort -t'\t' -k2,2 -nr     按词频降序 → 最终文件
```

---

## 三层过滤设计

| 层 | 位置 | 何时 | 过滤规则 | 影响 |
|----|------|------|----------|------|
| **1** | `segment.py` worker 内 | 分词后、计数前 | `^\d{5,}$`（纯数字 5 位+）<br>`^[a-zA-Z0-9_]{12,}$`（纯 ASCII 12 位+） | 滤掉长数字串/长 ASCII 串 |
| **2** | `pipeline.py` Stage 3 | 所有 chunk 合并后 | `_has_chinese(w)` — 仅保留含 CJK 字符的词 | 去掉纯英文/数字/标点 |
| **3** | `pipeline.py` Stage 3 | 同 Stage 3 | `freq >= min_freq`（默认 5） | 去掉低频噪声 |

**设计要点：**
- 第 1 层在 worker 内尽早过滤，被滤掉的词不进入 Counter（不进 hash 表），节省内存和计算
- 第 2 层在合并后统一过滤，保证最终输出词表只含中文字符
- 第 3 层灵活可配（`--min-freq`）
- CJK 检测范围覆盖基本区（`\u4E00-\u9FFF`）、扩展A（`\u3400-\u4DBF`）、兼容区（`\uF900-\uFAFF`）

---

## 模块职责

### `config.py` — 配置常量

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `OUTPUT_FILE_SUFFIX` | `"wordfreq.txt"` | 输出文件后缀 |
| `DEFAULT_MEM_MB` | 1024 | sort 内存预算 (MB)，仅 GNU sort 生效 |
| `MIN_FREQ` | 5 | 最终输出最低词频 |
| `WORKERS` | `cpu_count()//2` | worker 进程数 |
| `CHUNK_LINES` | 5000 | 每次从文件读取的行数 |
| `CHUNKS_PER_BATCH` | 10 | 合并多少个 chunk 为一个 batch 提交给 worker |
| `SINGLE_LINE_CHAR_CHUNK` | 200000 | 单行文件每段最大字符数 |
| `BYTE_BUF` | 8MB | 单行文件二进制读缓冲 |
| `TEXT_EXTENSIONS` | `.txt,.csv,.json,...` | 识别的文本扩展名 |

#### 并行度调优

- `CHUNK_LINES × CHUNKS_PER_BATCH = 50,000` 行/batch
- batch 数量 = 总行数 ÷ 50,000
- 实际并发 worker 数 = min(WORKERS, batch 数量)
- 背压上限 `max_pending = WORKERS × 4`
- 使用 `collections.deque` 管理 futures，O(1) pop

若希望更充分利用多核，可减小 `CHUNK_LINES` 或 `CHUNKS_PER_BATCH`；若临时文件过多，可增大。

### `segment.py` — 分词 Worker

**公开函数：**

| 函数 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `cut_and_count(lines)` | `list[str]` | `Counter[str]` | 无 POS 模式 worker |
| `cut_and_count_pos(lines)` | `list[str]` | `Counter[(word,pos)]` | POS 模式 worker |
| `cut_and_count_text(text)` | `str` | `Counter[str]` | 单行文件无 POS |
| `cut_and_count_text_pos(text)` | `str` | `Counter[(word,pos)]` | 单行文件 POS |

所有四个公开函数共享 `_cut_impl` 内部实现。`strip_html` 参数启用时：
- 多行模式：`''.join(lines)` 整批交 BeautifulSoup → `decompose(script/style/link/meta/head)` → `get_text`
- 文本模式：整段交 BeautifulSoup 处理后分词

**垃圾过滤规则（`_is_garbage`）：**
- `^\d{5,}$` — 整个词是 5 位以上纯数字（如 `35190107271`）
- `^[a-zA-Z0-9_]{12,}$` — 整个词是 12 位以上纯 ASCII（如 `T1Z0eVXy8iXXXXXXXX`）

两个正则都用 `match`（整词匹配），行为对称。

### `pipeline.py` — 主流程

**公开函数：**
- `run_pipeline(input_path, output_path, mem_mb, workers, min_freq, no_pos, strip_html, force)` → `str`（输出路径）

**关键实现细节：**

| 功能 | 实现 |
|------|------|
| Sort 跨平台 | 优选 `gsort`，回退 BSD `sort`，仅 GNU 时传 `-S`/`--parallel` |
| Sort 进程清理 | Ctrl+C 时 `_kill_running_sorts()` 终止所有子进程 |
| 输出保护 | 默认不覆盖已有输出，需 `--force` |
| 并行安全性 | `set_start_method("fork")`，兼容 spawn 环境 |
| Futures 管理 | `collections.deque`，O(1) pop |
| 编码检测 | CJK 比率用 `len(text)` 字符数（非 bytes）做分母 |
| 合并防护 | `len(parts)` 检查防 IndexError |
| TSV 转义 | 写文件时 `\t`/`\n`/`\r` 替换为空格 |

### `__main__.py` — CLI

全部参数见上方 CLI 参考。

### `__init__.py` — 公开 API

```python
from dict_seg import run_pipeline, cut_and_count, cut_and_count_pos
```

可作为 Python 库直接调用：

```python
from dict_seg import run_pipeline
run_pipeline("/path/to/corpus", min_freq=10, workers=8)
```

---

## 编码检测策略

`_detect_file_encoding` 采用五步检测：

1. 尝试严格 UTF-8 解码，若无 `\ufffd` 且成功 → UTF-8
2. 若 `UnicodeDecodeError`，用 `errors="replace"` 解码，CJK 字符占比 > 0.5%（按字符数） → UTF-8（受损）
3. 试 GB18030 → GBK → BIG5，取 CJK 字符占比最高者
4. 若仍不明确 → `charset_normalizer.from_path()` 兜底
5. 所有方法失败 → 回退 `"utf-8"`

**注意：** CJK 范围覆盖基本区（`\u4E00-\u9FFF`）、扩展A（`\u3400-\u4DBF`）、兼容区（`\uF900-\uFAFF`）。

---

## 单行超大文件处理

当文件 >100MB 且换行数 <10 时，走字节级分流：

```
8MB 字节块读取 → UnicodeDecodeError 多字节边界修正（退 1-4 字节）
  → 每 200K 字符一段 → jieba.cut → Counter → chunk_*.txt
```

关键：carryover 缓冲区保证多字节字符（UTF-8 1-4B, GB18030 1-4B）不被读边界截断。

---

## 依赖

| 包 | 用途 |
|----|------|
| `jieba >= 0.42` | 中文分词引擎 |
| `click >= 8.0` | CLI 参数解析 |
| `tqdm >= 4.0` | 进度条 |
| `beautifulsoup4 >= 4.12` | HTML 标签去除（`--strip-html`） |
| `charset-normalizer >= 3.0` | 编码检测兜底 |
| `pytest >= 8.0` (dev) | 测试框架 |

---

## 设计决策

1. **jieba 黑盒原则：** 分词完全交给 jieba，本项目只做 I/O、并行、归并、输出。不干预 jieba 内部。
2. **三层过滤：** 层1 在 worker 尽早滤掉长数字/长 ASCII 垃圾，减少 Counter 体积；层2-3 在合并后统一过，保证输出质量。
3. **不使用 jieba 内置并行：** 避免与我们自己的 Pool 并行框架冲突。每个 worker 进程独立加载 jieba 模型。使用 `fork` 模式确保进程继承主进程状态。
4. **外部排序：** 使用系统 `sort`（C 代码级排序，比 Python 快 10x+），自动适配 GNU/BSD。GNU 时传 `-S`/`--parallel` 优化内存和并行度。
5. **临时文件隔离：** 各阶段用子目录，`finally` 统一清理 `shutil.rmtree(ignore_errors=True)`，同时 `_kill_running_sorts()` 终止未完成子进程。
6. **默认无 POS：** 词性标注速度慢 ~6x，默认关闭。
7. **`--strip-html`：** 可选功能，使用 BeautifulSoup 在分词前去 HTML 标签。处理微信/网页语料时必备。

---

## 已知局限

1. **BSD sort：** 系统 `sort` 不支持 `-S`/`--parallel`，仅 GNU sort（`gsort`）获得内存和并行优化。功能正常，仅性能略有差异。
2. **单行文件分块：** `SINGLE_LINE_CHAR_CHUNK` 分块可能切断多字符词，导致分词不完整。对大多数中文词汇影响可忽略。
3. **Surrogate pairs：** 极少见的超出 BMP 的 CJK 字符（如 Extension B+）在字符分块时可能被截断。
4. **进度条估算：** 进度条的总 chunk 估算基于 `file_size // 100` 假设平均行长 100 字节。对超长行文件（如 JSON/HTML）估算严重失准，但仅影响进度条显示，不影响结果。


---

## 测试

```bash
python3 -m pytest tests/ -v
```

覆盖范围（39 个用例）：

| 模块 | 覆盖内容 |
|------|----------|
| `segment.py` | `_is_garbage` 数字/ASCII/中文边界，`cut_and_count` 基本分词+垃圾过滤+HTML剥离，`cut_and_count_pos` POS 模式 |
| `pipeline.py` | `_collect_files` 文件/目录/空/不存在，`_has_chinese` 中文/混排/纯英文，`_make_output_path` 文件/目录，`_detect_file_encoding` UTF-8/BOM，`_is_single_line_file` 小文件/短文件，`_merge_sorted_word_counts` 基本合并/POS/空输入/空行/坏行，`_write_counter_to_file` 普通/POS/转义，`_make_sort_cmd` GNU/BSD |
| 集成测试 | 完整流水线、`min_freq` 过滤、`--force` 覆盖保护 |


---

## 修改指南

### 调整分词参数

修改 `segment.py::_tokenize` 中 `cutter(text, HMM=True)` 的 `HMM` 参数。

### 调整过滤规则

**层1（垃圾过滤）：** 修改 `segment.py:_is_garbage()` 中的正则。当前使用 `match`（整词），如需子串匹配改为 `search`。

**层2（中文过滤）：** 修改 `pipeline.py:_CHINESE_RE` 的 Unicode 范围。

**层3（频次过滤）：** CLI 传 `--min-freq` 或修改 `config.py:MIN_FREQ`。

### 添加 HTML 处理能力

`--strip-html` 默认关闭。启用时 BeautifulSoup 去除 `<script>/<style>/<link>/<meta>/<head>` 标签。多行 batch 模式会先 `''.join(lines)` 再交 BS4 解析，确保跨行标签完整。

### 添加新文件类型

修改 `config.py:TEXT_EXTENSIONS` 集合。

### 修改输出格式

修改 `pipeline.py:_merge_sorted_word_counts()` 的写出行格式，以及 Stage 3 的 sort 命令。

### 修改并行参数

见 `config.py` "并行度调优"说明。

### 添加新 CLI 选项

在 `__main__.py` 中添加 `@click.option`，并在 `run_pipeline` 签名中增加对应参数。
