import collections
import re

import jieba
from jieba.posseg import cut as pseg_cut
import jieba.posseg as pseg

_LONG_DIGITS_RE = re.compile(r"^\d{5,}$")
_LONG_ASCII_RE = re.compile(r"^[a-zA-Z0-9_]{12,}$")


def _is_garbage(word: str) -> bool:
    return bool(_LONG_DIGITS_RE.match(word) or _LONG_ASCII_RE.match(word))


def _strip_html(text: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    for tag in soup(['script', 'style', 'link', 'meta', 'head']):
        tag.decompose()
    return soup.get_text('\n', strip=True)


def _tokenize(text: str, cutter, use_pos: bool,
              counter: collections.Counter) -> None:
    for w in cutter(text, HMM=True):
        word = w if isinstance(w, str) else w.word
        word = word.strip()
        if not word or _is_garbage(word):
            continue
        key = (word, w.flag) if use_pos else word
        counter[key] += 1


def _cut_impl(source, use_pos: bool, strip_html: bool,
              counter: collections.Counter) -> None:
    cutter = pseg_cut if use_pos else jieba.cut
    if isinstance(source, list):
        if strip_html:
            source = [_strip_html('\n'.join(source))]
        for line in source:
            _tokenize(line, cutter, use_pos, counter)
    else:
        if strip_html:
            source = _strip_html(source)
        _tokenize(source, cutter, use_pos, counter)


def cut_and_count(lines: list[str], strip_html: bool = False) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    _cut_impl(lines, use_pos=False, strip_html=strip_html, counter=counter)
    return counter


def cut_and_count_pos(lines: list[str], strip_html: bool = False) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    _cut_impl(lines, use_pos=True, strip_html=strip_html, counter=counter)
    return counter


def cut_and_count_text(text: str, strip_html: bool = False) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    _cut_impl(text, use_pos=False, strip_html=strip_html, counter=counter)
    return counter


def cut_and_count_text_pos(text: str, strip_html: bool = False) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    _cut_impl(text, use_pos=True, strip_html=strip_html, counter=counter)
    return counter


def count_presegmented(lines: list[str]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for line in lines:
        for token in line.split("\t"):
            w = token.strip()
            if w and not _is_garbage(w):
                counter[w] += 1
    return counter


def count_presegmented_text(text: str) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for token in text.split("\t"):
        w = token.strip()
        if w and not _is_garbage(w):
            counter[w] += 1
    return counter


def count_presegmented_pos(lines: list[str]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    pos_dict = pseg.dt.word_tag_tab
    for line in lines:
        for token in line.split("\t"):
            w = token.strip()
            if not w or _is_garbage(w):
                continue
            pos = pos_dict.get(w, 'x')
            counter[(w, pos)] += 1
    return counter


def count_presegmented_text_pos(text: str) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    pos_dict = pseg.dt.word_tag_tab
    for token in text.split("\t"):
        w = token.strip()
        if not w or _is_garbage(w):
            continue
        pos = pos_dict.get(w, 'x')
        counter[(w, pos)] += 1
    return counter
