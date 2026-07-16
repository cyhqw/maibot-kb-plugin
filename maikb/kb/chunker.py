"""kb.chunker

Markdown 语义切分器 — 与 A_memorix 的滑动窗口硬切完全不同的设计。

设计原则：
1. 按 markdown 标题（# / ## / ### / ...）切分章节，保留标题层级路径
2. 章节内按段落（双换行）累积，达到目标大小时输出 chunk
3. 不做字符数硬切（除非单段落超过 max_size 上限）
4. 重叠只发生在段落边界，不会切断段落中间
5. 每个 chunk 携带 title_path（如 ["蒙德", "第二幕", "月宫与葬火"]）

支持 .md 和 .txt：
- .md 按标题层级切分
- .txt 没有标题，整体按段落切分
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable


# 标题正则：# / ## / ### ... (1-6 个 #)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)

# 纯文本文件的虚拟章节切分（按双换行分段，累积到 target_size 输出）
_DEFAULT_TARGET_CHARS = 500
_DEFAULT_MAX_CHARS = 1500
_DEFAULT_MIN_CHARS = 80  # 低于此长度的 chunk 不单独输出，合并到下一个


@dataclass
class Chunk:
    """切分产物。"""

    content: str
    title_path: list[str] = field(default_factory=list)
    heading: str = ""
    chunk_index: int = 0
    char_count: int = 0
    token_count: int = 0
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.char_count == 0:
            self.char_count = len(self.content)
        if self.token_count == 0:
            self.token_count = _estimate_tokens(self.content)
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    中文：每字约 1 token
    英文：每 4 字符约 1 token
    """

    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return chinese_chars + other_chars // 4


@dataclass
class _Section:
    """切分中间产物：一个标题下的所有内容。"""

    title_path: list[str]
    heading: str
    content: str


def _split_markdown_into_sections(text: str) -> list[_Section]:
    """按 # 标题切分 markdown，返回 sections。

    每个 section 包含 title_path（标题层级路径）和该标题下的正文。
    没有标题的内容归入一个空 title_path 的 section。

    示例：
        # 蒙德
        ## 第二幕 月宫与葬火
        内容...
        ## 第三幕 ...
        内容...

    → [
        _Section(title_path=["蒙德", "第二幕 月宫与葬火"], heading="第二幕 月宫与葬火", content="内容..."),
        _Section(title_path=["蒙德", "第三幕 ..."], heading="第三幕 ...", content="内容..."),
    ]
    """

    # 找到所有标题位置
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        # 没有标题，整体作为一个 section
        stripped = text.strip()
        if not stripped:
            return []
        return [_Section(title_path=[], heading="", content=stripped)]

    sections: list[_Section] = []

    # 标题前的内容（如果有）
    if matches[0].start() > 0:
        pre_content = text[: matches[0].start()].strip()
        if pre_content:
            sections.append(_Section(title_path=[], heading="", content=pre_content))

    # 维护当前标题层级栈
    title_stack: list[tuple[int, str]] = []  # [(level, title), ...]

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        # 弹出栈中 level >= 当前的标题
        while title_stack and title_stack[-1][0] >= level:
            title_stack.pop()
        title_stack.append((level, title))

        title_path = [t for _, t in title_stack]

        # section 内容：从当前标题行结束到下一个标题开始
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        if content:
            sections.append(
                _Section(
                    title_path=title_path,
                    heading=title,
                    content=content,
                )
            )

    return sections


def _chunk_section_by_paragraphs(
    section: _Section,
    target_chars: int,
    max_chars: int,
    min_chars: int,
) -> list[Chunk]:
    """在 section 内按段落累积切分。

    - 段落以双换行分隔
    - 累积到 >= target_chars 时输出一个 chunk
    - 单段落超过 max_chars 时硬切（带边界回退）
    - 短 chunk 合并到下一个
    """

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.content) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_chars = 0

    for para in paragraphs:
        # 如果单段落本身就超过 max_chars，先 flush buffer，再硬切段落
        if len(para) > max_chars:
            # flush 当前 buffer
            if buffer and buffer_chars >= min_chars:
                chunks.append(_build_chunk(section, "\n\n".join(buffer)))
                buffer = []
                buffer_chars = 0

            # 硬切大段落
            for sub in _hard_split(para, target_chars, max_chars):
                chunks.append(_build_chunk(section, sub))
            continue

        # 普通段落：加入 buffer
        buffer.append(para)
        buffer_chars += len(para) + 2  # +2 for \n\n

        # 达到目标大小，输出
        if buffer_chars >= target_chars:
            chunks.append(_build_chunk(section, "\n\n".join(buffer)))
            buffer = []
            buffer_chars = 0

    # 收尾
    if buffer:
        last_text = "\n\n".join(buffer)
        # 如果太短且已有上一个 chunk，合并到上一个
        if len(last_text) < min_chars and chunks:
            prev = chunks[-1]
            prev.content = prev.content + "\n\n" + last_text
            prev.char_count = len(prev.content)
            prev.token_count = _estimate_tokens(prev.content)
            prev.content_hash = hashlib.sha256(prev.content.encode("utf-8")).hexdigest()
        else:
            chunks.append(_build_chunk(section, last_text))

    return chunks


def _build_chunk(section: _Section, content: str) -> Chunk:
    return Chunk(
        content=content,
        title_path=list(section.title_path),
        heading=section.heading,
    )


def _hard_split(text: str, target_chars: int, max_chars: int) -> list[str]:
    """对超长段落做硬切，优先在句号/换行处切。"""

    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        # 在 target_chars 附近找最近的句号或换行
        cut = _find_best_cut(remaining, target_chars, max_chars)
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _find_best_cut(text: str, target: int, max_size: int) -> int:
    """在 [target, max_size] 区间找最佳切点（优先句号、换行）。"""

    # 候选切点：句号、感叹号、问号、换行
    candidates = []
    for delim in ["。", "！", "？", "…", ". ", "! ", "? ", "\n"]:
        idx = text.rfind(delim, target, max_size)
        if idx > 0:
            candidates.append(idx + len(delim))
    if candidates:
        return max(candidates)  # 取最大的，尽量切长一点
    # 找不到，硬切到 max_size
    return max_size


def chunk_markdown(
    text: str,
    *,
    target_chars: int = _DEFAULT_TARGET_CHARS,
    max_chars: int = _DEFAULT_MAX_CHARS,
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """切分 markdown 文本。

    Args:
        text: markdown 原文
        target_chars: 目标 chunk 字符数（实际会略大，按段落边界对齐）
        max_chars: 单 chunk 最大字符数（超过会硬切）
        min_chars: 最小 chunk 字符数（小于此值的 chunk 会合并到上一个）

    Returns:
        List[Chunk]，每个 chunk 已填好 chunk_index / char_count / token_count / content_hash
    """

    sections = _split_markdown_into_sections(text)
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(
            _chunk_section_by_paragraphs(section, target_chars, max_chars, min_chars)
        )

    # 重新编号 chunk_index
    for i, c in enumerate(chunks):
        c.chunk_index = i

    return chunks


def chunk_plain_text(
    text: str,
    *,
    target_chars: int = _DEFAULT_TARGET_CHARS,
    max_chars: int = _DEFAULT_MAX_CHARS,
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """切分纯文本（无 markdown 标题）。"""

    section = _Section(title_path=[], heading="", content=text.strip())
    chunks = _chunk_section_by_paragraphs(section, target_chars, max_chars, min_chars)
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks


def chunk_file(
    file_path: str,
    text: str,
    *,
    target_chars: int = _DEFAULT_TARGET_CHARS,
    max_chars: int = _DEFAULT_MAX_CHARS,
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """根据文件扩展名自动选择切分策略。"""

    if file_path.lower().endswith(".md") or file_path.lower().endswith(".markdown"):
        return chunk_markdown(text, target_chars=target_chars, max_chars=max_chars, min_chars=min_chars)
    return chunk_plain_text(text, target_chars=target_chars, max_chars=max_chars, min_chars=min_chars)


__all__ = [
    "Chunk",
    "chunk_markdown",
    "chunk_plain_text",
    "chunk_file",
]
