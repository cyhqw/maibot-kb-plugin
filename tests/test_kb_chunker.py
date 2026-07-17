"""tests.test_kb_chunker — 测试 markdown 语义切分"""

import pytest

from maikb.kb import chunk_markdown, chunk_plain_text


def test_basic_markdown_chunking():
    """按 # / ## 标题切分，保留标题路径。"""

    md = """# 蒙德

## 第二幕 月宫与葬火

法涅斯是原初之人，从蛋中诞生后创造了提瓦特世界的秩序与法则。

## 第三幕 高塔孤王

温迪推翻了高塔孤王迭卡拉庇安，解放了蒙德城的人民。
"""
    chunks = chunk_markdown(md, min_chars=0)
    assert len(chunks) == 2

    assert chunks[0].heading == "第二幕 月宫与葬火"
    assert chunks[0].title_path == ["蒙德", "第二幕 月宫与葬火"]
    assert "法涅斯" in chunks[0].content

    assert chunks[1].heading == "第三幕 高塔孤王"
    assert chunks[1].title_path == ["蒙德", "第三幕 高塔孤王"]
    assert "温迪" in chunks[1].content


def test_three_level_headings():
    """三级标题路径。"""

    md = """# 蒙德

## 第二幕

### 法涅斯的诞生

法涅斯从蛋中诞生，这是原初之人降临提瓦特的故事，他创造了世界的秩序与光暗法则。

### 法涅斯与龙族

法涅斯击败了七位龙王，统一了提瓦特大陆，这是创世之战的壮丽史诗。
"""
    chunks = chunk_markdown(md, min_chars=0)
    assert len(chunks) == 2

    assert chunks[0].heading == "法涅斯的诞生"
    assert chunks[0].title_path == ["蒙德", "第二幕", "法涅斯的诞生"]

    assert chunks[1].heading == "法涅斯与龙族"
    assert chunks[1].title_path == ["蒙德", "第二幕", "法涅斯与龙族"]


def test_pre_heading_content():
    """标题前的内容归入空 title_path section。"""

    md = """这是开头的引言，描述了提瓦特大陆的基本背景和世界观设定。

# 第一章

第一章内容，讲述了旅行者初次来到蒙德城时遇到的种种事件。
"""
    chunks = chunk_markdown(md, min_chars=0)
    assert len(chunks) == 2
    assert chunks[0].title_path == []
    assert "引言" in chunks[0].content
    assert chunks[1].title_path == ["第一章"]


def test_paragraph_accumulation():
    """多个段落累积到 target_chars 输出为一个 chunk。"""

    md = """# 章节

段落一，内容比较短。

段落二，内容也比较短。

段落三，内容同样短小。

段落四，内容继续短小。

段落五，依然短小。
"""
    chunks = chunk_markdown(md, target_chars=80, max_chars=500, min_chars=10)
    # 应该至少切出 1 个 chunk
    assert len(chunks) >= 1
    # 所有 chunk 都带章节路径
    for c in chunks:
        assert c.title_path == ["章节"]


def test_large_paragraph_hard_split():
    """超长段落按句号硬切。"""

    long_para = "这是第一句。" + "内容很长。" * 200 + "这是最后一句。"
    md = f"# 章节\n\n{long_para}"
    chunks = chunk_markdown(md, target_chars=100, max_chars=200, min_chars=10)
    # 应该被切成多个
    assert len(chunks) > 1
    # 总内容应该完整
    all_content = "".join(c.content for c in chunks)
    assert "这是第一句" in all_content
    assert "这是最后一句" in all_content


def test_short_chunk_merged_to_previous():
    """太短的尾部 chunk 合并到上一个。"""

    md = """# 章节

这是一个比较长的段落，足够达到目标大小，应该被独立切出来。

短。
"""
    chunks = chunk_markdown(md, target_chars=30, max_chars=500, min_chars=20, overlap_chars=0)
    # 短段落应该合并到第一个 chunk
    assert len(chunks) == 1
    assert "短" in chunks[0].content


def test_plain_text_no_headings():
    """纯文本无标题时整体作为一个 section。"""

    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = chunk_plain_text(text, target_chars=15, max_chars=200, min_chars=5)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.title_path == []
        assert c.heading == ""


def test_chunk_metadata_filled():
    """每个 chunk 应填好 chunk_index / char_count / token_count / content_hash。"""

    md = "# 标题\n\n内容一。\n\n内容二。"
    chunks = chunk_markdown(md)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.char_count == len(c.content)
        assert c.token_count > 0
        assert len(c.content_hash) == 64  # SHA256 hex


def test_chunk_content_hash_deterministic():
    """相同内容产生相同 hash。"""

    md = "# 标题\n\n内容。"
    chunks1 = chunk_markdown(md)
    chunks2 = chunk_markdown(md)
    assert chunks1[0].content_hash == chunks2[0].content_hash


def test_no_content_returns_empty():
    """空内容返回空列表。"""

    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  \n") == []


def test_overlap_between_chunks():
    """overlap_chars > 0 时相邻 chunk 应有共享段落。"""

    md = """# 测试

段落一，这是一段足够长的文字，需要超过目标字符数才能触发切分。

段落二，这是第二段，同样需要足够长才能触发切分。

段落三，这是第三段，确保能切出至少两个 chunk。
"""
    chunks = chunk_markdown(md, target_chars=30, max_chars=500, min_chars=5, overlap_chars=50)
    assert len(chunks) >= 2
    # 第二个 chunk 的开头应该包含第一个 chunk 末尾的段落
    # 段落二应该同时出现在 chunk[0] 和 chunk[1] 中
    if len(chunks) >= 2:
        # 找到在两个 chunk 中都出现的段落
        overlap_found = False
        for line in chunks[0].content.split("\n\n"):
            if line.strip() and line.strip() in chunks[1].content:
                overlap_found = True
                break
        assert overlap_found, "相邻 chunk 之间应有重叠段落"


def test_zero_overlap_no_sharing():
    """overlap_chars=0 时相邻 chunk 不共享内容。"""

    md = """# 测试

段落一，这是一段足够长的文字，需要超过目标字符数才能触发切分。

段落二，这是第二段，同样需要足够长才能触发切分。

段落三，这是第三段，确保能切出至少两个 chunk。
"""
    chunks = chunk_markdown(md, target_chars=30, max_chars=500, min_chars=5, overlap_chars=0)
    assert len(chunks) >= 2
    # 段落二不应同时出现在两个 chunk 中
    if len(chunks) >= 2:
        for line in chunks[0].content.split("\n\n"):
            if line.strip():
                assert line.strip() not in chunks[1].content, "overlap=0 时不应有共享段落"
