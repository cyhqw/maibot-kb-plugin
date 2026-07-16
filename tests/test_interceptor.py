"""tests.test_interceptor — 前缀拦截器测试"""

import pytest

from maikb.interceptor import (
    DEFAULT_PREFIXES,
    extract_message_text,
    should_block,
)


def test_should_block_default_prefixes():
    """默认前缀 / [ # 命中。"""

    assert should_block("/help", DEFAULT_PREFIXES) == (True, "/")
    assert should_block("[quote]", DEFAULT_PREFIXES) == (True, "[")
    assert should_block("#tag", DEFAULT_PREFIXES) == (True, "#")
    assert should_block("hello", DEFAULT_PREFIXES) == (False, "")
    assert should_block("", DEFAULT_PREFIXES) == (False, "")
    assert should_block("   ", DEFAULT_PREFIXES) == (False, "")


def test_should_block_strip_leading_whitespace():
    """前导空格也应被识别。"""

    assert should_block("  /help", DEFAULT_PREFIXES) == (True, "/")
    assert should_block("\t[quote]", DEFAULT_PREFIXES) == (True, "[")


def test_should_block_custom_prefixes():
    """自定义前缀列表。"""

    prefixes = ["!", ".", "!!"]
    assert should_block("!ping", prefixes) == (True, "!")
    assert should_block(".help", prefixes) == (True, ".")
    assert should_block("!!important", prefixes) == (True, "!!")
    assert should_block("/cmd", prefixes) == (False, "")  # / 不在自定义列表


def test_extract_message_text_dict():
    """从 dict 形式的 message 提取文本。"""

    assert extract_message_text({"processed_plain_text": "/help"}) == "/help"
    assert extract_message_text({"plain_text": "hello"}) == "hello"
    assert extract_message_text({"raw_message": "world"}) == "world"
    assert extract_message_text({"content": "test"}) == "test"
    assert extract_message_text({}) == ""
    assert extract_message_text({"other": "value"}) == ""


def test_extract_message_text_object():
    """从对象形式的 message 提取文本。"""

    class FakeMsg:
        processed_plain_text = "/help me"

    assert extract_message_text(FakeMsg()) == "/help me"


def test_extract_message_text_none():
    assert extract_message_text(None) == ""


def test_interceptor_mixin_class():
    """InterceptorMixin 类可以被继承。"""

    from maikb.interceptor import InterceptorMixin

    class FakePlugin(InterceptorMixin):
        pass

    assert hasattr(FakePlugin, "hook_prefix_guard")


def test_hook_handler_decorator_applied():
    """hook_prefix_guard 方法被 @HookHandler 装饰。"""

    from maikb.interceptor import InterceptorMixin
    from maibot_sdk.components import _COMPONENT_INFO_ATTR, HookHandlerComponentInfo

    # 检查方法上有 HookHandler 装饰器元数据
    info = getattr(InterceptorMixin.hook_prefix_guard, _COMPONENT_INFO_ATTR, None)
    assert info is not None
    # 应该是 HookHandlerComponentInfo 类型
    print(f"Hook info type: {type(info).__name__}")
    print(f"Hook name: {getattr(info, 'name', None)}")
    print(f"Hook: {getattr(info, 'hook', None)}")
