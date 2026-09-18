from unittest import mock
import pytest

from cli import utils
from tradingagents.llm_clients.model_catalog import get_model_options


@pytest.mark.unit
class TestDeepModelOptions:
    def test_include_quick_models_for_google(self):
        deep_only = [v for _, v in get_model_options("google", "deep", include_quick=False)]
        with_quick = [v for _, v in get_model_options("google", "deep", include_quick=True)]
        assert "gemini-3.1-flash-lite" not in deep_only
        assert "gemini-3.1-flash-lite" in with_quick
        assert "gemini-3.5-flash" in with_quick
        assert "gemini-3.1-pro-preview" in with_quick

    def test_include_quick_models_for_openai(self):
        with_quick = [v for _, v in get_model_options("openai", "deep", include_quick=True)]
        assert "gpt-5.6-luna" in with_quick
        assert "gpt-5.4-mini" in with_quick
        assert "gpt-5.6" in with_quick

    def test_select_deep_thinking_agent_presents_quick_models(self):
        captured = {}

        def fake_select(message, **kwargs):
            captured["choices"] = [c.value for c in kwargs["choices"]]
            return mock.Mock(ask=mock.Mock(return_value="gemini-3.1-flash-lite"))

        with mock.patch.object(utils.questionary, "select", side_effect=fake_select):
            selected = utils.select_deep_thinking_agent("google", quick_model="gemini-3.1-flash-lite")

        assert selected == "gemini-3.1-flash-lite"
        assert "gemini-3.1-flash-lite" in captured["choices"]
        assert "gemini-3.5-flash" in captured["choices"]
        assert "gemini-3.1-pro-preview" in captured["choices"]

    def test_select_deep_thinking_agent_presents_custom_quick_model(self):
        captured = {}

        def fake_select(message, **kwargs):
            captured["choices"] = [c.value for c in kwargs["choices"]]
            return mock.Mock(ask=mock.Mock(return_value="my-special-model"))

        with mock.patch.object(utils.questionary, "select", side_effect=fake_select):
            selected = utils.select_deep_thinking_agent("google", quick_model="my-special-model")

        assert selected == "my-special-model"
        assert "my-special-model" in captured["choices"]

