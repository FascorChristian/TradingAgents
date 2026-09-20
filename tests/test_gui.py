"""Tests for the TradingAgents Python GUI application."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from cli.gui import ALL_AGENT_KEYS, TradingAgentsApp
from cli.main import app as typer_app
from typer.testing import CliRunner


@pytest.fixture
def gui_app():
    """Create a headless TradingAgentsApp instance for testing."""
    app = TradingAgentsApp()
    app.withdraw()  # Keep hidden during test
    yield app
    try:
        app.destroy()
    except Exception:
        pass


def test_gui_app_initialization(gui_app):
    assert "TradingAgents" in gui_app.title()
    assert gui_app.entry_ticker.get() == "SPY"
    assert gui_app.combo_lang.get() in ("English", "Spanish")
    assert gui_app.combo_provider.get() in ("google", "openai")
    assert len(gui_app.badge_labels) == len(ALL_AGENT_KEYS)
    assert str(gui_app.btn_start.cget("state")) == "normal"
    assert str(gui_app.btn_stop.cget("state")) == "disabled"


def test_gui_provider_change_updates_models(gui_app):
    # Select Google provider
    gui_app.combo_provider.set("google")
    gui_app._on_provider_change()
    google_quick = list(gui_app.combo_quick_model["values"])
    google_deep = list(gui_app.combo_deep_model["values"])
    assert any("gemini" in m for m in google_quick)
    assert any("gemini" in m for m in google_deep)

    # Select OpenAI provider
    gui_app.combo_provider.set("openai")
    gui_app._on_provider_change()
    openai_quick = list(gui_app.combo_quick_model["values"])
    openai_deep = list(gui_app.combo_deep_model["values"])
    assert any("gpt" in m for m in openai_quick)
    assert any("gpt" in m for m in openai_deep)


def test_gui_queue_processing(gui_app):
    # Put mock events
    gui_app.event_queue.put(("log", "Iniciando analisis de prueba...", "SYSTEM"))
    gui_app.event_queue.put(("progress", 45.0, "Ejecutando debate..."))
    gui_app.event_queue.put(("agent_status", "market", "completed"))
    gui_app.event_queue.put(("agent_status", "bull", "in_progress"))
    gui_app.event_queue.put(("stats", 8, 16, 24500))
    gui_app.event_queue.put(("report_section", "market_report", "### Informe de Mercado OHLCV"))
    gui_app.event_queue.put(("report_section", "final_trade_decision", "PROPUESTA FINAL: BUY / COMPRAR 50 ACCIONES"))

    gui_app._process_queue()

    assert gui_app.progress_var.get() == 45.0
    assert gui_app.lbl_status_text.cget("text") == "Ejecutando debate..."
    assert gui_app.lbl_metric_llm.cget("text") == "🤖 LLM Calls: 8"
    assert gui_app.lbl_metric_tools.cget("text") == "🛠️ Tools: 16"
    assert gui_app.lbl_metric_tokens.cget("text") == "📊 Tokens: 24,500"

    # Verify badge states
    _, market_lbl = gui_app.badge_labels["market"]
    assert "Completado" in market_lbl.cget("text")
    _, bull_lbl = gui_app.badge_labels["bull"]
    assert "En Ejecución" in bull_lbl.cget("text")

    # Verify final decision banner
    assert "COMPRAR (BUY)" in gui_app.lbl_decision_action.cget("text")
    assert "market_report" in gui_app.report_data


def test_gui_input_validation_empty_ticker(gui_app, monkeypatch):
    mock_error = MagicMock()
    monkeypatch.setattr("tkinter.messagebox.showerror", mock_error)

    gui_app.entry_ticker.delete(0, "end")
    gui_app._start_analysis()

    assert mock_error.called
    assert "ticker" in mock_error.call_args[0][1].lower()
    assert gui_app.is_running is False


def test_gui_input_validation_invalid_date(gui_app, monkeypatch):
    mock_error = MagicMock()
    monkeypatch.setattr("tkinter.messagebox.showerror", mock_error)

    gui_app.entry_date.delete(0, "end")
    gui_app.entry_date.insert(0, "not-a-valid-date")
    gui_app._start_analysis()

    assert mock_error.called
    assert "fecha" in mock_error.call_args[0][1].lower()
    assert gui_app.is_running is False


def test_cli_main_triggers_gui(monkeypatch):
    mock_launch = MagicMock()
    monkeypatch.setattr("cli.gui.launch_gui", mock_launch)

    # When invoking with --gui flag, it should call launch_gui
    result = CliRunner().invoke(typer_app, ["--gui"])
    assert result.exit_code == 0
    assert mock_launch.called


def test_gui_stats_callback_handler_compatible_with_google_client():
    import queue
    from cli.gui import GUIStatsCallbackHandler
    from tradingagents.llm_clients.google_client import NormalizedChatGoogleGenerativeAI

    q = queue.Queue()
    handler = GUIStatsCallbackHandler(q)
    # Must not raise Pydantic validation error for callbacks
    llm = NormalizedChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key="fake-key",
        callbacks=[handler],
    )
    assert llm.callbacks == [handler]
