"""Graphical User Interface (GUI) for TradingAgents.

Provides a desktop window to configure parameters, monitor multi-agent
execution in real time, view agent statuses, stream logs (including model
fallbacks), and inspect generated financial reports and final portfolio decisions.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List, Optional

from cli.models import AnalystType, AssetType
from cli.stats_handler import StatsCallbackHandler
from cli.utils import detect_asset_type, normalize_ticker_symbol
from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS, get_model_options

logger = logging.getLogger(__name__)

# Agent display mappings
ANALYST_KEYS = ["market", "social", "news", "fundamentals"]
ANALYST_DISPLAY_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_KEYS = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}

ALL_AGENT_KEYS = [
    "market",
    "social",
    "news",
    "fundamentals",
    "bull",
    "bear",
    "research_manager",
    "trader",
    "risk_debate",
    "portfolio_manager",
]

AGENT_LABELS = {
    "market": "1. Market Analyst",
    "social": "2. Sentiment Analyst",
    "news": "3. News Analyst",
    "fundamentals": "4. Fundamentals",
    "bull": "5. Bull Researcher",
    "bear": "6. Bear Researcher",
    "research_manager": "7. Research Mgr",
    "trader": "8. Trader",
    "risk_debate": "9. Risk Committee",
    "portfolio_manager": "10. Portfolio Mgr",
}


class GUIStatsCallbackHandler(StatsCallbackHandler):
    """Callback handler that sends live metric updates to the GUI queue."""

    def __init__(self, eq: queue.Queue):
        super().__init__()
        self.eq = eq

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        super().on_llm_start(serialized, prompts, **kwargs)
        self.eq.put(("stats", self.llm_calls, self.tool_calls, self.tokens_in + self.tokens_out))

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        super().on_chat_model_start(serialized, messages, **kwargs)
        self.eq.put(("stats", self.llm_calls, self.tool_calls, self.tokens_in + self.tokens_out))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        super().on_llm_end(response, **kwargs)
        self.eq.put(("stats", self.llm_calls, self.tool_calls, self.tokens_in + self.tokens_out))

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        super().on_tool_start(serialized, input_str, **kwargs)
        self.eq.put(("stats", self.llm_calls, self.tool_calls, self.tokens_in + self.tokens_out))


# ---------------------------------------------------------------------------
# Background Analysis Worker
# ---------------------------------------------------------------------------

class AnalysisWorker(threading.Thread):
    """Background worker that streams the TradingAgents graph to avoid freezing the UI."""

    def __init__(
        self,
        config: Dict[str, Any],
        selections: Dict[str, Any],
        event_queue: queue.Queue,
    ):
        super().__init__(daemon=True)
        self.config = config
        self.selections = selections
        self.event_queue = event_queue
        self._stop_requested = threading.Event()

    def request_stop(self):
        self._stop_requested.set()

    def run(self):
        ticker = self.selections["ticker"]
        analysis_date = self.selections["analysis_date"]
        asset_type = self.selections["asset_type"]
        selected_analyst_keys = self.selections["analysts"]

        self.event_queue.put(("log", f"Iniciando analisis multi-agente para {ticker} ({analysis_date})...", "SYSTEM"))
        self.event_queue.put(("progress", 5, "Inicializando grafo y modelos..."))

        # Setup results directory
        results_dir = Path(self.config["results_dir"]) / ticker / analysis_date
        results_dir.mkdir(parents=True, exist_ok=True)
        report_dir = results_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        log_file = results_dir / "gui_execution.log"

        stats_cb = GUIStatsCallbackHandler(self.event_queue)

        try:
            # Initialize Graph
            graph = TradingAgentsGraph(
                selected_analysts=selected_analyst_keys,
                config=self.config,
                debug=True,
                callbacks=[stats_cb],
            )

            instrument_context = graph.resolve_instrument_context(ticker, asset_type)
            init_agent_state = graph.propagator.create_initial_state(
                ticker,
                analysis_date,
                asset_type=asset_type,
                instrument_context=instrument_context,
            )
            args = graph.propagator.get_graph_args(callbacks=[stats_cb])

            checkpoint_tid = graph.begin_checkpoint(ticker, analysis_date, asset_type)
            if checkpoint_tid is not None:
                args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = checkpoint_tid

            self.event_queue.put(("progress", 10, "Ejecutando equipo de analistas..."))

            # Mark first analyst in progress
            if selected_analyst_keys:
                self.event_queue.put(("agent_status", selected_analyst_keys[0], "in_progress"))

            accumulated_reports = {}
            processed_msg_ids = set()

            for chunk in graph.graph.stream(graph.checkpoint_input(init_agent_state), **args):
                if self._stop_requested.is_set():
                    self.event_queue.put(("log", "[AVISO] Analisis detenido por el usuario.", "WARNING"))
                    self.event_queue.put(("finished", False, "Analisis detenido por el usuario."))
                    return

                # Process chunk messages
                for message in chunk.get("messages", []):
                    msg_id = getattr(message, "id", None)
                    if msg_id is not None:
                        if msg_id in processed_msg_ids:
                            continue
                        processed_msg_ids.add(msg_id)

                    content = getattr(message, "content", None)
                    if content and isinstance(content, str) and content.strip():
                        # Truncate preview in log
                        preview = content.strip().replace("\n", " ")
                        if len(preview) > 180:
                            preview = preview[:177] + "..."
                        msg_type = getattr(message, "type", "agent")
                        self.event_queue.put(("log", f"[{msg_type.upper()}] {preview}", "AGENT"))

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tc in message.tool_calls:
                            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "tool")
                            args_str = str(tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", ""))
                            if len(args_str) > 80:
                                args_str = args_str[:77] + "..."
                            self.event_queue.put(("log", f"[TOOL] {name}({args_str})", "TOOL"))

                # Check analyst reports
                completed_analysts = 0
                for a_key in selected_analyst_keys:
                    rep_key = ANALYST_REPORT_KEYS.get(a_key)
                    if rep_key and chunk.get(rep_key):
                        accumulated_reports[rep_key] = chunk[rep_key]
                        self.event_queue.put(("report_section", rep_key, chunk[rep_key]))
                        self.event_queue.put(("agent_status", a_key, "completed"))
                        self.event_queue.put(("log", f"[✓] Reporte completado: {ANALYST_DISPLAY_NAMES[a_key]}", "SUCCESS"))
                        # Save report to disk
                        with open(report_dir / f"{rep_key}.md", "w", encoding="utf-8") as f:
                            f.write(chunk[rep_key])

                    if rep_key and rep_key in accumulated_reports:
                        completed_analysts += 1

                # Update next analyst in progress
                for idx, a_key in enumerate(selected_analyst_keys):
                    rep_key = ANALYST_REPORT_KEYS.get(a_key)
                    if rep_key not in accumulated_reports:
                        self.event_queue.put(("agent_status", a_key, "in_progress"))
                        break

                analyst_progress = 10 + int((completed_analysts / max(1, len(selected_analyst_keys))) * 25)
                self.event_queue.put(("progress", analyst_progress, f"Analistas completados: {completed_analysts}/{len(selected_analyst_keys)}"))

                # Research Team - Investment Debate
                if chunk.get("investment_debate_state"):
                    debate_state = chunk["investment_debate_state"]
                    bull_hist = debate_state.get("bull_history", "").strip()
                    bear_hist = debate_state.get("bear_history", "").strip()
                    judge = debate_state.get("judge_decision", "").strip()

                    if bull_hist or bear_hist:
                        self.event_queue.put(("agent_status", "bull", "in_progress"))
                        self.event_queue.put(("agent_status", "bear", "in_progress"))
                        self.event_queue.put(("progress", 45, "Debate de inversion en curso (Bull vs Bear)..."))

                    if bull_hist:
                        accumulated_reports["bull_research"] = bull_hist
                        self.event_queue.put(("report_section", "bull_research", bull_hist))
                    if bear_hist:
                        accumulated_reports["bear_research"] = bear_hist
                        self.event_queue.put(("report_section", "bear_research", bear_hist))

                    if judge:
                        accumulated_reports["investment_plan"] = judge
                        self.event_queue.put(("report_section", "investment_plan", judge))
                        self.event_queue.put(("agent_status", "bull", "completed"))
                        self.event_queue.put(("agent_status", "bear", "completed"))
                        self.event_queue.put(("agent_status", "research_manager", "completed"))
                        self.event_queue.put(("agent_status", "trader", "in_progress"))
                        self.event_queue.put(("log", "[✓] Decision del Research Manager completada.", "SUCCESS"))
                        self.event_queue.put(("progress", 60, "Research Manager aprobo plan. Pasando al Trader..."))
                        with open(report_dir / "investment_plan.md", "w", encoding="utf-8") as f:
                            f.write(judge)

                # Trader investment plan
                if chunk.get("trader_investment_plan"):
                    plan = chunk["trader_investment_plan"]
                    accumulated_reports["trader_plan"] = plan
                    self.event_queue.put(("report_section", "trader_plan", plan))
                    self.event_queue.put(("agent_status", "trader", "completed"))
                    self.event_queue.put(("agent_status", "risk_debate", "in_progress"))
                    self.event_queue.put(("log", "[✓] Plan de trading generado por el Trader.", "SUCCESS"))
                    self.event_queue.put(("progress", 75, "Evaluando riesgos con Comite de Riesgo..."))
                    with open(report_dir / "trader_investment_plan.md", "w", encoding="utf-8") as f:
                        f.write(plan)

                # Risk Debate & Portfolio Manager
                if chunk.get("risk_debate_state"):
                    risk_state = chunk["risk_debate_state"]
                    agg_hist = risk_state.get("aggressive_history", "").strip()
                    con_hist = risk_state.get("conservative_history", "").strip()
                    neu_hist = risk_state.get("neutral_history", "").strip()
                    judge = risk_state.get("judge_decision", "").strip()

                    risk_combined = []
                    if agg_hist:
                        risk_combined.append(f"### Analista Agresivo\n{agg_hist}")
                    if con_hist:
                        risk_combined.append(f"### Analista Conservador\n{con_hist}")
                    if neu_hist:
                        risk_combined.append(f"### Analista Neutral\n{neu_hist}")

                    if risk_combined:
                        accumulated_reports["risk_debate"] = "\n\n".join(risk_combined)
                        self.event_queue.put(("report_section", "risk_debate", accumulated_reports["risk_debate"]))

                    if judge:
                        accumulated_reports["final_trade_decision"] = judge
                        self.event_queue.put(("report_section", "final_trade_decision", judge))
                        self.event_queue.put(("agent_status", "risk_debate", "completed"))
                        self.event_queue.put(("agent_status", "portfolio_manager", "completed"))
                        self.event_queue.put(("log", "[✓] Decision final del Portfolio Manager completada!", "SUCCESS"))
                        self.event_queue.put(("progress", 95, "Finalizando y consolidando reportes..."))
                        with open(report_dir / "final_trade_decision.md", "w", encoding="utf-8") as f:
                            f.write(judge)

            # Mark complete
            graph.clear_checkpoint_on_success(ticker, analysis_date, asset_type)
            self.event_queue.put(("progress", 100, "Analisis completado con exito!"))
            self.event_queue.put(("log", f"Analisis finalizado exitosamente. Archivos guardados en: {results_dir}", "SUCCESS"))
            self.event_queue.put(("finished", True, str(results_dir)))

        except Exception as exc:
            logger.exception("Error during background analysis")
            self.event_queue.put(("log", f"[ERROR] {exc}", "ERROR"))
            self.event_queue.put(("finished", False, str(exc)))


# ---------------------------------------------------------------------------
# Main GUI Window
# ---------------------------------------------------------------------------

class TradingAgentsApp(tk.Tk):
    """Modern graphical user interface for TradingAgents."""

    def __init__(self, checkpoint: Optional[bool] = None, clear_checkpoints: bool = False):
        super().__init__()

        self.title("TradingAgents - Multi-Agent Financial Trading Framework")
        self.geometry("1180x820")
        self.minsize(980, 680)

        self.initial_checkpoint = checkpoint
        self.clear_checkpoints_on_start = clear_checkpoints
        self.worker: Optional[AnalysisWorker] = None
        self.event_queue: queue.Queue = queue.Queue()
        self.start_time: Optional[float] = None
        self.is_running = False

        self.results_dir_path: Optional[str] = None
        self.report_data: Dict[str, str] = {}

        # Configure styles and layout
        self._setup_theme()
        self._build_ui()
        self._populate_defaults()

        if self.clear_checkpoints_on_start:
            from tradingagents.graph.checkpointer import clear_all_checkpoints
            cleared = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
            self._log(f"[AVISO] Se eliminaron {cleared} checkpoint(s) previos.", "WARNING")

        # Start periodic queue polling
        self.after(100, self._process_queue)

    # -----------------------------------------------------------------------
    # Theme & Styling
    # -----------------------------------------------------------------------

    def _setup_theme(self):
        # Color Palette - Material Deep Ocean / Prism Dark Theme
        self.colors = {
            "bg_main": "#1e1e2e",          # Material Deep Ocean / Prism Dark background
            "bg_card": "#25283d",          # Clean elevated card surface
            "bg_card_alt": "#2c304d",      # Secondary elevated card / metrics / active container
            "bg_input": "#2a2d42",         # Input fields & comboboxes background
            "bg_input_focus": "#343854",   # Input focus background
            "border": "#3e4466",           # Subtle crisp border
            "text_primary": "#ffffff",     # Crisp pure white for primary text, inputs, titles
            "text_secondary": "#e2e8f0",   # Bright slate-lavender for subtitles, labels
            "text_muted": "#94a3b8",       # Clean legible slate gray
            "accent_purple": "#c792ea",    # Orchid purple (keywords, section titles, active tabs)
            "accent_cyan": "#80cbc4",      # Mint cyan (subheaders, tools, metrics)
            "accent_blue": "#82aaff",      # Sky blue (main header, LLM calls)
            "accent_green": "#34d399",     # Mint emerald green (buy, success, tags)
            "accent_green_btn": "#10b981", # Vibrant action button green
            "accent_green_active": "#059669", # Active action button
            "accent_yellow": "#ffcb6b",    # Amber gold (tokens, warnings, in progress)
            "accent_red": "#ff5370",       # Coral red (errors, sell, stop button)
            "accent_red_active": "#e11d48", # Active stop button
            "terminal_bg": "#151622",      # Deep clean code editor / console background
            "terminal_fg": "#f8f8f2",      # Crisp code editor text
            "selection_bg": "#444267",     # Text selection background
            "selection_fg": "#ffffff",     # Text selection foreground
        }

        self.configure(bg=self.colors["bg_main"])

        # Configure standard Tk Listbox popup used by Combobox
        self.option_add("*TCombobox*Listbox.background", self.colors["bg_card"])
        self.option_add("*TCombobox*Listbox.foreground", self.colors["text_primary"])
        self.option_add("*TCombobox*Listbox.selectBackground", self.colors["accent_purple"])
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 9))
        self.option_add("*TCombobox*Listbox.relief", "flat")
        self.option_add("*TCombobox*Listbox.borderWidth", "0")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=self.colors["bg_main"], foreground=self.colors["text_primary"])
        style.configure("TFrame", background=self.colors["bg_main"])
        style.configure("Card.TFrame", background=self.colors["bg_card"], relief="flat")
        style.configure("TLabel", background=self.colors["bg_card"], foreground=self.colors["text_secondary"], font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=self.colors["bg_main"], foreground=self.colors["accent_blue"], font=("Segoe UI", 12, "bold"))
        style.configure("SubHeader.TLabel", background=self.colors["bg_card"], foreground=self.colors["accent_purple"], font=("Segoe UI", 10, "bold"))
        style.configure("Muted.TLabel", background=self.colors["bg_card"], foreground=self.colors["text_muted"], font=("Segoe UI", 8))

        # Entry styling
        style.configure(
            "TEntry",
            fieldbackground=self.colors["bg_input"],
            foreground=self.colors["text_primary"],
            insertcolor=self.colors["accent_purple"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
            padding=6,
        )
        style.map(
            "TEntry",
            bordercolor=[("focus", self.colors["accent_purple"])],
            fieldbackground=[("focus", self.colors["bg_input_focus"])],
        )

        # Combobox styling
        style.configure(
            "TCombobox",
            background=self.colors["bg_card_alt"],
            fieldbackground=self.colors["bg_input"],
            foreground=self.colors["text_primary"],
            arrowcolor=self.colors["accent_cyan"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
            padding=5,
        )
        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", self.colors["bg_input"]),
                ("focus", self.colors["bg_input_focus"]),
                ("active", self.colors["bg_input"]),
            ],
            foreground=[
                ("readonly", self.colors["text_primary"]),
                ("focus", self.colors["text_primary"]),
                ("active", self.colors["text_primary"]),
            ],
            bordercolor=[("focus", self.colors["accent_purple"])],
            arrowcolor=[("focus", self.colors["accent_purple"]), ("hover", self.colors["accent_purple"])],
        )

        # Checkbutton styling
        style.configure(
            "TCheckbutton",
            background=self.colors["bg_card"],
            foreground=self.colors["text_secondary"],
            indicatorcolor=self.colors["bg_input"],
            indicatorbackground=self.colors["bg_input"],
            font=("Segoe UI", 9),
        )
        style.map(
            "TCheckbutton",
            background=[("active", self.colors["bg_card"])],
            foreground=[("active", self.colors["text_primary"])],
            indicatorcolor=[
                ("selected", self.colors["accent_purple"]),
                ("pressed", self.colors["accent_purple"]),
            ],
        )

        # Buttons
        style.configure(
            "Primary.TButton",
            background=self.colors["accent_green_btn"],
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=8,
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.colors["accent_green_active"]), ("disabled", "#333852")],
            foreground=[("disabled", "#64748b")],
        )

        style.configure(
            "Stop.TButton",
            background=self.colors["accent_red"],
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=8,
        )
        style.map(
            "Stop.TButton",
            background=[("active", self.colors["accent_red_active"]), ("disabled", "#333852")],
            foreground=[("disabled", "#64748b")],
        )

        style.configure(
            "Secondary.TButton",
            background=self.colors["border"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 9),
            padding=6,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#4f5782"), ("disabled", "#25283d")],
            foreground=[("disabled", "#64748b")],
        )

        # Progressbar
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=self.colors["bg_card_alt"],
            background=self.colors["accent_cyan"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["accent_cyan"],
            darkcolor=self.colors["accent_cyan"],
        )

        # Notebook tabs
        style.configure("TNotebook", background=self.colors["bg_main"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=self.colors["bg_card"],
            foreground=self.colors["text_muted"],
            padding=[14, 8],
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.colors["bg_card_alt"]), ("active", self.colors["border"])],
            foreground=[("selected", self.colors["accent_purple"]), ("active", self.colors["text_primary"])],
        )

        # Scrollbar styling
        style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background=self.colors["bg_card_alt"],
            troughcolor=self.colors["bg_card"],
            bordercolor=self.colors["border"],
            arrowcolor=self.colors["accent_cyan"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", self.colors["accent_purple"])],
        )

    # -----------------------------------------------------------------------
    # UI Layout Construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # Top banner
        header_frame = tk.Frame(self, bg=self.colors["bg_main"], height=55)
        header_frame.pack(fill="x", padx=16, pady=(12, 6))

        title_lbl = tk.Label(
            header_frame,
            text="📈 TradingAgents — Multi-Agent Financial Trading Framework",
            font=("Segoe UI", 14, "bold"),
            fg=self.colors["accent_blue"],
            bg=self.colors["bg_main"],
        )
        title_lbl.pack(side="left")

        subtitle_lbl = tk.Label(
            header_frame,
            text="Autonomous LLM Agents for Market Research, Debate & Portfolio Decision",
            font=("Segoe UI", 9),
            fg=self.colors["accent_cyan"],
            bg=self.colors["bg_main"],
        )
        subtitle_lbl.pack(side="left", padx=(16, 0), pady=(4, 0))

        # Main Paned layout (Left = Config, Right = Monitor & Reports)
        paned = tk.PanedWindow(self, orient="horizontal", bg=self.colors["border"], sashwidth=4, bd=0)
        paned.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        # -------------------------------------------------------------------
        # Left Panel: Configuration Card
        # -------------------------------------------------------------------
        left_container = tk.Frame(paned, bg=self.colors["bg_card"], width=360)
        paned.add(left_container, minsize=320)

        # Scrollable canvas for config options
        canvas = tk.Canvas(left_container, bg=self.colors["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        scroll_content = ttk.Frame(canvas, style="Card.TFrame")

        scroll_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.configure(xscrollcommand=None, yscrollcommand=scrollbar.set)

        def _on_canvas_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Config Sections inside scroll_content
        pad_opts = {"padx": 14, "pady": 4}

        # Header
        ttk.Label(scroll_content, text="⚙️ Parámetros del Análisis", style="SubHeader.TLabel").pack(anchor="w", padx=14, pady=(14, 8))

        # Ticker Symbol
        ttk.Label(scroll_content, text="Símbolo Ticker (ej. SPY, NVDA, 0700.HK, BTC-USD):").pack(anchor="w", **pad_opts)
        self.entry_ticker = ttk.Entry(scroll_content)
        self.entry_ticker.pack(fill="x", **pad_opts)

        # Analysis Date
        ttk.Label(scroll_content, text="Fecha del Análisis (YYYY-MM-DD):").pack(anchor="w", **pad_opts)
        self.entry_date = ttk.Entry(scroll_content)
        self.entry_date.pack(fill="x", **pad_opts)

        # Output Language
        ttk.Label(scroll_content, text="Idioma de Salida:").pack(anchor="w", **pad_opts)
        self.combo_lang = ttk.Combobox(
            scroll_content,
            values=["English", "Spanish", "Chinese", "French", "German", "Japanese"],
            state="readonly",
        )
        self.combo_lang.pack(fill="x", **pad_opts)

        # Research Depth
        ttk.Label(scroll_content, text="Profundidad de Investigación:").pack(anchor="w", **pad_opts)
        self.combo_depth = ttk.Combobox(
            scroll_content,
            values=["1 - Rápido (1 ronda debate/riesgo)", "2 - Estándar (2 rondas)", "3 - Profundo (3 rondas)"],
            state="readonly",
        )
        self.combo_depth.pack(fill="x", **pad_opts)

        # LLM Provider
        ttk.Label(scroll_content, text="Proveedor LLM:").pack(anchor="w", padx=14, pady=(10, 4))
        provider_list = sorted(list(MODEL_OPTIONS.keys()))
        self.combo_provider = ttk.Combobox(
            scroll_content,
            values=provider_list,
            state="readonly",
        )
        self.combo_provider.pack(fill="x", **pad_opts)
        self.combo_provider.bind("<<ComboboxSelected>>", self._on_provider_change)

        # Quick Thinking Model
        ttk.Label(scroll_content, text="Modelo Quick Thinking:").pack(anchor="w", **pad_opts)
        self.combo_quick_model = ttk.Combobox(scroll_content)
        self.combo_quick_model.pack(fill="x", **pad_opts)

        # Deep Thinking Model
        ttk.Label(scroll_content, text="Modelo Deep Thinking (con fallback automático):").pack(anchor="w", **pad_opts)
        self.combo_deep_model = ttk.Combobox(scroll_content)
        self.combo_deep_model.pack(fill="x", **pad_opts)

        # Provider Thinking Config
        ttk.Label(scroll_content, text="Modo de Pensamiento / Razonamiento:").pack(anchor="w", **pad_opts)
        self.combo_thinking = ttk.Combobox(
            scroll_content,
            values=["default", "high", "medium", "low", "minimal"],
            state="readonly",
        )
        self.combo_thinking.pack(fill="x", **pad_opts)

        # Analysts Team Checklist
        ttk.Label(scroll_content, text="👥 Equipo de Analistas:", style="SubHeader.TLabel").pack(anchor="w", padx=14, pady=(14, 6))

        self.chk_vars = {}
        for key in ANALYST_KEYS:
            var = tk.BooleanVar(value=True)
            self.chk_vars[key] = var
            chk = ttk.Checkbutton(
                scroll_content,
                text=ANALYST_DISPLAY_NAMES[key],
                variable=var,
            )
            chk.pack(anchor="w", padx=20, pady=2)

        # Options Checklist
        ttk.Label(scroll_content, text="🔧 Opciones:", style="SubHeader.TLabel").pack(anchor="w", padx=14, pady=(14, 6))
        self.var_checkpoint = tk.BooleanVar(value=bool(self.initial_checkpoint or DEFAULT_CONFIG["checkpoint_enabled"]))
        chk_cp = ttk.Checkbutton(
            scroll_content,
            text="Habilitar Checkpoints (reanudar si se interrumpe)",
            variable=self.var_checkpoint,
        )
        chk_cp.pack(anchor="w", padx=20, pady=2)

        self.var_auto_scroll = tk.BooleanVar(value=True)
        chk_as = ttk.Checkbutton(
            scroll_content,
            text="Auto-scroll en consola en vivo",
            variable=self.var_auto_scroll,
        )
        chk_as.pack(anchor="w", padx=20, pady=2)

        # Action Buttons
        btn_frame = ttk.Frame(scroll_content, style="Card.TFrame")
        btn_frame.pack(fill="x", padx=14, pady=(18, 20))

        self.btn_start = ttk.Button(
            btn_frame,
            text="▶ Iniciar Análisis",
            style="Primary.TButton",
            command=self._start_analysis,
        )
        self.btn_start.pack(fill="x", pady=4)

        self.btn_stop = ttk.Button(
            btn_frame,
            text="⏹ Detener",
            style="Stop.TButton",
            command=self._stop_analysis,
            state="disabled",
        )
        self.btn_stop.pack(fill="x", pady=4)

        self.btn_open_folder = ttk.Button(
            btn_frame,
            text="📁 Abrir Carpeta de Resultados",
            style="Secondary.TButton",
            command=self._open_results_dir,
            state="disabled",
        )
        self.btn_open_folder.pack(fill="x", pady=4)

        # -------------------------------------------------------------------
        # Right Panel: Monitor, Status Badges & Notebook
        # -------------------------------------------------------------------
        right_container = tk.Frame(paned, bg=self.colors["bg_main"])
        paned.add(right_container, minsize=600)

        # Top Status & Metrics Card
        status_card = tk.Frame(right_container, bg=self.colors["bg_card"], bd=0)
        status_card.pack(fill="x", pady=(0, 10))

        # Metrics Bar
        metrics_bar = tk.Frame(status_card, bg=self.colors["bg_card_alt"], height=36)
        metrics_bar.pack(fill="x", padx=10, pady=(10, 8))

        self.lbl_metric_time = tk.Label(metrics_bar, text="⏱ Tiempo: 00:00:00", bg=self.colors["bg_card_alt"], fg=self.colors["text_primary"], font=("Segoe UI", 9, "bold"))
        self.lbl_metric_time.pack(side="left", padx=12, pady=6)

        self.lbl_metric_llm = tk.Label(metrics_bar, text="🤖 LLM Calls: 0", bg=self.colors["bg_card_alt"], fg=self.colors["accent_blue"], font=("Segoe UI", 9))
        self.lbl_metric_llm.pack(side="left", padx=12, pady=6)

        self.lbl_metric_tools = tk.Label(metrics_bar, text="🛠️ Tools: 0", bg=self.colors["bg_card_alt"], fg=self.colors["accent_cyan"], font=("Segoe UI", 9))
        self.lbl_metric_tools.pack(side="left", padx=12, pady=6)

        self.lbl_metric_tokens = tk.Label(metrics_bar, text="📊 Tokens: 0", bg=self.colors["bg_card_alt"], fg=self.colors["accent_yellow"], font=("Segoe UI", 9))
        self.lbl_metric_tokens.pack(side="left", padx=12, pady=6)

        self.lbl_status_text = tk.Label(metrics_bar, text="Listo para iniciar", bg=self.colors["bg_card_alt"], fg=self.colors["text_secondary"], font=("Segoe UI", 9, "italic"))
        self.lbl_status_text.pack(side="right", padx=14, pady=6)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            status_card,
            variable=self.progress_var,
            maximum=100.0,
            style="Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))

        # Agent Status Badges Grid
        badges_frame = tk.Frame(status_card, bg=self.colors["bg_card"])
        badges_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.badge_labels = {}
        for idx, key in enumerate(ALL_AGENT_KEYS):
            row = idx // 5
            col = idx % 5

            badge_box = tk.Frame(badges_frame, bg=self.colors["bg_card_alt"], highlightbackground=self.colors["border"], highlightthickness=1)
            badge_box.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            badges_frame.columnconfigure(col, weight=1)

            lbl_name = tk.Label(
                badge_box,
                text=AGENT_LABELS[key],
                bg=self.colors["bg_card_alt"],
                fg=self.colors["text_primary"],
                font=("Segoe UI", 8, "bold"),
            )
            lbl_name.pack(anchor="w", padx=6, pady=(4, 1))

            lbl_status = tk.Label(
                badge_box,
                text="⏳ Pendiente",
                bg=self.colors["bg_card_alt"],
                fg=self.colors["text_muted"],
                font=("Segoe UI", 8),
            )
            lbl_status.pack(anchor="w", padx=6, pady=(0, 4))

            self.badge_labels[key] = (badge_box, lbl_status)

        # -------------------------------------------------------------------
        # Notebook for Reports & Live Stream
        # -------------------------------------------------------------------
        self.notebook = ttk.Notebook(right_container)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Final Decision Banner & Overview
        self.tab_final = tk.Frame(self.notebook, bg=self.colors["bg_card"])
        self.notebook.add(self.tab_final, text="🏆 Decisión Final")
        self._build_final_decision_tab()

        # Tab 2: Analyst Reports
        self.tab_analysts = tk.Frame(self.notebook, bg=self.colors["bg_card"])
        self.notebook.add(self.tab_analysts, text="📊 Reportes de Analistas")
        self._build_analyst_reports_tab()

        # Tab 3: Debate & Risk
        self.tab_debates = tk.Frame(self.notebook, bg=self.colors["bg_card"])
        self.notebook.add(self.tab_debates, text="⚔️ Debates & Riesgo")
        self._build_debates_tab()

        # Tab 4: Live Logs Console
        self.tab_logs = tk.Frame(self.notebook, bg=self.colors["terminal_bg"])
        self.notebook.add(self.tab_logs, text="📜 Registro en Vivo (Logs)")
        self._build_logs_tab()

    def _build_final_decision_tab(self):
        # Top Decision Banner
        self.decision_banner = tk.Frame(self.tab_final, bg=self.colors["bg_card_alt"], height=70)
        self.decision_banner.pack(fill="x", padx=12, pady=12)

        self.lbl_decision_action = tk.Label(
            self.decision_banner,
            text="PROPUESTA: PENDIENTE",
            font=("Segoe UI", 16, "bold"),
            bg=self.colors["bg_card_alt"],
            fg=self.colors["accent_yellow"],
        )
        self.lbl_decision_action.pack(side="left", padx=16, pady=12)

        self.lbl_decision_meta = tk.Label(
            self.decision_banner,
            text="Inicie el análisis para obtener la recomendación del Portfolio Manager.",
            font=("Segoe UI", 9),
            bg=self.colors["bg_card_alt"],
            fg=self.colors["text_secondary"],
        )
        self.lbl_decision_meta.pack(side="left", padx=8, pady=12)

        # Full Decision Text
        self.txt_final_decision = ScrolledText(
            self.tab_final,
            wrap="word",
            bg=self.colors["terminal_bg"],
            fg=self.colors["terminal_fg"],
            insertbackground=self.colors["accent_purple"],
            selectbackground=self.colors["selection_bg"],
            selectforeground=self.colors["selection_fg"],
            font=("Consolas", 10),
            bd=0,
            padx=14,
            pady=14,
        )
        self.txt_final_decision.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt_final_decision.insert("1.0", "Esperando ejecución del análisis para generar el informe final...")
        self.txt_final_decision.configure(state="disabled")

    def _build_analyst_reports_tab(self):
        nav_frame = tk.Frame(self.tab_analysts, bg=self.colors["bg_card"])
        nav_frame.pack(fill="x", padx=12, pady=(10, 6))

        tk.Label(
            nav_frame,
            text="Seleccionar Reporte:",
            bg=self.colors["bg_card"],
            fg=self.colors["accent_cyan"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 8))

        self.combo_report_section = ttk.Combobox(
            nav_frame,
            values=[
                "Market Analyst (Técnico / OHLCV / Indicadores)",
                "Sentiment Analyst (Redes Sociales / Sentimiento)",
                "News Analyst (Noticias Globales / Macro)",
                "Fundamentals Analyst (Financiero / Ratios)",
            ],
            state="readonly",
            width=48,
        )
        self.combo_report_section.current(0)
        self.combo_report_section.pack(side="left", padx=4)
        self.combo_report_section.bind("<<ComboboxSelected>>", self._on_report_section_select)

        self.txt_analyst_report = ScrolledText(
            self.tab_analysts,
            wrap="word",
            bg=self.colors["terminal_bg"],
            fg=self.colors["terminal_fg"],
            insertbackground=self.colors["accent_purple"],
            selectbackground=self.colors["selection_bg"],
            selectforeground=self.colors["selection_fg"],
            font=("Consolas", 10),
            bd=0,
            padx=14,
            pady=14,
        )
        self.txt_analyst_report.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt_analyst_report.insert("1.0", "Seleccione un reporte para visualizar el contenido una vez completado.")
        self.txt_analyst_report.configure(state="disabled")

    def _build_debates_tab(self):
        nav_frame = tk.Frame(self.tab_debates, bg=self.colors["bg_card"])
        nav_frame.pack(fill="x", padx=12, pady=(10, 6))

        tk.Label(
            nav_frame,
            text="Sección de Debate:",
            bg=self.colors["bg_card"],
            fg=self.colors["accent_cyan"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 8))

        self.combo_debate_section = ttk.Combobox(
            nav_frame,
            values=[
                "Plan de Inversión (Research Manager Decision)",
                "Tesis Alcista (Bull Researcher)",
                "Tesis Bajista (Bear Researcher)",
                "Plan del Trader (Trader Investment Plan)",
                "Comité de Riesgo (Agresivo / Conservador / Neutral)",
            ],
            state="readonly",
            width=48,
        )
        self.combo_debate_section.current(0)
        self.combo_debate_section.pack(side="left", padx=4)
        self.combo_debate_section.bind("<<ComboboxSelected>>", self._on_debate_section_select)

        self.txt_debate_report = ScrolledText(
            self.tab_debates,
            wrap="word",
            bg=self.colors["terminal_bg"],
            fg=self.colors["terminal_fg"],
            insertbackground=self.colors["accent_purple"],
            selectbackground=self.colors["selection_bg"],
            selectforeground=self.colors["selection_fg"],
            font=("Consolas", 10),
            bd=0,
            padx=14,
            pady=14,
        )
        self.txt_debate_report.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt_debate_report.insert("1.0", "Los debates entre los investigadores y comités de riesgo aparecerán aquí.")
        self.txt_debate_report.configure(state="disabled")

    def _build_logs_tab(self):
        self.txt_logs = ScrolledText(
            self.tab_logs,
            wrap="word",
            bg=self.colors["terminal_bg"],
            fg=self.colors["terminal_fg"],
            insertbackground=self.colors["accent_purple"],
            selectbackground=self.colors["selection_bg"],
            selectforeground=self.colors["selection_fg"],
            font=("Consolas", 9),
            bd=0,
            padx=12,
            pady=12,
        )
        self.txt_logs.pack(fill="both", expand=True, padx=6, pady=6)

        # Configure color tags for log levels
        self.txt_logs.tag_config("SYSTEM", foreground=self.colors["accent_cyan"])
        self.txt_logs.tag_config("AGENT", foreground=self.colors["text_primary"])
        self.txt_logs.tag_config("TOOL", foreground=self.colors["accent_purple"])
        self.txt_logs.tag_config("SUCCESS", foreground=self.colors["accent_green"])
        self.txt_logs.tag_config("WARNING", foreground=self.colors["accent_yellow"])
        self.txt_logs.tag_config("ERROR", foreground=self.colors["accent_red"])

    # -----------------------------------------------------------------------
    # Defaults & Model Configuration
    # -----------------------------------------------------------------------

    def _populate_defaults(self):
        self.entry_ticker.insert(0, DEFAULT_CONFIG.get("benchmark_ticker") or "SPY")
        self.entry_date.insert(0, datetime.datetime.now().strftime("%Y-%m-%d"))

        out_lang = DEFAULT_CONFIG.get("output_language", "English")
        if out_lang in self.combo_lang["values"]:
            self.combo_lang.set(out_lang)
        else:
            self.combo_lang.set("English")

        self.combo_depth.set("1 - Rápido (1 ronda debate/riesgo)")

        provider = (DEFAULT_CONFIG.get("llm_provider") or "google").lower()
        if provider in self.combo_provider["values"]:
            self.combo_provider.set(provider)
        else:
            self.combo_provider.set("google")

        self._refresh_model_lists(self.combo_provider.get())

        # Select provider default models
        quick_def = DEFAULT_CONFIG.get("quick_think_llm")
        if quick_def and quick_def in self.combo_quick_model["values"]:
            self.combo_quick_model.set(quick_def)
        elif self.combo_quick_model["values"]:
            self.combo_quick_model.current(0)

        deep_def = DEFAULT_CONFIG.get("deep_think_llm")
        if deep_def and deep_def in self.combo_deep_model["values"]:
            self.combo_deep_model.set(deep_def)
        elif self.combo_deep_model["values"]:
            self.combo_deep_model.current(0)

        thinking_def = (
            DEFAULT_CONFIG.get("google_thinking_level")
            or DEFAULT_CONFIG.get("openai_reasoning_effort")
            or DEFAULT_CONFIG.get("anthropic_effort")
            or "default"
        )
        self.combo_thinking.set(thinking_def)

    def _refresh_model_lists(self, provider: str):
        p = provider.lower()
        quick_opts = [val for _, val in get_model_options(p, "quick") if val != "custom"]
        deep_opts = [val for _, val in get_model_options(p, "deep", include_quick=True) if val != "custom"]

        if not quick_opts:
            quick_opts = [DEFAULT_CONFIG.get("quick_think_llm", "default")]
        if not deep_opts:
            deep_opts = [DEFAULT_CONFIG.get("deep_think_llm", "default")]

        self.combo_quick_model["values"] = quick_opts
        self.combo_deep_model["values"] = deep_opts

        if quick_opts:
            self.combo_quick_model.set(quick_opts[0])
        if deep_opts:
            self.combo_deep_model.set(deep_opts[0])

    def _on_provider_change(self, event=None):
        provider = self.combo_provider.get()
        self._refresh_model_lists(provider)

    # -----------------------------------------------------------------------
    # Logging & Queue Processing
    # -----------------------------------------------------------------------

    def _log(self, text: str, tag: str = "SYSTEM"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {text}\n"
        self.txt_logs.insert(tk.END, formatted, tag)
        if self.var_auto_scroll.get():
            self.txt_logs.see(tk.END)

    def _set_agent_badge_status(self, key: str, status: str):
        if key not in self.badge_labels:
            return
        box, lbl = self.badge_labels[key]
        if status == "in_progress":
            lbl.configure(text="⚙️ En Ejecución", fg=self.colors["accent_yellow"])
            box.configure(highlightbackground=self.colors["accent_yellow"])
        elif status == "completed":
            lbl.configure(text="✅ Completado", fg=self.colors["accent_green"])
            box.configure(highlightbackground=self.colors["accent_green"])
        else:
            lbl.configure(text="⏳ Pendiente", fg=self.colors["text_muted"])
            box.configure(highlightbackground=self.colors["border"])

    def _reset_all_badges(self):
        for key in ALL_AGENT_KEYS:
            self._set_agent_badge_status(key, "pending")

    def _process_queue(self):
        """Poll events from worker thread and update UI widgets safely."""
        try:
            while True:
                event = self.event_queue.get_nowait()
                msg_type = event[0]

                if msg_type == "log":
                    _, text, tag = event
                    self._log(text, tag)

                elif msg_type == "progress":
                    _, pct, label = event
                    self.progress_var.set(pct)
                    self.lbl_status_text.configure(text=label)

                elif msg_type == "agent_status":
                    _, agent_key, status = event
                    self._set_agent_badge_status(agent_key, status)

                elif msg_type == "stats":
                    _, llm_c, tool_c, tokens = event
                    self.lbl_metric_llm.configure(text=f"🤖 LLM Calls: {llm_c}")
                    self.lbl_metric_tools.configure(text=f"🛠️ Tools: {tool_c}")
                    self.lbl_metric_tokens.configure(text=f"📊 Tokens: {tokens:,}")

                elif msg_type == "report_section":
                    _, sec_name, content = event
                    self.report_data[sec_name] = content
                    if sec_name == "final_trade_decision":
                        self._render_final_decision(content)
                    self._refresh_active_report_views()

                elif msg_type == "finished":
                    _, success, msg = event
                    self.is_running = False
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")

                    if success:
                        self.results_dir_path = msg
                        self.btn_open_folder.configure(state="normal")
                        messagebox.showinfo("TradingAgents", "¡El análisis ha finalizado con éxito!")
                        self.notebook.select(self.tab_final)
                    else:
                        messagebox.showwarning("TradingAgents", f"El análisis finalizó con aviso: {msg}")

        except queue.Empty:
            pass

        # Update elapsed timer
        if self.is_running and self.start_time is not None:
            elapsed = int(time.time() - self.start_time)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self.lbl_metric_time.configure(text=f"⏱ Tiempo: {h:02d}:{m:02d}:{s:02d}")

        self.after(100, self._process_queue)

    def _render_final_decision(self, text: str):
        self.txt_final_decision.configure(state="normal")
        self.txt_final_decision.delete("1.0", tk.END)
        self.txt_final_decision.insert("1.0", text)
        self.txt_final_decision.configure(state="disabled")

        upper_text = text.upper()
        if "BUY" in upper_text or "COMPRA" in upper_text:
            self.lbl_decision_action.configure(text="PROPUESTA: COMPRAR (BUY)", fg="#34d399")
            self.decision_banner.configure(bg="#064e3b")
            self.lbl_decision_action.configure(bg="#064e3b")
            self.lbl_decision_meta.configure(bg="#064e3b", fg="#e2e8f0", text="Recomendación final consolidada por el Portfolio Manager.")
        elif "SELL" in upper_text or "VENTA" in upper_text:
            self.lbl_decision_action.configure(text="PROPUESTA: VENDER (SELL)", fg="#ff5370")
            self.decision_banner.configure(bg="#4c0519")
            self.lbl_decision_action.configure(bg="#4c0519")
            self.lbl_decision_meta.configure(bg="#4c0519", fg="#e2e8f0", text="Recomendación final consolidada por el Portfolio Manager.")
        elif "HOLD" in upper_text or "MANTENER" in upper_text:
            self.lbl_decision_action.configure(text="PROPUESTA: MANTENER (HOLD)", fg="#ffcb6b")
            self.decision_banner.configure(bg="#451a03")
            self.lbl_decision_action.configure(bg="#451a03")
            self.lbl_decision_meta.configure(bg="#451a03", fg="#e2e8f0", text="Recomendación final consolidada por el Portfolio Manager.")

    def _refresh_active_report_views(self):
        self._on_report_section_select()
        self._on_debate_section_select()

    def _on_report_section_select(self, event=None):
        idx = self.combo_report_section.current()
        key_map = [
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
        ]
        key = key_map[idx] if idx < len(key_map) else "market_report"
        content = self.report_data.get(key, "Aún no se ha generado este reporte.")
        self.txt_analyst_report.configure(state="normal")
        self.txt_analyst_report.delete("1.0", tk.END)
        self.txt_analyst_report.insert("1.0", content)
        self.txt_analyst_report.configure(state="disabled")

    def _on_debate_section_select(self, event=None):
        idx = self.combo_debate_section.current()
        key_map = [
            "investment_plan",
            "bull_research",
            "bear_research",
            "trader_plan",
            "risk_debate",
        ]
        key = key_map[idx] if idx < len(key_map) else "investment_plan"
        content = self.report_data.get(key, "Aún no se ha generado esta sección de debate.")
        self.txt_debate_report.configure(state="normal")
        self.txt_debate_report.delete("1.0", tk.END)
        self.txt_debate_report.insert("1.0", content)
        self.txt_debate_report.configure(state="disabled")

    # -----------------------------------------------------------------------
    # Action Handlers
    # -----------------------------------------------------------------------

    def _start_analysis(self):
        if self.is_running:
            return

        ticker = self.entry_ticker.get().strip()
        if not ticker:
            messagebox.showerror("Error", "Debe ingresar un símbolo ticker.")
            return

        date_str = self.entry_date.get().strip()
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Error", "Formato de fecha inválido. Utilice YYYY-MM-DD.")
            return

        selected_analysts = [key for key, var in self.chk_vars.items() if var.get()]
        if not selected_analysts:
            messagebox.showerror("Error", "Debe seleccionar al menos un analista.")
            return

        depth_val = 1
        if "2" in self.combo_depth.get():
            depth_val = 2
        elif "3" in self.combo_depth.get():
            depth_val = 3

        provider = self.combo_provider.get()
        quick_model = self.combo_quick_model.get().strip()
        deep_model = self.combo_deep_model.get().strip()
        thinking = self.combo_thinking.get()
        if thinking == "default":
            thinking = None

        asset_type = detect_asset_type(ticker).value

        # Prepare config
        run_config = DEFAULT_CONFIG.copy()
        run_config["llm_provider"] = provider.lower()
        run_config["quick_think_llm"] = quick_model
        run_config["deep_think_llm"] = deep_model
        run_config["max_debate_rounds"] = depth_val
        run_config["max_risk_discuss_rounds"] = depth_val
        run_config["output_language"] = self.combo_lang.get()
        run_config["checkpoint_enabled"] = self.var_checkpoint.get()

        if provider == "google":
            run_config["google_thinking_level"] = thinking
        elif provider == "openai":
            run_config["openai_reasoning_effort"] = thinking
        elif provider == "anthropic":
            run_config["anthropic_effort"] = thinking

        selections = {
            "ticker": ticker,
            "analysis_date": date_str,
            "asset_type": asset_type,
            "analysts": selected_analysts,
        }

        # Reset UI
        self._reset_all_badges()
        self.report_data.clear()
        self.progress_var.set(0.0)
        self.is_running = True
        self.start_time = time.time()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_open_folder.configure(state="disabled")

        self.notebook.select(self.tab_logs)

        # Launch worker
        self.worker = AnalysisWorker(
            config=run_config,
            selections=selections,
            event_queue=self.event_queue,
        )
        self.worker.start()

    def _stop_analysis(self):
        if self.worker and self.worker.is_alive():
            self._log("[AVISO] Solicitando detención...", "WARNING")
            self.worker.request_stop()
            self.btn_stop.configure(state="disabled")

    def _open_results_dir(self):
        if not self.results_dir_path:
            return
        path = os.path.abspath(self.results_dir_path)
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def launch_gui(checkpoint: Optional[bool] = None, clear_checkpoints: bool = False):
    """Launch the TradingAgents desktop user interface window."""
    app = TradingAgentsApp(checkpoint=checkpoint, clear_checkpoints=clear_checkpoints)
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
