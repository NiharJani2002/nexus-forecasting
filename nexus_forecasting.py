"""
NEXUS: An Agentic Framework for Time Series Forecasting
========================================================

Architecture:
  Stage 1: Contextualization       → Historical Context Agent (A_ctx)
  Stage 2: Dual-Resolution Outlook → Macro-Reasoning Agent (A_macro)
                                    + Micro-Reasoning Agent (A_micro)
  Stage 3: Synthesis & Calibration → Forecast Synthesizer / Value Predictor Agent (A_syn)
                                    + Calibration Agent (A_calib)
"""
!pip install anthropic

import os
import json
import re
import math
import time
import logging
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesWindow:
    """One window of data passed to the framework."""
    timestamps: list[str]          # e.g. ["2025-01-06", ...]
    values: list[float]
    texts: list[str]               # parallel to timestamps; "" if no text
    target_name: str               # e.g. "MSFT closing price"
    domain: str                    # e.g. "Stock Market"
    frequency: str                 # e.g. "Weekly"
    future_timestamps: list[str]   # the T steps we need to predict


@dataclass
class AgentOutputs:
    structured_history: str = ""
    macro_values: list[float] = field(default_factory=list)
    macro_reasoning: str = ""
    micro_values: list[float] = field(default_factory=list)
    micro_reasoning: str = ""
    final_values: list[float] = field(default_factory=list)
    final_reasoning: str = ""


@dataclass
class CalibrationGuidelines:
    """Accumulated guidelines from backtesting (master G)."""
    rules: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        if not self.rules:
            return ""
        joined = "\n".join(f"- {r}" for r in self.rules)
        return f"**Calibration Guidelines:**\n{joined}"


# ---------------------------------------------------------------------------
# LLM Client (Anthropic)
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper around the Anthropic Messages API.

    Paper §4.1: sampling temperature = 0.1 for deterministic outputs.
    Model: Claude-4.5-Sonnet (Anthropic [2025]).
    """

    # Fix 5: model string matches the paper's referenced model
    MODEL = "claude-sonnet-4-5"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.1

    def __init__(self):
        self.client = anthropic.Anthropic()

    def call(self, system: str, user: str) -> str:
        """Return the text content of the first response block."""
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    temperature=self.TEMPERATURE,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return response.content[0].text
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}): {exc}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)
        raise RuntimeError("LLM call failed after 3 attempts.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_ts_features(values: list[float]) -> str:
    """Return a short summary of basic time-series features."""
    if not values:
        return "No data."
    n = len(values)
    mean_v = sum(values) / n
    variance = sum((v - mean_v) ** 2 for v in values) / n
    std_v = math.sqrt(variance)
    trend = (
        "upward" if values[-1] > values[0]
        else "downward" if values[-1] < values[0]
        else "flat"
    )
    min_v, max_v = min(values), max(values)
    if n > 1:
        diffs = [values[i] - values[i - 1] for i in range(1, n)]
        mean_d = sum(diffs) / len(diffs)
        momentum = (
            "positive" if mean_d > 0 else "negative" if mean_d < 0 else "neutral"
        )
    else:
        momentum = "unknown"
    return (
        f"Length={n}, Mean={mean_v:.4f}, Std={std_v:.4f}, "
        f"Min={min_v:.4f}, Max={max_v:.4f}, "
        f"Overall Trend={trend}, Recent Momentum={momentum}"
    )


def build_history_str(window: TimeSeriesWindow) -> str:
    lines = []
    for ts, val, txt in zip(window.timestamps, window.values, window.texts):
        line = f"{ts}: Value={val:.4f}"
        if txt.strip():
            line += f" | Context: {txt.strip()}"
        lines.append(line)
    return "\n".join(lines)


def parse_forecasted_values(text: str, horizon: int) -> list[float]:
    """Extract numeric array from <forecasted_values> tag or fallback."""
    m = re.search(r"<forecasted_values>\s*\[([^\]]+)\]", text, re.DOTALL)
    if not m:
        m = re.search(r"\[([0-9.,\s\-]+)\]", text)
    if m:
        raw = m.group(1)
        nums = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(nums) == horizon:
            return nums
        if nums:
            if len(nums) < horizon:
                nums += [nums[-1]] * (horizon - len(nums))
            return nums[:horizon]
    logger.warning("Could not parse forecasted_values; using zero fallback.")
    return [0.0] * horizon


def parse_reasoning(text: str) -> str:
    m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def compute_mape(predicted: list[float], actual: list[float]) -> float:
    pairs = [(p, a) for p, a in zip(predicted, actual) if a != 0.0]
    if not pairs:
        return float("nan")
    return sum(abs(p - a) / abs(a) for p, a in pairs) / len(pairs)


def compute_rmse(predicted: list[float], actual: list[float]) -> float:
    n = len(predicted)
    if n == 0:
        return float("nan")
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(predicted, actual)) / n)


# ---------------------------------------------------------------------------
# Stage 1 – Historical Context Agent  (A_ctx)
# ---------------------------------------------------------------------------

_CTX_SYSTEM = (
    "You are an expert causal analysis agent. Your goal is to identify key events "
    "from historical text and analyze how they historically impacted the target variable. "
    "Your knowledge cutoff date is January 2025."
)

_CTX_USER_TMPL = """\
**Task:**
Read the historical data, extract the key explicit and implicit factors, and explain \
how they historically impacted the values of the target variable "{target_name}".

**Basic Time Series Features**
{ts_features}

**Domain:** {domain}

**Historical Data (Text & Values):**
{history_str}

**Output:**
You must output a structured timeline that chronologically tracks the values and the \
events that drove them.

For each timestamp, output exactly in this format:

Date/Timestamp: [Date]
Value: [Value]
Textual Content: [A concise, organized summary covering ALL events and factors that \
influenced the value change]

You must explicitly make sure that you DO NOT miss any single fact, event, or detail \
listed under any single timestamp/date in the provided history. Ensure the \
"Textual Content" is well-organized and covers all relevant information from the raw history.
"""


class HistoricalContextAgent:
    """A_ctx — transforms raw multimodal history into structured H_{1:τ}."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, window: TimeSeriesWindow) -> str:
        ts_features = compute_ts_features(window.values)
        history_str = build_history_str(window)
        user_prompt = _CTX_USER_TMPL.format(
            target_name=window.target_name,
            ts_features=ts_features,
            domain=window.domain,
            history_str=history_str,
        )
        logger.info("  [A_ctx] Running Historical Context Agent…")
        return self.llm.call(_CTX_SYSTEM, user_prompt)


# ---------------------------------------------------------------------------
# Stage 2a – Macro-Reasoning Agent  (A_macro)
# ---------------------------------------------------------------------------

_MACRO_SYSTEM = (
    "You are a contextual numerical forecasting agent. Your task is to predict future "
    "values based on the provided historical context. Your knowledge cutoff date is January 2025."
)

_MACRO_USER_TMPL = """\
**Task:**
Predict the next {horizon} values for the target variable "{target_name}" using the \
provided historical context.

**Historical Context:**
{history_str}

**Instructions:**
First, carefully think step by step. Provide your chain of thought that explicitly \
breaks down your reasoning across the {horizon} future steps.

Put this exhaustive, full step-by-step reasoning inside <reasoning> tags.

Then, based on that reasoning, output the final predicted numerical values as an array \
inside <forecasted_values> tags. You must predict exactly {horizon} values.

**Output Format:**
<reasoning>
[Full step by step reasoning here in full detail breaking down the reasoning over future time steps in the horizon]
</reasoning>
<forecasted_values>
[10.5, 11.2, 12.1, ...]
</forecasted_values>
"""


class MacroReasoningAgent:
    """A_macro — top-down coarse forecast over the full horizon T."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self, structured_history: str, window: TimeSeriesWindow
    ) -> tuple[list[float], str]:
        horizon = len(window.future_timestamps)
        user_prompt = _MACRO_USER_TMPL.format(
            horizon=horizon,
            target_name=window.target_name,
            history_str=structured_history,
        )
        logger.info("  [A_macro] Running Macro-Reasoning Agent…")
        raw = self.llm.call(_MACRO_SYSTEM, user_prompt)
        values = parse_forecasted_values(raw, horizon)
        reasoning = parse_reasoning(raw)
        return values, reasoning


# ---------------------------------------------------------------------------
# Stage 2b – Micro-Reasoning Agent  (A_micro)
# ---------------------------------------------------------------------------

_MICRO_SYSTEM = (
    "You are a forecasting agent. Your goal is to predict future events/factors and target "
    "values based on the provided historical context. Your knowledge cutoff date is January 2025."
)

# Fix 2: date field matches paper's Appendix B.3 exactly —
#   "YYYY-MM-DD (Day of Week) (or similar format to that of historical dates)"
_MICRO_USER_TMPL = """\
**Task:**
Predict the next {horizon} values and events for the target variable "{target_name}".

**Context:**
- Forecast Horizon: {horizon} steps at a step size of "{frequency}" each step
- Step Size / Frequency: {frequency}

**Historical Data:**
{history_str}

**Required Output Format:**
You must return a single valid JSON object containing a "timestamp_forecasts" list.
The keys must match exactly, and you must generate forecast for every timestamp \
at frequency of {horizon} steps.

{{
  "timestamp_forecasts": [
    {{
      "timestamp": 1,
      "date": "YYYY-MM-DD (Day of Week) (or similar format to that of historical dates)",
      "day_info": "factor or event...",
      "reasoning": {{
        "movement_label": "Up / Down / Stable",
        "key_drivers": "Concise explanation of the primary factor driving this value."
      }},
      "adjusted_forecast_value": 123.45
    }}
  ]
}}

Return ONLY valid JSON. No markdown fences, no extra text.
"""


def parse_micro_output(text: str, horizon: int) -> tuple[list[float], str]:
    """Parse micro-agent JSON output → (values, reasoning_summary)."""
    clean = re.sub(r"```json|```", "", text).strip()
    try:
        data = json.loads(clean)
        forecasts = data.get("timestamp_forecasts", [])
        values: list[float] = []
        reasoning_parts: list[str] = []
        for item in forecasts:
            values.append(float(item.get("adjusted_forecast_value", 0.0)))
            r = item.get("reasoning", {})
            reasoning_parts.append(
                f"Step {item.get('timestamp', '?')} ({item.get('date', '')}): "
                f"{r.get('movement_label', '?')} — {r.get('key_drivers', '')}"
            )
        if len(values) < horizon:
            values += [values[-1] if values else 0.0] * (horizon - len(values))
        return values[:horizon], "\n".join(reasoning_parts)
    except json.JSONDecodeError:
        logger.warning("Micro agent returned invalid JSON; falling back to regex extraction.")
        nums = re.findall(r'"adjusted_forecast_value"\s*:\s*([\d.]+)', text)
        values = [float(x) for x in nums]
        if not values:
            values = [0.0] * horizon
        if len(values) < horizon:
            values += [values[-1]] * (horizon - len(values))
        return values[:horizon], text[:500]


class MicroReasoningAgent:
    """A_micro — granular, step-by-step forecast for each future timestep."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self, structured_history: str, window: TimeSeriesWindow
    ) -> tuple[list[float], str]:
        horizon = len(window.future_timestamps)
        user_prompt = _MICRO_USER_TMPL.format(
            horizon=horizon,
            target_name=window.target_name,
            frequency=window.frequency,
            history_str=structured_history,
        )
        logger.info("  [A_micro] Running Micro-Reasoning Agent…")
        raw = self.llm.call(_MICRO_SYSTEM, user_prompt)
        return parse_micro_output(raw, horizon)


# ---------------------------------------------------------------------------
# Stage 3a – Forecast Synthesizer / Value Predictor Agent  (A_syn)
# ---------------------------------------------------------------------------

_SYNTHESIZER_SYSTEM = (
    "You are a forecasting agent. Your task is to predict future values based on the "
    "provided historical context. For reference, you have access to forecasts from "
    "high-level macro reasoning and granular micro reasoning which you can utilize for "
    "final forecast. Your knowledge cutoff date is January 2025."
)

# Fix 1: {event_predictions_section} restored between Historical Data and
#         Macro-Reasoning inputs, exactly as in paper Appendix B.5.
_SYNTHESIZER_USER_TMPL = """\
**Task:**
Predict the final numerical value of **{target_name}** for the next {horizon} steps. \
You are provided with a Macro-Reasoning outlook and a Micro-Reasoning step-by-step \
breakdown. Think and reason very carefully for giving the final output.

**Context:**
- Forecast Horizon: {horizon} steps at a step size of "{frequency}"
- Step Size / Frequency: {frequency}
- Required Future Dates: {future_dates}

**Inputs:**
1. Historical Data:
{history_str}
{event_predictions_section}
2. Macro-Reasoning Outlook (Overarching Logic & Values):
{macro_reasoning_str}

3. Micro-Reasoning Breakdown (Step-by-Step Events & Values):
{micro_reasoning_str}

{guidelines_section}

**Instructions:**
First, carefully think step by step. You MUST provide step-by-step chain of thought \
that explicitly breaks down your reasoning for each of the {horizon} future steps. \
Explain how you adjusted the Macro and Micro perspectives to arrive at your final value.

Put this exhaustive, full step-by-step reasoning inside <reasoning> tags.

Then, based on that reasoning, output the final predicted numerical values as an array \
inside <forecasted_values> tags. You must predict exactly {horizon} values.

**Output Format:**
<reasoning>
[Full step by step reasoning here in full detail breaking down the reasoning over future time steps in the horizon]
</reasoning>
<forecasted_values>
[10.5, 11.2, 12.1, ...]
</forecasted_values>
"""


class ForecastSynthesizerAgent:
    """A_syn — merges macro + micro outlooks into the final forecast."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self,
        structured_history: str,
        window: TimeSeriesWindow,
        macro_values: list[float],
        macro_reasoning: str,
        micro_values: list[float],
        micro_reasoning: str,
        guidelines: CalibrationGuidelines,
        event_predictions: str = "",   # Fix 1: new optional input
    ) -> tuple[list[float], str]:
        horizon = len(window.future_timestamps)

        macro_str = (
            f"Macro Predicted Values: {macro_values}\n"
            f"Macro Reasoning:\n{macro_reasoning}"
        )
        micro_str = (
            f"Micro Predicted Values: {micro_values}\n"
            f"Micro Step-by-Step Reasoning:\n{micro_reasoning}"
        )

        guidelines_section = guidelines.as_text() if guidelines.rules else ""

        # Fix 1: include event_predictions_section when provided
        event_predictions_section = (
            f"\n{event_predictions.strip()}\n" if event_predictions.strip() else ""
        )

        user_prompt = _SYNTHESIZER_USER_TMPL.format(
            target_name=window.target_name,
            horizon=horizon,
            frequency=window.frequency,
            future_dates=", ".join(window.future_timestamps),
            history_str=structured_history,
            event_predictions_section=event_predictions_section,
            macro_reasoning_str=macro_str,
            micro_reasoning_str=micro_str,
            guidelines_section=guidelines_section,
        )

        logger.info("  [A_syn] Running Forecast Synthesizer (Value Predictor) Agent…")
        raw = self.llm.call(_SYNTHESIZER_SYSTEM, user_prompt)
        values = parse_forecasted_values(raw, horizon)
        reasoning = parse_reasoning(raw)
        return values, reasoning


# ---------------------------------------------------------------------------
# Stage 3b – Calibration Agent  (A_calib)
# ---------------------------------------------------------------------------

_CALIBRATION_SYSTEM = (
    "You are a Forecasting Strategy Optimizer. Your goal is to analyze past predictions "
    "against actual ground truth and generate specific 'Review Guidelines' for a Sanity "
    "Check Agent to enforce in future predictions."
)

_CALIBRATION_USER_TMPL = """\
**Task:**
Critique the Value Predictor Agent's reasoning and numerical predictions against the \
Ground Truth. Focus primarily on how it translated its reasoning into actual numerical \
values (e.g., did it overestimate/underestimate impacts?).

**1. Agent's Prompt:**
{value_predictor_prompt}

**2. Agent's Output:**
Reasoning: {agent_reasoning}
Predicted Values: {agent_values}
Error (MAPE): {agent_error:.4f}

**3. Upstream Performance (For Context):**
Macro-Reasoning MAPE: {macro_mape:.4f}
Micro-Reasoning MAPE: {micro_mape:.4f}

**4. Ground Truth:**
Events: {actual_events_summary}
Values: {actual_values}

**Output Format:**
1. <diagnosis>: A brief critique of the agent's numerical calibration and logical flaws.
2. <guidelines>: A single, short paragraph of robust advice for future predictions to \
fix these numerical estimation errors. Make sure the guidelines are generalized rather \
than too specific which may not apply to future scenarios.
"""


class CalibrationAgent:
    """A_calib — analyses prediction errors and emits generalised guidelines."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self,
        value_predictor_prompt: str,
        agent_reasoning: str,
        agent_values: list[float],
        actual_values: list[float],
        actual_events: str,
        macro_values: list[float],
        micro_values: list[float],
    ) -> str:
        agent_error = compute_mape(agent_values, actual_values)
        macro_mape = compute_mape(macro_values, actual_values)
        micro_mape = compute_mape(micro_values, actual_values)

        user_prompt = _CALIBRATION_USER_TMPL.format(
            value_predictor_prompt=value_predictor_prompt[:1000],
            agent_reasoning=agent_reasoning[:800],
            agent_values=agent_values,
            agent_error=agent_error,
            macro_mape=macro_mape,
            micro_mape=micro_mape,
            actual_events_summary=actual_events[:500],
            actual_values=actual_values,
        )

        logger.info("  [A_calib] Running Calibration Agent…")
        raw = self.llm.call(_CALIBRATION_SYSTEM, user_prompt)

        m = re.search(r"<guidelines>(.*?)</guidelines>", raw, re.DOTALL)
        return m.group(1).strip() if m else raw.strip()


# ---------------------------------------------------------------------------
# Calibration Loop  (§3.3)
# ---------------------------------------------------------------------------

def _intersect_guidelines(lists_of_guidelines: list[list[str]]) -> list[str]:
    """
    Fix 3: TRUE INTERSECTION (∩) across folds.

    Paper §3.3: G = ∩_{i=1}^{n-1} G_i

    A guideline is retained in the master set only if it appears in
    EVERY training fold's guideline output.  We use exact string match
    after normalising whitespace.
    """
    if not lists_of_guidelines:
        return []
    # Normalise: strip leading/trailing whitespace
    normalised = [
        {g.strip() for g in fold_guidelines}
        for fold_guidelines in lists_of_guidelines
    ]
    # Intersection across all folds
    common: set[str] = normalised[0]
    for fold_set in normalised[1:]:
        common = common & fold_set
    # Preserve a stable order (use first fold's ordering for those that survived)
    first_fold_order = [g.strip() for g in lists_of_guidelines[0]]
    return [g for g in first_fold_order if g in common]


def run_calibration_loop(
    nexus: "NEXUSFramework",
    historical_windows: list[TimeSeriesWindow],
    actual_futures: list[list[float]],
    n_splits: int = 6,
    min_improvement_pct: float = 0.05,
) -> CalibrationGuidelines:
    """
    Implements the backtesting calibration loop from §3.3.

    Fix 3: Guidelines are INTERSECTED (∩), not union-deduplicated.
    Fix 4: Training-fold predictions run in PARALLEL via ThreadPoolExecutor.

    Steps:
      1. Divide historical_windows into n_splits sequential folds.
      2. Held-out = last fold; training = all preceding folds.
      3. Generate per-fold predictions IN PARALLEL.
      4. Generate per-fold guidelines (A_calib) from prediction errors.
      5. Intersect all training-fold guidelines → master G.
      6. Validate G on held-out fold; apply only if MAPE improves ≥ min_improvement_pct.
    """
    n = len(historical_windows)
    if n < 2:
        logger.warning("Not enough windows for calibration; skipping.")
        return CalibrationGuidelines()

    splits = min(n_splits, n)
    fold_size = max(1, n // splits)
    training_windows = historical_windows[: n - fold_size]
    training_actuals = actual_futures[: n - fold_size]
    val_window = historical_windows[n - fold_size]
    val_actual = actual_futures[n - fold_size]

    # ------------------------------------------------------------------
    # Fix 4: Run all training-fold predictions IN PARALLEL
    # ------------------------------------------------------------------
    logger.info(
        f"[Calibration] Running {len(training_windows)} training folds IN PARALLEL…"
    )

    empty_guidelines = CalibrationGuidelines()

    def _run_fold(idx_win_act):
        idx, win, act = idx_win_act
        logger.info(f"  Fold {idx + 1}/{len(training_windows)} (parallel)")
        outputs = nexus._run_single(win, empty_guidelines)
        return idx, outputs, act

    fold_results: dict[int, tuple] = {}  # idx → (outputs, actuals)
    with ThreadPoolExecutor(max_workers=min(len(training_windows), 8)) as executor:
        futures = {
            executor.submit(_run_fold, (idx, win, act)): idx
            for idx, (win, act) in enumerate(
                zip(training_windows, training_actuals)
            )
        }
        for future in as_completed(futures):
            idx, outputs, act = future.result()
            fold_results[idx] = (outputs, act)

    # ------------------------------------------------------------------
    # Generate per-fold guidelines (sequential — each needs prior outputs)
    # ------------------------------------------------------------------
    all_fold_guidelines: list[list[str]] = []

    for idx in sorted(fold_results.keys()):
        outputs, act = fold_results[idx]
        win = training_windows[idx]
        guideline_text = nexus.calibration_agent.run(
            value_predictor_prompt=(
                f"Predict {win.target_name} for {len(win.future_timestamps)} steps"
            ),
            agent_reasoning=outputs.final_reasoning,
            agent_values=outputs.final_values,
            actual_values=act,
            actual_events="(training fold events)",
            macro_values=outputs.macro_values,
            micro_values=outputs.micro_values,
        )
        # Each guideline text is treated as one rule entry per fold
        all_fold_guidelines.append([guideline_text])

    # ------------------------------------------------------------------
    # Fix 3: True intersection of guidelines across folds
    # ------------------------------------------------------------------
    intersected = _intersect_guidelines(all_fold_guidelines)

    # If intersection is empty (no guideline appeared in every fold),
    # fall back gracefully to the full union — better than no guidance.
    # The paper does not explicitly handle this edge case; we log a warning.
    if not intersected:
        logger.warning(
            "[Calibration] Intersection of guidelines is empty "
            "(no guideline common across ALL folds). "
            "Falling back to union for robustness."
        )
        seen: dict[str, None] = {}
        for fold_g in all_fold_guidelines:
            for g in fold_g:
                seen[g] = None
        intersected = list(seen.keys())

    candidate = CalibrationGuidelines(rules=intersected)

    # ------------------------------------------------------------------
    # Validate on held-out fold
    # ------------------------------------------------------------------
    logger.info("[Calibration] Validating guidelines on held-out fold…")
    baseline_outputs = nexus._run_single(val_window, CalibrationGuidelines())
    baseline_mape = compute_mape(baseline_outputs.final_values, val_actual)

    guided_outputs = nexus._run_single(val_window, candidate)
    guided_mape = compute_mape(guided_outputs.final_values, val_actual)

    improvement = (baseline_mape - guided_mape) / (baseline_mape + 1e-9)
    logger.info(
        f"[Calibration] Baseline MAPE={baseline_mape:.4f}, "
        f"Guided MAPE={guided_mape:.4f}, "
        f"Improvement={improvement * 100:.1f}% "
        f"(threshold={min_improvement_pct * 100:.0f}%)"
    )

    if improvement >= min_improvement_pct:
        logger.info("[Calibration] Guidelines ACCEPTED.")
        return candidate
    else:
        logger.info("[Calibration] Guidelines did not improve enough; DISCARDED.")
        return CalibrationGuidelines()


# ---------------------------------------------------------------------------
# NEXUS Framework  (top-level orchestrator)
# ---------------------------------------------------------------------------

class NEXUSFramework:
    """
    Full NEXUS multi-agent forecasting framework.

    F(X_{1:τ}, E_{1:τ}) → (X_{τ+1:τ+T}, R)

    Usage (single window):
        nexus  = NEXUSFramework()
        result = nexus.forecast(window)

    Usage (with calibration):
        result, guidelines = nexus.forecast_with_calibration(windows, actuals)
    """

    def __init__(self):
        self.llm               = LLMClient()
        self.ctx_agent         = HistoricalContextAgent(self.llm)
        self.macro_agent       = MacroReasoningAgent(self.llm)
        self.micro_agent       = MicroReasoningAgent(self.llm)
        self.synthesizer       = ForecastSynthesizerAgent(self.llm)
        self.calibration_agent = CalibrationAgent(self.llm)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def forecast(
        self,
        window: TimeSeriesWindow,
        guidelines: Optional[CalibrationGuidelines] = None,
        event_predictions: str = "",
    ) -> AgentOutputs:
        """
        Run the full NEXUS pipeline for a single forecasting window.
        """
        if guidelines is None:
            guidelines = CalibrationGuidelines()
        return self._run_single(window, guidelines, event_predictions)

    def forecast_with_calibration(
        self,
        windows: list[TimeSeriesWindow],
        actuals: list[list[float]],
        n_splits: int = 6,
        min_improvement_pct: float = 0.05,
    ) -> tuple[AgentOutputs, CalibrationGuidelines]:
        """
        Run calibration loop on historical windows then forecast the last window.
        Returns (final_outputs, learned_guidelines).
        """
        guidelines = run_calibration_loop(
            self,
            windows[:-1],
            actuals[:-1],
            n_splits,
            min_improvement_pct,
        )
        final_outputs = self.forecast(windows[-1], guidelines)
        return final_outputs, guidelines

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _run_single(
        self,
        window: TimeSeriesWindow,
        guidelines: CalibrationGuidelines,
        event_predictions: str = "",
    ) -> AgentOutputs:
        outputs = AgentOutputs()

        # Stage 1: Contextualization
        outputs.structured_history = self.ctx_agent.run(window)

        # Stage 2: Dual-resolution outlook (macro + micro)
        outputs.macro_values, outputs.macro_reasoning = self.macro_agent.run(
            outputs.structured_history, window
        )
        outputs.micro_values, outputs.micro_reasoning = self.micro_agent.run(
            outputs.structured_history, window
        )

        # Stage 3: Synthesis & Calibration
        outputs.final_values, outputs.final_reasoning = self.synthesizer.run(
            structured_history=outputs.structured_history,
            window=window,
            macro_values=outputs.macro_values,
            macro_reasoning=outputs.macro_reasoning,
            micro_values=outputs.micro_values,
            micro_reasoning=outputs.micro_reasoning,
            guidelines=guidelines,
            event_predictions=event_predictions,   # Fix 1
        )

        return outputs


# ---------------------------------------------------------------------------
# Quick smoke-test  (remove / comment out in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime

    # Minimal synthetic window — replace with real data
    base = datetime.date(2025, 1, 6)
    timestamps = [(base + datetime.timedelta(weeks=i)).isoformat() for i in range(8)]
    values = [100.0 + i * 2.5 + (i % 3) * 0.5 for i in range(8)]
    texts = [""] * 8

    future_base = base + datetime.timedelta(weeks=8)
    future_timestamps = [
        (future_base + datetime.timedelta(weeks=i)).isoformat() for i in range(4)
    ]

    window = TimeSeriesWindow(
        timestamps=timestamps,
        values=values,
        texts=texts,
        target_name="Synthetic Asset Price",
        domain="Finance",
        frequency="Weekly",
        future_timestamps=future_timestamps,
    )

    nexus = NEXUSFramework()
    result = nexus.forecast(window)

    print("\n=== NEXUS Forecast ===")
    print(f"Final Values  : {result.final_values}")
    print(f"Macro Values  : {result.macro_values}")
    print(f"Micro Values  : {result.micro_values}")
    print(f"\nFinal Reasoning (truncated):\n{result.final_reasoning[:400]}…")