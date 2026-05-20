# =============================================================================
#  NEXUS: An Agentic Framework for Time Series Forecasting
#  ------------------------------------------------------------
#  Author  : Nihar Mahesh Jani
#  Email   : niharmaheshjani@gmail.com
#
#  Hey! Before you dive in — let me be upfront with you.
#  This is MY independent implementation of the NEXUS paper by Das et al.
#  (arXiv:2605.14389, 2026). It is NOT the official code from the authors.
#  I have done my best to match the paper faithfully, but please review
#  everything carefully before you use it in any serious project.
#  Mistakes are possible — always verify before you trust any output!
#
#  Now, let me walk you through what this does and WHY it is designed
#  the way it is. I find code much easier to use when someone actually
#  explains the thinking behind it, not just dumps functions on you.
# =============================================================================
#
#  THE BIG IDEA
#  ------------
#  Most forecasting tools either crunch numbers (TSFMs) or reason in text
#  (LLMs). Neither alone is great. NEXUS bridges that gap by breaking
#  forecasting into focused stages — like asking different specialists
#  before making a final call.
#
#  Stage 1 → Contextualization        : A_ctx   cleans & structures history
#  Stage 2 → Dual-Resolution Outlook  : A_macro (big picture trend)
#                                      + A_micro (step-by-step events)
#  Stage 3 → Synthesis & Calibration  : A_syn   merges both views
#                                      + A_calib learns from past mistakes
#
#  One pip install. That's it:
#       pip install anthropic
# =============================================================================

import json
import re
import math
import time
import logging
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

# I like clean logs — helps a lot when something goes wrong at 2 AM.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
#  SECTION 1 — Data Structures
#  ----------------------------
#  I kept these as simple dataclasses. No magic, no hidden state.
#  You can see exactly what goes in and what comes out.
# =============================================================================

@dataclass
class TimeSeriesWindow:
    """
    Everything the framework needs to know about one forecasting task.

    Think of it as a folder you hand to an analyst:
      - Here is the history (timestamps + values)
      - Here is the news/context for each date (texts)
      - Here is what we are trying to predict (target_name, domain)
      - Here are the future dates we need values for (future_timestamps)
    """
    timestamps: list[str]        # e.g. ["2025-01-06", "2025-01-13", ...]
    values: list[float]          # numerical series aligned to timestamps
    texts: list[str]             # news / context per timestamp; "" if none
    target_name: str             # e.g. "MSFT closing price"
    domain: str                  # e.g. "Stock Market"
    frequency: str               # e.g. "Weekly"
    future_timestamps: list[str] # the T future steps we need to predict


@dataclass
class AgentOutputs:
    """
    Collects everything each agent produced so nothing gets lost.
    I store intermediate outputs (macro, micro) alongside the final ones
    because they are useful for calibration and for debugging.
    """
    structured_history: str = ""
    macro_values: list[float] = field(default_factory=list)
    macro_reasoning: str = ""
    micro_values: list[float] = field(default_factory=list)
    micro_reasoning: str = ""
    final_values: list[float] = field(default_factory=list)
    final_reasoning: str = ""


@dataclass
class CalibrationGuidelines:
    """
    The 'master guidelines' G that the calibration loop produces.

    The paper derives G by INTERSECTING per-fold guidelines — keeping only
    the advice that generalises across ALL historical splits. I'll explain
    this more when we get to the calibration section.
    """
    rules: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        """Format rules into a prompt-ready block."""
        if not self.rules:
            return ""
        joined = "\n".join(f"- {r}" for r in self.rules)
        return f"**Calibration Guidelines:**\n{joined}"


# =============================================================================
#  SECTION 2 — LLM Client
#  -----------------------
#  A thin wrapper around Anthropic's API. I added retry logic with
#  exponential back-off because API hiccups are real and annoying.
#
#  Temperature = 0.1 as specified in §4.1 of the paper — we want
#  deterministic, reproducible outputs, not creative surprises.
# =============================================================================

class LLMClient:
    """
    Handles all calls to Claude. One place, one responsibility.

    Model  : claude-sonnet-4-5  (Claude-4.5-Sonnet, Anthropic 2025)
    Temp   : 0.1  — highly deterministic, as the paper specifies
    Tokens : 4096 max per response
    """

    MODEL       = "claude-sonnet-4-5"
    MAX_TOKENS  = 4096
    TEMPERATURE = 0.1

    def __init__(self):
        # Reads ANTHROPIC_API_KEY from your environment automatically.
        # Make sure you have set it: export ANTHROPIC_API_KEY=your_key
        self.client = anthropic.Anthropic()

    def call(self, system: str, user: str) -> str:
        """
        Send a prompt, get a response. Retries up to 3 times on failure.
        Returns the plain text of the first content block.
        """
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
                    f"LLM call failed (attempt {attempt + 1}/3): {exc}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)
        raise RuntimeError(
            "LLM call failed after 3 attempts. "
            "Check your API key and network connection."
        )


# =============================================================================
#  SECTION 3 — Shared Helper Functions
#  -------------------------------------
#  Small utilities used by multiple agents. I pulled them out here so
#  each agent stays focused on its own job.
# =============================================================================

def compute_ts_features(values: list[float]) -> str:
    """
    Computes basic statistical features of the time series.

    Why? Because the Historical Context Agent needs a quick numerical
    summary alongside the raw data — it helps the LLM understand the
    overall shape of the series before reading every data point.
    """
    if not values:
        return "No data."

    n      = len(values)
    mean_v = sum(values) / n
    var    = sum((v - mean_v) ** 2 for v in values) / n
    std_v  = math.sqrt(var)
    trend  = (
        "upward"   if values[-1] > values[0] else
        "downward" if values[-1] < values[0] else
        "flat"
    )
    min_v, max_v = min(values), max(values)

    if n > 1:
        diffs    = [values[i] - values[i - 1] for i in range(1, n)]
        mean_d   = sum(diffs) / len(diffs)
        momentum = (
            "positive" if mean_d > 0 else
            "negative" if mean_d < 0 else
            "neutral"
        )
    else:
        momentum = "unknown"

    return (
        f"Length={n}, Mean={mean_v:.4f}, Std={std_v:.4f}, "
        f"Min={min_v:.4f}, Max={max_v:.4f}, "
        f"Overall Trend={trend}, Recent Momentum={momentum}"
    )


def build_history_str(window: TimeSeriesWindow) -> str:
    """
    Merges timestamps, values, and text into one readable block.
    If there is no text for a timestamp, we just show the value.
    """
    lines = []
    for ts, val, txt in zip(window.timestamps, window.values, window.texts):
        line = f"{ts}: Value={val:.4f}"
        if txt.strip():
            line += f" | Context: {txt.strip()}"
        lines.append(line)
    return "\n".join(lines)


def parse_forecasted_values(text: str, horizon: int) -> list[float]:
    """
    Extracts the predicted numbers from the LLM's response.

    Primary  : looks for <forecasted_values>[...]</forecasted_values>
    Fallback : looks for any [...] containing numbers
    Last resort: returns zeros (and logs a warning so you know)
    """
    m = re.search(r"<forecasted_values>\s*\[([^\]]+)\]", text, re.DOTALL)
    if not m:
        m = re.search(r"\[([0-9.,\s\-]+)\]", text)

    if m:
        raw  = m.group(1)
        nums = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(nums) == horizon:
            return nums
        if nums:
            # Pad or trim to match requested horizon
            if len(nums) < horizon:
                nums += [nums[-1]] * (horizon - len(nums))
            return nums[:horizon]

    logger.warning(
        "Could not parse <forecasted_values>. Returning zeros. "
        "This usually means the LLM deviated from the output format."
    )
    return [0.0] * horizon


def parse_reasoning(text: str) -> str:
    """Extracts the chain-of-thought from <reasoning>...</reasoning> tags."""
    m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def compute_mape(predicted: list[float], actual: list[float]) -> float:
    """
    Mean Absolute Percentage Error.
    Skips zero-actual values to avoid division by zero.
    """
    pairs = [(p, a) for p, a in zip(predicted, actual) if a != 0.0]
    if not pairs:
        return float("nan")
    return sum(abs(p - a) / abs(a) for p, a in pairs) / len(pairs)


def compute_rmse(predicted: list[float], actual: list[float]) -> float:
    """Root Mean Square Error."""
    n = len(predicted)
    if n == 0:
        return float("nan")
    return math.sqrt(
        sum((p - a) ** 2 for p, a in zip(predicted, actual)) / n
    )


# =============================================================================
#  SECTION 4 — Stage 1: Historical Context Agent (A_ctx)
#  -------------------------------------------------------
#  This agent does one thing: take messy raw history and turn it into
#  a clean, structured timeline that downstream agents can reason over
#  without getting lost in noise.
#
#  Paper reference: §3.1 — "A_ctx(X_{1:τ}, E_{1:τ}) → H_{1:τ}"
# =============================================================================

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
    """
    A_ctx — the 'librarian' of the framework.

    It reads everything, organises it chronologically, and hands a clean
    structured history H_{1:τ} to the forecasting agents.
    Without this step, the LLMs have to parse messy raw data themselves,
    which hurts their forecasting quality significantly.
    """

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
        logger.info("  [A_ctx] Stage 1 — Structuring historical context…")
        return self.llm.call(_CTX_SYSTEM, user_prompt)


# =============================================================================
#  SECTION 5 — Stage 2a: Macro-Reasoning Agent (A_macro)
#  -------------------------------------------------------
#  A_macro zooms OUT. It looks at the full history and asks:
#  "What is the broad trajectory over the entire forecast horizon?"
#
#  Think of it as the senior analyst who gives you the big-picture view
#  before anyone looks at the day-to-day details.
#
#  Paper reference: §3.2 — "A_macro(H_{1:τ}) → (X^macro_{τ+1:τ+T}, R^macro)"
# =============================================================================

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
    """
    A_macro — the 'big picture' forecaster.

    Top-down. Looks at the full horizon T and establishes the expected
    regime. Does not get distracted by individual step fluctuations.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self, structured_history: str, window: TimeSeriesWindow
    ) -> tuple[list[float], str]:
        horizon     = len(window.future_timestamps)
        user_prompt = _MACRO_USER_TMPL.format(
            horizon=horizon,
            target_name=window.target_name,
            history_str=structured_history,
        )
        logger.info("  [A_macro] Stage 2a — Macro-level trajectory reasoning…")
        raw       = self.llm.call(_MACRO_SYSTEM, user_prompt)
        values    = parse_forecasted_values(raw, horizon)
        reasoning = parse_reasoning(raw)
        return values, reasoning


# =============================================================================
#  SECTION 6 — Stage 2b: Micro-Reasoning Agent (A_micro)
#  -------------------------------------------------------
#  A_micro zooms IN. It walks through the forecast horizon one step at a
#  time, asking: "What specific event or catalyst drives THIS week?"
#
#  Returns a JSON object with per-step dates, events, movement labels,
#  and adjusted forecast values. The JSON structure keeps things precise
#  and machine-parseable.
#
#  Paper reference: §3.2 — "A_micro(H_{1:τ}) → (X^micro_{τ+1:τ+T}, R^micro_{τ+1:τ+T})"
# =============================================================================

_MICRO_SYSTEM = (
    "You are a forecasting agent. Your goal is to predict future events/factors and target "
    "values based on the provided historical context. Your knowledge cutoff date is January 2025."
)

# Note on the date format below:
# The paper's Appendix B.3 specifies exactly this string for the date field.
# I kept it verbatim — do not simplify to "YYYY-MM-DD" only.
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
    """
    Parses the micro-agent's JSON output into (values, reasoning_summary).

    I strip markdown fences first because LLMs sometimes wrap JSON in
    ```json ... ``` even when told not to. Better to handle it than crash.
    """
    clean = re.sub(r"```json|```", "", text).strip()
    try:
        data      = json.loads(clean)
        forecasts = data.get("timestamp_forecasts", [])
        values: list[float]  = []
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
        logger.warning(
            "Micro agent returned invalid JSON. "
            "Falling back to regex extraction — results may be less precise."
        )
        nums = re.findall(r'"adjusted_forecast_value"\s*:\s*([\d.]+)', text)
        values = [float(x) for x in nums]
        if not values:
            values = [0.0] * horizon
        if len(values) < horizon:
            values += [values[-1]] * (horizon - len(values))
        return values[:horizon], text[:500]


class MicroReasoningAgent:
    """
    A_micro — the 'detail-oriented' forecaster.

    Granular, step-by-step. For each future timestep, it thinks about
    immediate catalysts and short-term volatility. Complements A_macro
    by providing the fine-grained view.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self, structured_history: str, window: TimeSeriesWindow
    ) -> tuple[list[float], str]:
        horizon     = len(window.future_timestamps)
        user_prompt = _MICRO_USER_TMPL.format(
            horizon=horizon,
            target_name=window.target_name,
            frequency=window.frequency,
            history_str=structured_history,
        )
        logger.info("  [A_micro] Stage 2b — Micro-level step-by-step reasoning…")
        raw = self.llm.call(_MICRO_SYSTEM, user_prompt)
        return parse_micro_output(raw, horizon)


# =============================================================================
#  SECTION 7 — Stage 3a: Forecast Synthesizer / Value Predictor Agent (A_syn)
#  ---------------------------------------------------------------------------
#  This is where everything comes together. A_syn receives:
#    - The structured history from A_ctx
#    - The broad trajectory from A_macro
#    - The step-by-step breakdown from A_micro
#    - Any calibration guidelines G learned from past errors
#    - Optionally, event predictions for future timestamps
#
#  Its job: weigh all of this and produce the FINAL forecast + reasoning.
#
#  Paper reference: §3.3 — "A_syn(H_{1:τ}, X^macro, R^macro, X^micro, R^micro, G) → (X_{τ+1:τ+T}, R)"
#
#  Important implementation note:
#  The paper's Appendix B.5 includes an {event_predictions_section} between
#  Historical Data and Macro-Reasoning. I make sure this is present here.
# =============================================================================

_SYNTHESIZER_SYSTEM = (
    "You are a forecasting agent. Your task is to predict future values based on the "
    "provided historical context. For reference, you have access to forecasts from "
    "high-level macro reasoning and granular micro reasoning which you can utilize for "
    "final forecast. Your knowledge cutoff date is January 2025."
)

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
    """
    A_syn — the 'final decision maker'.

    It does not blindly average macro and micro. It reasons about how to
    weight them given the specific situation, guided by any calibration
    rules that were learned from past prediction errors.

    The optional `event_predictions` parameter carries future event context
    (e.g. scheduled earnings, macro data releases) — placed between
    Historical Data and Macro-Reasoning as the paper specifies.
    """

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
        event_predictions: str = "",
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

        guidelines_section         = guidelines.as_text() if guidelines.rules else ""
        event_predictions_section  = (
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

        logger.info("  [A_syn] Stage 3 — Synthesizing final forecast…")
        raw       = self.llm.call(_SYNTHESIZER_SYSTEM, user_prompt)
        values    = parse_forecasted_values(raw, horizon)
        reasoning = parse_reasoning(raw)
        return values, reasoning


# =============================================================================
#  SECTION 8 — Stage 3b: Calibration Agent (A_calib)
#  ---------------------------------------------------
#  A_calib looks at past mistakes and asks: "What went wrong, and what
#  generalised advice should the synthesizer follow next time?"
#
#  Key design principle from the paper: the advice must be GENERALISED.
#  Too-specific rules overfit to one historical period and hurt elsewhere.
#
#  Paper reference: §3.3 — calibration loop
# =============================================================================

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
    """
    A_calib — the 'post-mortem analyst'.

    After seeing what actually happened vs. what was predicted, it writes
    a short generalised guideline paragraph. These get collected across
    multiple historical folds and then intersected to keep only the
    advice that holds consistently.
    """

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
        macro_mape  = compute_mape(macro_values,  actual_values)
        micro_mape  = compute_mape(micro_values,  actual_values)

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

        logger.info("  [A_calib] Calibration — analysing prediction errors…")
        raw = self.llm.call(_CALIBRATION_SYSTEM, user_prompt)
        m   = re.search(r"<guidelines>(.*?)</guidelines>", raw, re.DOTALL)
        return m.group(1).strip() if m else raw.strip()


# =============================================================================
#  SECTION 9 — Calibration Loop (§3.3)
#  -------------------------------------
#  This is the most nuanced part of the framework. Let me explain it
#  carefully because it is easy to get wrong.
#
#  The loop:
#    1. Split historical windows into n_splits sequential folds
#    2. Last fold = hidden validation set; rest = training folds
#    3. Run all training-fold predictions IN PARALLEL (ThreadPoolExecutor)
#       — the paper explicitly says "in parallel"
#    4. For each training fold, run A_calib to get guideline text G_i
#    5. INTERSECT all G_i → master G (only advice common to ALL folds)
#       — paper §3.3: G = ∩_{i=1}^{n-1} G_i
#    6. Test G on validation fold; keep only if MAPE improves ≥ threshold
#
#  WHY INTERSECTION and not union?
#  Because a guideline that only helped in ONE historical period is
#  probably overfitting to that specific market condition. The intersection
#  keeps only the advice that was useful consistently — that is the
#  generalised wisdom we want.
# =============================================================================

def _intersect_guidelines(lists_of_guidelines: list[list[str]]) -> list[str]:
    """
    Implements G = ∩_{i=1}^{n-1} G_i from §3.3.

    Keeps only guidelines that appear in EVERY fold's output.
    Uses exact string match after normalising whitespace.
    Preserves the ordering from the first fold for stability.
    """
    if not lists_of_guidelines:
        return []

    normalised = [
        {g.strip() for g in fold_guidelines}
        for fold_guidelines in lists_of_guidelines
    ]

    # Start with the first fold's set and intersect all others into it
    common: set[str] = normalised[0]
    for fold_set in normalised[1:]:
        common = common & fold_set

    # Return in the order they appeared in fold 0
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
    Full backtesting calibration loop from §3.3.

    Parameters
    ----------
    nexus                 : the NEXUSFramework instance
    historical_windows    : list of TimeSeriesWindow (training history)
    actual_futures        : ground-truth future values per window
    n_splits              : number of sequential backtest folds (default 6)
    min_improvement_pct   : minimum MAPE improvement to accept guidelines (default 5%)

    Returns
    -------
    CalibrationGuidelines : master guidelines G (empty if validation failed)
    """
    n = len(historical_windows)
    if n < 2:
        logger.warning("Not enough windows for calibration — skipping.")
        return CalibrationGuidelines()

    splits           = min(n_splits, n)
    fold_size        = max(1, n // splits)
    training_windows = historical_windows[: n - fold_size]
    training_actuals = actual_futures[: n - fold_size]
    val_window       = historical_windows[n - fold_size]
    val_actual       = actual_futures[n - fold_size]

    # ------------------------------------------------------------------
    # Step 3: Run training-fold predictions IN PARALLEL
    # The paper says "in parallel" and it makes a real difference on speed
    # when you have 5+ folds each requiring multiple LLM calls.
    # ------------------------------------------------------------------
    logger.info(
        f"[Calibration] Running {len(training_windows)} training folds "
        f"in parallel (max 8 workers)…"
    )

    empty_guidelines = CalibrationGuidelines()

    def _run_fold(args):
        idx, win, act = args
        logger.info(f"  [Fold {idx + 1}/{len(training_windows)}] Running pipeline…")
        outputs = nexus._run_single(win, empty_guidelines)
        return idx, outputs, act

    fold_results: dict[int, tuple] = {}
    with ThreadPoolExecutor(max_workers=min(len(training_windows), 8)) as pool:
        futures = {
            pool.submit(_run_fold, (idx, win, act)): idx
            for idx, (win, act) in enumerate(zip(training_windows, training_actuals))
        }
        for future in as_completed(futures):
            idx, outputs, act = future.result()
            fold_results[idx] = (outputs, act)

    # ------------------------------------------------------------------
    # Step 4: Generate per-fold guidelines (sequential — order matters)
    # ------------------------------------------------------------------
    all_fold_guidelines: list[list[str]] = []

    for idx in sorted(fold_results.keys()):
        outputs, act = fold_results[idx]
        win          = training_windows[idx]

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
        all_fold_guidelines.append([guideline_text])

    # ------------------------------------------------------------------
    # Step 5: TRUE INTERSECTION — not union, not deduplication
    # ------------------------------------------------------------------
    intersected = _intersect_guidelines(all_fold_guidelines)

    if not intersected:
        # Edge case the paper does not explicitly address.
        # With LLM-generated text, exact string matches across folds are rare.
        # Fallback: use union so we have SOMETHING to validate against.
        logger.warning(
            "[Calibration] Intersection is empty — no guideline was identical "
            "across all folds. Falling back to union. "
            "Consider fuzzy matching for production use."
        )
        seen: dict[str, None] = {}
        for fold_g in all_fold_guidelines:
            for g in fold_g:
                seen[g] = None
        intersected = list(seen.keys())

    candidate = CalibrationGuidelines(rules=intersected)

    # ------------------------------------------------------------------
    # Step 6: Validate on held-out fold before committing
    # ------------------------------------------------------------------
    logger.info("[Calibration] Validating master guidelines on held-out fold…")

    baseline_outputs = nexus._run_single(val_window, CalibrationGuidelines())
    baseline_mape    = compute_mape(baseline_outputs.final_values, val_actual)

    guided_outputs   = nexus._run_single(val_window, candidate)
    guided_mape      = compute_mape(guided_outputs.final_values, val_actual)

    improvement = (baseline_mape - guided_mape) / (baseline_mape + 1e-9)

    logger.info(
        f"[Calibration] Baseline MAPE = {baseline_mape:.4f} | "
        f"Guided MAPE = {guided_mape:.4f} | "
        f"Improvement = {improvement * 100:.1f}% "
        f"(threshold = {min_improvement_pct * 100:.0f}%)"
    )

    if improvement >= min_improvement_pct:
        logger.info("[Calibration] Guidelines ACCEPTED — will be used in final forecast.")
        return candidate
    else:
        logger.info(
            "[Calibration] Guidelines did not clear the improvement threshold. "
            "DISCARDED — forecasting without calibration guidelines."
        )
        return CalibrationGuidelines()


# =============================================================================
#  SECTION 10 — NEXUS Framework (Top-Level Orchestrator)
#  -------------------------------------------------------
#  This is the class you actually use. It wires all the agents together
#  and exposes a clean API with two modes:
#
#    nexus.forecast(window)
#        → Single window, no calibration. Fast. Good for quick tests.
#
#    nexus.forecast_with_calibration(windows, actuals)
#        → Full pipeline with backtesting calibration loop. Slower but
#          adapts to your specific domain and data.
#
#  The internal _run_single() method is the pipeline itself —
#  I kept it separate so the calibration loop can call it cleanly.
# =============================================================================

class NEXUSFramework:
    """
    The complete NEXUS multi-agent forecasting framework.

    Implements the full mapping:
        F(X_{1:τ}, E_{1:τ}) → (X_{τ+1:τ+T}, R)

    All agents share one LLMClient instance to avoid redundant API
    client initialisation.

    Quick start:
        nexus  = NEXUSFramework()
        result = nexus.forecast(window)
        print(result.final_values)
        print(result.final_reasoning)
    """

    def __init__(self):
        # One shared LLM client for all agents
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
        Run the full 3-stage NEXUS pipeline for a single window.

        Parameters
        ----------
        window            : the data window to forecast
        guidelines        : calibration guidelines G (optional; empty by default)
        event_predictions : known future events/context (optional)

        Returns
        -------
        AgentOutputs with final_values, final_reasoning, and all intermediate outputs
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
        Run the calibration loop over historical windows, then forecast
        the final window using the learned guidelines.

        Parameters
        ----------
        windows              : list of TimeSeriesWindow (last one = target to forecast)
        actuals              : ground-truth future values for all but the last window
        n_splits             : number of backtest folds (paper default = 6)
        min_improvement_pct  : minimum MAPE improvement needed to keep guidelines (default 5%)

        Returns
        -------
        (AgentOutputs, CalibrationGuidelines)
        """
        guidelines    = run_calibration_loop(
            self, windows[:-1], actuals[:-1], n_splits, min_improvement_pct
        )
        final_outputs = self.forecast(windows[-1], guidelines)
        return final_outputs, guidelines

    # ------------------------------------------------------------------ #
    # Internal pipeline — called by both forecast() and calibration loop
    # ------------------------------------------------------------------ #

    def _run_single(
        self,
        window: TimeSeriesWindow,
        guidelines: CalibrationGuidelines,
        event_predictions: str = "",
    ) -> AgentOutputs:
        """
        Runs the 3-stage pipeline for one window.

        Stage 1 → A_ctx   : structure raw history
        Stage 2 → A_macro : broad trajectory
                  A_micro : step-by-step events
        Stage 3 → A_syn   : synthesize final forecast
        """
        outputs = AgentOutputs()

        # Stage 1: Contextualization
        outputs.structured_history = self.ctx_agent.run(window)

        # Stage 2: Dual-resolution outlook
        outputs.macro_values, outputs.macro_reasoning = self.macro_agent.run(
            outputs.structured_history, window
        )
        outputs.micro_values, outputs.micro_reasoning = self.micro_agent.run(
            outputs.structured_history, window
        )

        # Stage 3: Synthesis
        outputs.final_values, outputs.final_reasoning = self.synthesizer.run(
            structured_history=outputs.structured_history,
            window=window,
            macro_values=outputs.macro_values,
            macro_reasoning=outputs.macro_reasoning,
            micro_values=outputs.micro_values,
            micro_reasoning=outputs.micro_reasoning,
            guidelines=guidelines,
            event_predictions=event_predictions,
        )

        return outputs


# =============================================================================
#  SECTION 11 — Smoke Test
#  ------------------------
#  A minimal synthetic example to verify everything wires together.
#  Replace the data with your own before doing anything real.
#
#  Run it with:   python nexus_nihar.py
# =============================================================================

if __name__ == "__main__":
    import datetime

    print("=" * 60)
    print("  NEXUS Forecasting — Smoke Test")
    print("  Author: Nihar Mahesh Jani (niharmaheshjani@gmail.com)")
    print("=" * 60)

    base       = datetime.date(2025, 1, 6)
    timestamps = [(base + datetime.timedelta(weeks=i)).isoformat() for i in range(8)]
    values     = [100.0 + i * 2.5 + (i % 3) * 0.5 for i in range(8)]
    texts      = [""] * 8

    future_base       = base + datetime.timedelta(weeks=8)
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

    nexus  = NEXUSFramework()
    result = nexus.forecast(window)

    print("\n=== Results ===")
    print(f"Final Forecast : {result.final_values}")
    print(f"Macro Values   : {result.macro_values}")
    print(f"Micro Values   : {result.micro_values}")
    print(f"\nReasoning (first 400 chars):\n{result.final_reasoning[:400]}…")
