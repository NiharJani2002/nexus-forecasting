# NEXUS Forecasting

> Multi-agent LLM framework for time series forecasting — with macro reasoning, micro reasoning, and self-calibration.

Implementation based on the paper:  
**NEXUS: An Agentic Framework for Time Series Forecasting** · Das et al., arXiv:2605.14389

---

## How It Works

```
Historical Data + Text
        │
        ▼
┌───────────────────┐
│  A_ctx            │  Structures raw history into a clean timeline
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
A_macro    A_micro       Broad trajectory  +  Step-by-step events
    └────┬────┘
         │
         ▼
┌───────────────────┐
│  A_syn            │  Merges both views → final forecast + reasoning
└───────────────────┘
         │
         ▼
┌───────────────────┐
│  A_calib          │  Learns from past errors → calibration guidelines
└───────────────────┘
```

---

## Install

```bash
pip install anthropic
```

```bash
export ANTHROPIC_API_KEY=your_key_here
```

---

## Quick Start

```python
from nexus_forecasting import NEXUSFramework, TimeSeriesWindow

window = TimeSeriesWindow(
    timestamps=["2025-01-06", "2025-01-13", "2025-01-20"],
    values=[420.5, 418.2, 423.7],
    texts=["Earnings beat expectations", "Broader market dip", "Recovery on Fed news"],
    target_name="MSFT closing price",
    domain="Stock Market",
    frequency="Weekly",
    future_timestamps=["2025-01-27", "2025-02-03", "2025-02-10"],
)

nexus = NEXUSFramework()
result = nexus.forecast(window)

print("Forecast :", result.final_values)
print("Reasoning:", result.final_reasoning)
```

---

## With Calibration Loop

```python
result, guidelines = nexus.forecast_with_calibration(
    windows=historical_windows,   # list[TimeSeriesWindow]
    actuals=actual_values,        # list[list[float]]
    n_splits=6,
    min_improvement_pct=0.05,
)
```

The calibration loop splits historical data into folds, generates guidelines per fold, intersects them into a master set **G**, and applies them only if they improve validation MAPE by ≥ 5%.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| anthropic | latest |
| Claude model | claude-sonnet-4-5 |

---

## Project Structure

```
nexus-forecasting/
├── nexus_forecasting.py   # Full framework (single file)
└── README.md
```

---

## Disclaimer

> **This is an independent implementation contributed by Nihar Mahesh Jani — not an official release by the original paper authors.**
>
> Every effort has been made to match the paper faithfully, but errors or deviations may exist.
> **Please review the code carefully before using it in any production or research context. Use at your own risk.**

---

## Reference

```bibtex
@article{das2026nexus,
  title   = {NEXUS: An Agentic Framework for Time Series Forecasting},
  author  = {Das, Sarkar Snigdha Sarathi and Goyal, Palash and Parmar, Mihir
             and Peng, Nanyun and Tirumalashetty, Vishy and Li, Chun-Liang
             and Zhang, Rui and Yoon, Jinsung and Pfister, Tomas},
  journal = {arXiv preprint arXiv:2605.14389},
  year    = {2026}
}
```

---

<div align="center">

Made with care by **Nihar Mahesh Jani**  
niharmaheshjani@gmail.com

</div>
