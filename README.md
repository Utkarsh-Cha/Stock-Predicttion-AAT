# Stock Price Prediction: ANN vs 1D-CNN vs LSTM

Deep Learning — Alternative Assessment Tool (AAT)

Predicting the next day's closing price from the previous 60 days, using three
neural network architectures compared under identical conditions and against a
naive baseline. Default dataset is NIFTY 50 (`^NSEI`); the pipeline was also
run on TCS (`TCS.NS`) as a cross-check.

---

## Quick start

```bash
~/Desktop/stock-prediction-aat/.venv/bin/jupyter notebook ~/Desktop/stock-prediction-aat/stock_prediction_aat.ipynb
```

Then `Run ▸ Run All Cells`. Confirm the kernel in the top-right reads
**`Python (stock-prediction-aat)`**.

> Use the full path above. Typing plain `jupyter notebook` starts Anaconda's
> copy, which does not have this project's packages installed.

All results are cached, so a full re-run takes seconds. To genuinely retrain,
set `CONFIG["FORCE_RETRAIN"] = True` in Section 1.

`HOW_TO_RUN.md` has a longer, step-by-step version of this with screenshots of
what to expect.

## Verify the installation

```bash
~/Desktop/stock-prediction-aat/.venv/bin/python ~/Desktop/stock-prediction-aat/check_install.py
```

Should end with `ALL CHECKS PASSED`.

---

## What's in the folder

```
stock-prediction-aat/
├── stock_prediction_aat.ipynb   the main deliverable — 87 cells, 8 sections
├── AAT_Team_Guide.pdf           16-page plain-English guide for the group
├── app.py                       optional Streamlit UI (not required by the brief)
├── VIVA_NOTES.md                extended report points + examiner Q&A
├── HOW_TO_RUN.md                beginner's step-by-step run guide
├── README.md                    this file
├── requirements.txt             exact package versions
├── check_install.py             installation verification
├── RESTART_AFTER_REBOOT.pdf     what to do after restarting the Mac
├── .venv/                       virtual environment (exclude from submission)
├── data/                        cached prices, prepared arrays, scalers
├── models/                      30 trained models (.keras) — 15 per ticker
├── results/                     metrics and comparison tables (CSV)
└── plots/                       32 figures (PNG) — 16 per ticker
```

Files are prefixed by ticker: `_NSEI_*` for NIFTY 50, `TCS_NS_*` for TCS.

## Notebook sections

| Section | Contents |
|---|---|
| 1 | Imports and `CONFIG` — every setting in one place |
| 2 | Download, inspect, clean and plot the data |
| 3 | Chronological split, scaling without leakage, sliding windows |
| 4 | The three models, with matched parameter counts |
| 5 | Training: 3 architectures × 5 seeds = 15 runs, with timing |
| 6 | Metrics, the naive baseline, directional accuracy |
| 7 | All plots |
| 8 | Summary, limitations and conclusions |

Section 8 in detail:

| | |
|---|---|
| 8.1 | Which model performed best, and why (architecture-level explanation) |
| 8.2 | Cross-ticker evidence: changing the company flips the ranking |
| 8.3 | Significance check — is any model better than a coin flip? |
| 8.4 | Limitations, stated honestly |
| 8.5 | Bonus: predicting daily *changes* instead of price levels |
| 8.6 | Points for your report and viva |

## Changing the experiment

Everything is driven by `CONFIG` in Section 1:

```python
"TICKER": "^NSEI",   # try "^BSESN", "RELIANCE.NS", "TCS.NS", "AAPL"
"WINDOW": 60,        # try 30 or 90
"SEEDS": [42, 7, 123, 2024, 99],
```

Change a value, set `FORCE_REDOWNLOAD` / `FORCE_RETRAIN` to `True`, re-run.
Nothing else is hard-coded. Output filenames carry the ticker, so results for
different tickers do not overwrite each other.

---

## Headline results (NIFTY 50)

| Model | Params | RMSE | R² | DirAcc | vs baseline |
|---|---|---|---|---|---|
| **Baseline** (tomorrow=today) | 0 | **196.10** | **0.9651** | n/a | 1.00× |
| ANN | 4,529 | 450.86 ± 79.59 | 0.8109 | 49.3% | 2.30× |
| 1D-CNN | 4,209 | 601.53 ± 27.09 | 0.6710 | 49.4% | 3.07× |
| LSTM | 4,385 | 772.84 ± 124.13 | 0.4466 | 53.0% | 3.94× |

Mean ± std over 5 seeds. Parameter spread is 1.08×, so differences are
architectural rather than a matter of model size.

**Three findings:**

1. **No neural network beat "tomorrow = today."** The best was 2.3× worse.
2. **None predicts direction better than chance.** The LSTM's 53.0% is 1.05
   standard errors above 50% — inside the range chance produces (§8.3).
3. **The ranking is caused by extrapolation, not sequence modelling.** Every
   NIFTY test price is above the whole training range, which favours the ANN's
   unbounded `relu` over the LSTM's saturating gates. Two independent checks
   confirm this: on TCS, whose test period sits *inside* its training range,
   the ranking flips to LSTM > ANN > CNN (§8.2); and modelling daily changes
   instead of levels cuts every model to ~1.00–1.03× baseline (§8.5).

See `VIVA_NOTES.md` for the full discussion and likely examiner questions.

---

## A note on `app.py`

`app.py` is an optional Streamlit interface for exploring other tickers
interactively. It is **not required by the assignment brief**, which asks for
the models, the analysis and the plots. Include it only if you want to
demonstrate the extra work; the notebook stands on its own without it.
