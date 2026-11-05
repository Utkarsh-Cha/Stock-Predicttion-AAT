# Report & Viva Notes
## Stock Price Prediction: ANN vs 1D-CNN vs LSTM

> This is the extended, printable version. The notebook's **§8.6** has a
> condensed set of the same points if you want something shorter to revise
> from. Where the two differ in wording they agree in substance; the numbers
> are identical.

---

## 1. The 60-second summary

> "I compared three neural network architectures — an ANN, a 1D-CNN and an LSTM —
> on predicting the next day's NIFTY 50 closing price from the previous 60 days.
> I matched their parameter counts within 8% and held everything else identical,
> so the only variable was architecture. I ran each one 5 times with different
> seeds and reported mean ± standard deviation.
>
> The key finding is that **none of them beat a naive baseline** that just
> predicts 'tomorrow = today'. The best model, the ANN, was 2.3× worse on RMSE.
> None of them could predict direction better than a coin flip.
>
> I diagnosed the cause as extrapolation — all test prices lie above the
> training range — and tested that diagnosis by re-running on daily *changes*
> instead of price levels. Every model improved dramatically, the LSTM most,
> and the ranking reversed. That confirmed the diagnosis, but even then no
> model beat the baseline."

---

## 2. Numbers to memorise

### The data
| Fact | Value |
|---|---|
| Ticker / period | ^NSEI (NIFTY 50), 2016-11-04 to 2026-11-03 |
| Trading days | 2,466 |
| Price range | ₹7,610 → ₹26,329 |
| Mean daily change | **+0.047%** |
| Std of daily change | **1.02%** |
| **Signal-to-noise** | **noise is ~20× the signal** |
| Split | 70/15/15 chronological = 1,726 / 369 / 371 days |
| Windows after splitting | 1,666 / 309 / 311 (120 lost at boundaries, 5%) |

### The models
| Model | Params | Architecture |
|---|---|---|
| ANN | 4,529 | Flatten → Dense48 → Dense32 → Dense1 |
| 1D-CNN | 4,209 | Conv16 → Pool → Conv16 → Pool → Flatten → Dense16 → Dense1 |
| LSTM | 4,385 | LSTM32 → Dense1 |

Spread = **1.08×**. That is the fairness evidence.

### Main results (price levels)
| Model | RMSE | MAE | MAPE | R² | DirAcc | vs baseline |
|---|---|---|---|---|---|---|
| **Baseline** | **196.10** | **148.13** | **0.611%** | **0.9651** | n/a | 1.00× |
| ANN | 450.86 ± 79.59 | 349.45 | 1.441% | 0.8109 | 49.3% | 2.30× |
| 1D-CNN | 601.53 ± 27.09 | 467.50 | 1.928% | 0.6710 | 49.4% | 3.07× |
| LSTM | 772.84 ± 124.13 | 675.33 | 2.708% | 0.4466 | 53.0% | 3.94× |

### Timing
| Model | Epochs | Sec/epoch | Total | Inference |
|---|---|---|---|---|
| ANN | 20.2 | 0.062 | 1.22s | 0.076 ms |
| 1D-CNN | 31.6 | 0.075 | 2.35s | 0.078 ms |
| LSTM | 14.8 | **0.295** | 4.36s | 0.110 ms |

**LSTM is 4.8× slower per epoch** despite fewer parameters than the ANN.

### Bonus (daily changes)
| Model | Levels | Changes | vs baseline |
|---|---|---|---|
| Baseline | 196.10 | 196.10 | 1.000× |
| ANN | 450.86 | 201.45 ± 2.88 | 1.027× |
| 1D-CNN | 601.53 | 201.04 ± 1.46 | 1.025× |
| LSTM | 772.84 | **196.77 ± 0.09** | **1.003×** |

**The ranking reversed.** LSTM went from worst to best.

### Cross-ticker check (notebook §8.2)

The extrapolation explanation was tested a second, independent way: by running
the identical pipeline on a stock whose test period sits *inside* its training
range.

| Ticker | Test data vs training range | Ranking (best → worst) |
|---|---|---|
| NIFTY 50 | entirely **above** (scaled 1.16–1.49) | ANN > CNN_1D > **LSTM** |
| TCS | entirely **inside** (scaled 0.43–0.98) | **LSTM** > ANN > CNN_1D |

**The LSTM went from last to first.** Two honest caveats to state yourself
before an examiner raises them: on TCS the LSTM-vs-ANN gap (36.5) is *smaller*
than the combined standard deviation (41.8), so the correct claim is "no
longer worst", not "best" — its lead over the CNN (200.2 vs noise 65.4) is the
part that holds. And two tickers is a consistency check, not a study.

**Why this matters more than the bonus section.** The bonus experiment changed
the *target representation* to remove extrapolation. The cross-ticker check
changed the *dataset* and showed the explanation still predicts correctly on
data it was never fitted to. That is a stronger test of an explanation.

---

## 3. Likely examiner questions, with answers

### On methodology

**Q: Why didn't you shuffle before splitting?**
Because order is the entire signal in a time series. If I shuffled, a window
from March 2019 could land in test while February and April 2019 stayed in
training — the model would be interpolating inside a period it has already
seen from both sides. That is filling a gap, not forecasting. Real prediction
never gets to see the future on both sides.

**Q: Where could data leakage creep into this project?**
Three places, and I handled all three:
1. **The scaler.** Fitting MinMaxScaler on the full series would let it learn
   the maximum price including the test period — the model would get a hint
   about how high the index eventually climbed. I used `fit_transform` on
   train and plain `transform` on val and test.
2. **Shuffling.** Addressed above.
3. **Windows across split boundaries.** I built windows *inside* each split, so
   a window can never contain training days as input and a test day as target.
   This costs 120 windows (5%), which I consider cheap for a guarantee.

**Q: Why forward fill for missing values, not interpolation?**
Forward fill is *causal* — it only looks backwards. Interpolation or backward
fill would fill a missing Tuesday using Wednesday's price, leaking future
information into the past. Forward fill also matches reality: if no trade
occurred, the last traded price is the best available estimate.

**Q: Why run 5 seeds?**
Because neural networks start from random weights, so one run confounds
"this architecture is better" with "this run got a lucky initialisation".
With 5 seeds I report mean ± std, and I only claim a difference is real when
the gap exceeds the combined standard deviations. My notebook checks this
automatically.

**Q: How did you make the comparison fair?**
Matched parameter counts (4,209–4,529, a 1.08× spread), identical input
tensors of shape (60, 1), same optimizer (Adam, lr=0.001), same loss (MSE),
same batch size (32), same max epochs (100), same EarlyStopping (patience 10
on val_loss), same 5 seeds. The only variable is the architecture.

### On the architectures

**Q: Why does the LSTM have 4,352 parameters for only 32 units?**
`4 × ((input_dim + units) × units + units)` = `4 × ((1 + 32) × 32 + 32)` =
`4 × 1,088` = `4,352`. The factor of 4 is the three gates (forget, input,
output) plus the candidate cell update, each with its own weight matrix.

**Q: What does the ANN lose by flattening?**
All structural knowledge that the inputs are ordered in time. After `Flatten`,
day 37 and day 38 are just two unrelated features. The ANN can still learn
the relationship, but it must learn it separately at each of the 60 positions.
A CNN gets this free through weight sharing; an LSTM gets it free through
recurrence.

**Q: What do the CNN's filters capture?**
Each filter is 3 weights plus a bias sliding along the time axis — a small
local pattern detector. They specialise into shapes like "rose then fell",
"three days of steady increase", "a one-day spike". The second Conv1D layer
sees the first layer's outputs, so it composes short shapes into longer ones.
MaxPooling then makes the network care *that* a pattern occurred more than
exactly *when*.

**Q: Why is the LSTM so much slower despite having fewer parameters?**
Sequential dependency. A Conv1D processes all 60 timesteps simultaneously —
independent operations the CPU runs in parallel. An LSTM cannot: the state at
day 37 requires the state at day 36. Sixty strictly sequential steps cannot be
parallelised. Cost here is driven by architecture, not parameter count.

**Q: You expected the LSTM to win. Why did the ANN win instead?**
Because the test set required **extrapolation**, not sequence modelling. Every
test price is above the entire training range (scaled 1.16–1.49, trained on
0–1). The ANN's `relu` is piecewise linear and unbounded, so it extends its
learned trend outward. The LSTM's gates use `sigmoid` and `tanh`, which
**saturate** — push them past their trained range and they flatten. So I was
really measuring "which architecture extrapolates best", not "which models
sequences best". My bonus experiment removed the extrapolation requirement,
and the ranking reversed — LSTM became best. That confirms the explanation.

### On the results

**Q: Your R² is 0.81. That's good, isn't it?**
No — and this is the most important point in my project. The naive baseline
scores **R² = 0.9651**, higher than all three models. R² measures improvement
over "always predict the mean", and on a strongly trending series the mean is
a terrible predictor, so anything that roughly tracks the price scores well.
A high R² here measures how trendy the data is, not how good the model is.

**Q: Why does the baseline matter so much?**
Because without it the results are unfalsifiable. "RMSE ₹451, R² 0.81" sounds
respectable in isolation. The baseline shows a zero-parameter rule achieves
₹196 and R² 0.965. The baseline is the difference between an honest conclusion
and an inflated one.

**Q: The LSTM got 53% directional accuracy. It can predict direction!**
No, and I checked this specifically. With 311 test days, the standard error of
a coin flip is `sqrt(0.25/311)` ≈ **2.84 percentage points**. 53.0% is only
**1.05 standard errors** above 50% — well inside the 95% range for random
guessing (44.4%–55.6%). The ±0.9 in my table is *seed-to-seed* variation,
which is a different quantity from sampling error; confusing the two is an
easy way to over-claim. I cannot distinguish any of my models from a coin flip.

**Q: Why do the predicted lines look so smooth?**
Because the models minimise MSE, which punishes large errors quadratically.
When the next move is essentially unpredictable, the MSE-optimal output is
close to the conditional mean — which here is "roughly no change, smoothed".
The models aren't failing at their objective; they're succeeding at it. The
objective just can't produce a useful forecast at this signal-to-noise ratio.

**Q: Could you make this model profitable?**
No, and I would not claim otherwise. Profit depends on direction, and none of
my models beat a coin flip on direction. A ~1.9% MAPE sounds impressive but
the naive baseline achieves 0.61%. Trading would also face transaction costs,
slippage and liquidity constraints that my evaluation ignores entirely.

**Q: How would you improve this?**
In priority order:
1. **Predict changes, not levels** — I showed this helps dramatically.
2. **Add features** — volume, OHLC, technical indicators, other indices.
3. **Walk-forward validation** instead of one fixed split.
4. **Tune hyperparameters** per architecture (dropout, units, learning rate).
5. **Reframe the task** — predicting direction (classification) or volatility
   is more tractable than predicting exact price.
6. Honestly: accept that daily closes alone contain very little predictive
   information, which is what the efficient market hypothesis predicts.

---

## 4. Points for the report's discussion section

1. **The baseline is the methodological centrepiece.** Without it the project
   would have concluded the opposite.
2. **A controlled comparison needs matched capacity.** 1.08× parameter spread
   is what licenses the claim that differences are architectural.
3. **Extrapolation is a distinct failure mode** from overfitting or
   underfitting, and bounded activations (`sigmoid`, `tanh`) are specifically
   vulnerable to it.
4. **Statistical significance matters.** 53% vs 50% looks like a result and
   isn't one.
5. **Compute cost is a real axis of comparison.** The LSTM cost 4.8× more per
   epoch for the worst accuracy on the main task.
6. **A negative result, properly diagnosed, is a real result.** The bonus
   experiment turns "the models didn't work" into "here is why, here is the
   test of that explanation, and here is what changed."

---

## 5. Honest limitations (say these before the examiner does)

1. Good RMSE ≠ tradeable. Direction accuracy is at chance.
2. One feature only (closing price).
3. One test window (Jul 2025 – Nov 2026); a different regime would differ.
4. Extrapolation confounds the level-based architecture ranking — so that
   ranking is not a general claim about these architectures.
5. Minimal hyperparameter tuning; a tuned LSTM might do better.
6. Five seeds is a small sample for estimating standard deviations.
7. Markets adapt; patterns found in 2016–2023 may not exist in 2026.

---

## 6. Files to include in your submission

Filenames are prefixed by ticker: `_NSEI_*` for NIFTY 50, `TCS_NS_*` for TCS.

| File | Why |
|---|---|
| `stock_prediction_aat.ipynb` | the whole project |
| `requirements.txt` | exact versions, for reproducibility |
| `results/_NSEI_comparison_table.csv` | the main results table |
| `results/_NSEI_model_parameters.csv` | the fairness evidence (1.08× spread) |
| `results/cross_ticker_comparison.csv` | NIFTY vs TCS — the ranking flip |
| `results/_NSEI_bonus_comparison.csv` | levels vs changes |
| `results/directional_significance.csv` | the coin-flip z-test |
| `plots/03__NSEI_split.png` | shows the extrapolation problem visually |
| `plots/05__NSEI_loss_curves_all_models.png` | LSTM overfitting visible |
| `plots/09__NSEI_zoom_last_60_days.png` | **strongest figure** — baseline is the lagged truth |
| `plots/10__NSEI_metric_comparison_bars.png` | the headline comparison |
| `plots/11__NSEI_directional_accuracy.png` | all models inside the chance band |
| `plots/12__NSEI_bonus_levels_vs_changes.png` | the diagnosis, confirmed |
| `plots/10_TCS_NS_metric_comparison_bars.png` | the cross-ticker ranking flip |

Exclude `.venv/` from your submission — it is 1.8 GB of installed packages and
`requirements.txt` already records exactly what it contains.
