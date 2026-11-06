# Marketplace Pricing Experiments

[![tests](https://github.com/f20220970-debug/marketplace-pricing-experiments/actions/workflows/tests.yml/badge.svg)](https://github.com/f20220970-debug/marketplace-pricing-experiments/actions/workflows/tests.yml)

How to run, read and act on price tests in a two-sided marketplace, where a price change moves **both** demand (customers book less) **and** supply (driver-partners accept more). It is built on a simulator with a **known ground truth**, so every estimate can be checked against the real answer.

> **Simulated data only:** 40 zones × 28 days × 2-hour blocks = 13,440 switchback blocks. No company data is used.

Run it: `pip install -r requirements.txt && python pricing.py && pytest -q` (about 2 seconds).

## What it shows

| # | Question | Method | Result vs truth |
| --- | --- | --- | --- |
| 1 | Does a plain switchback get the supply effect right? | Switchback with and without a washout block; drivers reposition toward zones that paid well in the last block (35% carryover) | Without a washout, the effect of +10% price on fulfilment is **2.5 pts vs a true 3.9 pts (37% too low)**. Carryover makes control blocks look better than they are. Fix: drop the first block after each switch, or use longer blocks. |
| 2 | How price-sensitive is demand? | Log-log regression of conversion on price, with zone and hour fixed effects and **cluster-robust SEs by zone** | Off-peak **−1.63 (SE 0.013) vs true −1.6**; peak **−0.71 (SE 0.025) vs true −0.7** |
| 3 | Can the same test read faster? | **CUPED**, using each zone's pre-period contribution as the covariate | Variance cut **87%**. A true +3% lift (+7,966 per zone) read as **−51,712 ± 27,845 raw** but **+3,891 ± 10,023 with CUPED** |
| 4 | How long must a test run? | Power calculation from block-level variation | **≈3,000 blocks per arm** to detect a 3% change in fulfilled orders at 80% power |
| 5 | What price should each zone charge? | Fitted acceptance curve + elasticities; maximise contribution per quote **subject to fulfilment ≥ 70%**, inside the tested range only | **Peak: raise to 1.25×** (fulfilment 65% → 73%). **Off-peak: cut to 0.85–0.95×** in 39 of 40 zones (demand is elastic and supply is idle). **Contribution +11.3%**; zone-hours meeting the 70% floor go from 51 to 66 of 80 |

## Why it matters for a category team

- **Never read a price test on bookings alone.** In this simulation, a peak-hour price rise loses bookings but wins on fulfilled orders and contribution.
- **Switchbacks need a washout.** Without one, the test understates how much supply responds to price, which pushes you to under-price peaks.
- **Fixed effects and clustered SEs keep the estimates honest.** Zones differ in size, and blocks within a zone are correlated. Ignoring either gives false confidence.
- **Stay inside the tested range.** The optimiser lands on 1.25× at peak, the top arm tested, so the next experiment should test 1.3–1.4× rather than assume the curve keeps going.

## Files

```
pricing.py          simulator + six analyses (ground truth is in TRUE at the top)
results.json        headline numbers
optimal_prices.csv  recommended price per zone x peak/off-peak, with expected fulfilment
tests/              4 checks: estimates recover the truth, carryover bias exists, CUPED helps, no extrapolation
```

Companion repo: [porter-category-growth-lab](https://github.com/f20220970-debug/porter-category-growth-lab), which applies these methods to city × vehicle-category pricing.
