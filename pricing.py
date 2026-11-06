"""Marketplace pricing experiments: design, estimation and optimisation on a simulated two-sided market.

The simulator has a known ground truth (true demand elasticity, true supply response, true carryover),
so every method below can be checked against the answer instead of trusted.

1. Switchback design: why a washout period matters when prices carry over between blocks
2. Demand elasticity: log-log regression with zone and hour fixed effects, cluster-robust SEs by block
3. Supply response: acceptance-rate curve vs driver payout and the demand/supply ratio
4. CUPED: variance reduction using each zone's pre-period metric
5. Power: how many switchback blocks are needed to detect a given effect
6. Optimiser: price per zone x hour that maximises contribution subject to a fulfilment floor
"""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

TRUE = dict(elasticity_offpeak=-1.6, elasticity_peak=-0.7, supply_beta=2.2, carryover=0.35, take_rate=0.18)
ARMS = np.array([0.85, 0.95, 1.05, 1.15, 1.25])


def simulate(seed=0, zones=40, days=28, block_h=2, washout=False):
    """Block-level marketplace. Supply in a block reacts partly to the PREVIOUS block's price (carryover):
    drivers reposition toward zones that paid well recently. That is the interference switchbacks must handle."""
    r = np.random.default_rng(seed)
    z_size = r.lognormal(4.2, .4, zones)              # zone demand scale
    z_supply = r.uniform(.7, 1.3, zones)              # zone supply richness
    hours = np.arange(0, 24, block_h)
    rows = []
    for z in range(zones):
        prev = 1.0
        for d in range(days):
            for h in hours:
                peak = h in (8, 10, 18, 20)
                arm = r.choice(ARMS)
                el = TRUE["elasticity_peak"] if peak else TRUE["elasticity_offpeak"]
                base_q = z_size[z] * (1.6 if peak else 1.0) * (0.85 if d % 7 in (5, 6) else 1)
                quotes = r.poisson(base_q)
                conv = np.clip(.45 * arm ** el, .01, .99)
                bookings = r.binomial(quotes, conv)
                eff_price = arm if washout else (1 - TRUE["carryover"]) * arm + TRUE["carryover"] * prev
                ds_ratio = (bookings + 1) / (z_supply[z] * z_size[z] * 1.15 * .45)  # supply does not scale up at peak
                logit = 1.0 + TRUE["supply_beta"] * (eff_price - 1) - 1.1 * (ds_ratio - 1) + r.normal(0, .15)
                acc = 1 / (1 + np.exp(-logit))
                fulfilled = r.binomial(bookings, acc)
                rows.append((z, d, h, peak, arm, prev, quotes, bookings, fulfilled, ds_ratio))
                prev = arm
    df = pd.DataFrame(rows, columns=["zone", "day", "hour", "peak", "price", "prev_price", "quotes", "bookings", "fulfilled", "ds_ratio"])
    df["block_id"] = df.zone.astype(str) + "_" + df.day.astype(str) + "_" + df.hour.astype(str)
    df["fulfil_rate"] = df.fulfilled / df.bookings.clip(lower=1)
    df["contrib"] = df.fulfilled * 400 * df.price * TRUE["take_rate"]
    return df


# ---------- helpers ----------
def ols_cluster(y, X, groups):
    """OLS with cluster-robust (CR1) standard errors. X must include the intercept / fixed effects."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    g_codes = pd.factorize(groups)[0]
    for g in np.unique(g_codes):
        s = X[g_codes == g].T @ resid[g_codes == g]
        meat += np.outer(s, s)
    G, n, k = g_codes.max() + 1, len(y), X.shape[1]
    adj = G / (G - 1) * (n - 1) / (n - k)
    V = adj * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V))


def design(df, cols):
    """Numeric columns + zone and hour fixed effects."""
    X = [np.ones(len(df))] + [df[c].values.astype(float) for c in cols]
    X += [(df.zone == z).values.astype(float) for z in sorted(df.zone.unique())[1:]]
    X += [(df.hour == h).values.astype(float) for h in sorted(df.hour.unique())[1:]]
    return np.column_stack(X)


# ---------- 1 + 3. Supply response, with and without washout ----------
def supply_effect(df):
    """Effect of +10% price on fulfilment rate, estimated naively from the current block's price only."""
    d = df[df.bookings > 0]
    X = design(d, ["price", "ds_ratio"])
    b, se = ols_cluster(d.fulfil_rate.values, X, d.zone.values)
    return b[1] * .10, se[1] * .10


def true_supply_effect(seed=0):
    """Ground truth: rerun with no carryover, so the current price is the whole story."""
    return supply_effect(simulate(seed=seed, washout=True))[0]


# ---------- 2. Demand elasticity ----------
def elasticity(df):
    d = df[(df.bookings > 0) & (df.quotes > 0)].copy()
    d["log_conv"] = np.log(d.bookings / d.quotes)
    d["log_p"] = np.log(d.price)
    d["log_p_peak"] = d.log_p * d.peak
    X = design(d, ["log_p", "log_p_peak"])
    b, se = ols_cluster(d.log_conv.values, X, d.zone.values)
    return dict(offpeak=(b[1], se[1]), peak=(b[1] + b[2], np.sqrt(se[1] ** 2 + se[2] ** 2)))


# ---------- 4. CUPED ----------
def cuped(df, seed=1):
    """Zone-level test: treated zones get +5% price for the last 14 days. Pre-period = first 14 days."""
    r = np.random.default_rng(seed)
    zones = df.zone.unique()
    treated = set(r.choice(zones, len(zones) // 2, replace=False))
    pre = df[df.day < 14].groupby("zone").contrib.sum()
    post = df[df.day >= 14].groupby("zone").contrib.sum()
    t = np.array([z in treated for z in post.index])
    post = post * r.lognormal(0, .12, len(post))         # zone-level drift the pre-period cannot explain
    post_adj = post * np.where(t, 1.03, 1.0)              # inject a true +3% lift
    theta = np.cov(post_adj, pre)[0, 1] / pre.var()
    cu = post_adj - theta * (pre - pre.mean())
    def diff_se(y):
        a, b = y[t], y[~t]
        return a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    raw, cup = diff_se(post_adj.values), diff_se(cu.values)
    return dict(raw_se=raw[1], cuped_se=cup[1], variance_reduction=1 - (cup[1] / raw[1]) ** 2,
                true_lift=0.03 * post[list(treated)].mean(), raw_est=raw[0], cuped_est=cup[0])


# ---------- 5. Power ----------
def blocks_needed(df, effect=0.03, alpha_z=1.96, power_z=0.84):
    """Blocks per arm to detect a relative change in fulfilled orders per block (two-arm switchback)."""
    y = df.fulfilled
    cv = y.std() / y.mean()
    return int(np.ceil(2 * (alpha_z + power_z) ** 2 * cv ** 2 / effect ** 2))


# ---------- 6. Optimiser ----------
def optimise(df, el, floor=0.70):
    """Fit the acceptance curve, then for each zone x peak/off-peak pick the price that maximises expected
    contribution per quote while keeping expected fulfilment >= floor."""
    d = df[df.bookings > 0]
    Xf = np.column_stack([d.price, d.ds_ratio])
    # expand to booking-level labels with weights instead of rows
    X2 = np.vstack([Xf, Xf]); y2 = np.r_[np.ones(len(d)), np.zeros(len(d))]
    w2 = np.r_[d.fulfilled.values, (d.bookings - d.fulfilled).values]
    acc = LogisticRegression(C=1e3, max_iter=500).fit(X2, y2, sample_weight=w2 + 1e-9)
    grid = np.round(np.arange(.85, 1.26, .05), 2)   # stay inside the tested price range: no extrapolation
    out = []
    for (z, peak), g in d.groupby(["zone", "peak"]):
        e = el["peak"][0] if peak else el["offpeak"][0]
        ds = g.ds_ratio.median()
        best, best_v, cur_v = None, -1, None
        for p in grid:
            conv = .45 * p ** e
            f = acc.predict_proba([[p, ds]])[0, 1]
            v = conv * f * 400 * p * TRUE["take_rate"]
            if abs(p - 1) < 1e-9:
                cur_v, cur_f = v, f
            if f >= floor and v > best_v:
                best, best_v, best_f = p, v, f
        if best is None:  # floor unreachable in grid: take the highest price
            best, best_v, best_f = grid[-1], v, f
        out.append(dict(zone=z, peak=peak, best_price=best, contrib_per_quote=best_v, fulfil=best_f,
                        contrib_at_1=cur_v, fulfil_at_1=cur_f))
    return pd.DataFrame(out)


def main():
    df = simulate()
    df_w = simulate(washout=True)
    naive, naive_se = supply_effect(df)
    truth = true_supply_effect()
    el = elasticity(df)
    cu = cuped(df)
    n_blocks = blocks_needed(df)
    opt = optimise(df, el)
    res = dict(
        blocks=len(df),
        supply_effect_of_10pct_price_pts=dict(no_washout=round(100 * naive, 2), no_washout_se=round(100 * naive_se, 2),
                                              truth=round(100 * truth, 2),
                                              bias_pct=round(100 * (naive / truth - 1), 1)),
        elasticity=dict(offpeak_est=round(el["offpeak"][0], 3), offpeak_se=round(el["offpeak"][1], 3),
                        offpeak_true=TRUE["elasticity_offpeak"],
                        peak_est=round(el["peak"][0], 3), peak_se=round(el["peak"][1], 3),
                        peak_true=TRUE["elasticity_peak"]),
        cuped=dict(variance_reduction_pct=round(100 * cu["variance_reduction"], 1),
                   raw_se=round(cu["raw_se"]), cuped_se=round(cu["cuped_se"]),
                   true_lift=round(cu["true_lift"]), raw_est=round(cu["raw_est"]), cuped_est=round(cu["cuped_est"])),
        blocks_per_arm_for_3pct=n_blocks,
        optimiser=dict(cells=len(opt),
                       peak_median_price=float(opt[opt.peak].best_price.median()),
                       offpeak_median_price=float(opt[~opt.peak].best_price.median()),
                       contrib_uplift_pct=round(100 * (opt.contrib_per_quote.sum() / opt.contrib_at_1.sum() - 1), 1),
                       cells_meeting_70pct_floor=int((opt.fulfil >= .70).sum()),
                       cells_meeting_floor_at_current_price=int((opt.fulfil_at_1 >= .70).sum())),
    )
    opt.round(3).to_csv("optimal_prices.csv", index=False)
    with open("results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
