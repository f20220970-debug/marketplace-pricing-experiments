import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pricing as P

df = P.simulate()


def test_elasticity_recovers_truth():
    el = P.elasticity(df)
    for k, true in (("offpeak", P.TRUE["elasticity_offpeak"]), ("peak", P.TRUE["elasticity_peak"])):
        est, se = el[k]
        assert abs(est - true) < 3 * se + 0.03


def test_carryover_biases_naive_switchback():
    naive, _ = P.supply_effect(df)
    truth = P.true_supply_effect()
    assert naive < truth * 0.8          # carryover attenuates the naive estimate


def test_cuped_reduces_variance():
    assert P.cuped(df)["variance_reduction"] > 0.5


def test_optimiser_stays_in_tested_range():
    opt = P.optimise(df, P.elasticity(df))
    assert opt.best_price.between(0.85, 1.25).all()
