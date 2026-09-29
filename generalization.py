"""Out-of-family tests and population sweep for the UEE preprint. Writes results/generalization.json.

1. Operators outside the controller's model family (the controller is unchanged):
   continuous cue weighting (uniform on [-1, 1]) instead of five types, and/or a different match function
   (triangular, or a Gaussian 1.6 times wider than the controller assumes).
2. Population sweep: the fraction of balanced operators varies from 0 to 1 (the rest split evenly over the other
   four types), so that the benefit over fixed control is shown as a function of how varied the population is.
In every scenario Mechanical Transparency (MT) is recomputed as the best fixed high-gain setting for that population,
and the Oracle as the best setting for each operator."""
import json, zlib
import numpy as np
from scipy import stats
import uee_sim as U

N, T, SEED = 200, 120, 11
GRID = np.linspace(-1, 1, 41)


def ci(d):
    d = np.asarray(d, float); m = d.mean(); h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return [float(m - h), float(m + h)]


def run_population(label, draw, match_fn=U.match_gauss, mt=None):
    """draw(rng) -> dict(w=...) or dict(w_pos=...) for one operator."""
    eu, em, eo = [], [], []
    for i in range(N):
        r = np.random.default_rng([SEED, zlib.crc32(label.encode()) % 10**6, i])
        kw = draw(r); kw['match_fn'] = match_fn
        oracle = U.best_setting(kw['w']) if 'w' in kw and match_fn is U.match_gauss else \
            U.best_setting_for(kw['w_pos'] if 'w_pos' in kw else U.W_POS[kw['w']], match_fn)
        seed = [SEED, zlib.crc32(label.encode()) % 10**6, i, 1]
        h = U.run_uee(U.build_agent(allowed_settings=U.FLOOR), U.Operator(np.random.default_rng(seed), **kw), T=T, dwell=5)
        eu.append(h['e'][60:].mean())
        em.append(U.run_schedule(U.Operator(np.random.default_rng(seed), **kw), [mt] * T)['e'][60:].mean())
        eo.append(U.run_schedule(U.Operator(np.random.default_rng(seed), **kw), [oracle] * T)['e'][60:].mean())
    eu, em, eo = map(np.array, (eu, em, eo))
    out = dict(label=label, MT_setting=U.SETTINGS[mt], UEE=float(eu.mean()), MT=float(em.mean()), Oracle=float(eo.mean()),
               UEE_minus_MT=float((eu - em).mean()), UEE_minus_MT_ci=ci(eu - em), dz_MT=float((eu - em).mean() / (eu - em).std(ddof=1)),
               Oracle_minus_MT=float((eo - em).mean()), Oracle_minus_MT_ci=ci(eo - em),
               UEE_minus_Oracle=float((eu - eo).mean()), UEE_minus_Oracle_ci=ci(eu - eo),
               share_of_oracle_gain=float((eu - em).mean() / (eo - em).mean()) if (eo - em).mean() > 1e-9 else None)
    print(f"{label:48s} MT={U.SETTINGS[mt]} UEE {out['UEE']:.2f} MT {out['MT']:.2f} Oracle {out['Oracle']:.2f} "
          f"UEE-MT {out['UEE_minus_MT']:+.2f} UEE-Oracle {out['UEE_minus_Oracle']:+.2f}", flush=True)
    return out


types = lambda r: dict(w=int(r.integers(U.NW)))
cont = lambda r: dict(w_pos=float(r.uniform(-1, 1)))
MT5 = lambda fn: U.best_fixed_setting_for(U.W_POS, fn)
MTc = lambda fn: U.best_fixed_setting_for(GRID, fn)
out = dict(out_of_family=[], sweep=[])
for label, draw, fn, mt in [
        ('five types, Gaussian match (in family)', types, U.match_gauss, MT5(U.match_gauss)),
        ('continuous cue weighting', cont, U.match_gauss, MTc(U.match_gauss)),
        ('five types, triangular match', types, U.match_triangular, MT5(U.match_triangular)),
        ('five types, match 1.6x wider', types, U.match_wide, MT5(U.match_wide)),
        ('continuous cue weighting, triangular match', cont, U.match_triangular, MTc(U.match_triangular))]:
    out['out_of_family'].append(run_population(label, draw, fn, mt))

for p_bal in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    probs = np.array([(1 - p_bal) / 4] * 2 + [p_bal] + [(1 - p_bal) / 4] * 2)
    draw = lambda r, probs=probs: dict(w=int(r.choice(U.NW, p=probs)))
    res = run_population(f'fraction balanced {p_bal:.1f}', draw, U.match_gauss, U.best_fixed_setting(probs))
    res['p_balanced'] = p_bal
    out['sweep'].append(res)
json.dump(out, open('results/generalization.json', 'w'), indent=2)
