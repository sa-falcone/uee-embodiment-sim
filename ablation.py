"""Ablation: separate the ingredients that protect against the degenerate objective (feedback floor, task preference)
and test whether the embodiment preference is needed. Same 300 operators and seeds as run_experiments.py.
Writes results/ablation.json."""
import json
import numpy as np
from scipy import stats
import uee_sim as U

N_OPS, T, SEED = 300, 120, 2026
FLOOR = U.FLOOR
NAIVE = dict(C_task=0.0, C_embodiment=0.0, C_pupil=2.0)
conds = {
    'UEE (preferences + floor)':        dict(allowed_settings=FLOOR),
    'UEE preferences, no floor':        dict(allowed_settings=None),
    'UEE without information gain':     dict(info_gain=False, allowed_settings=FLOOR),
    'UEE without embodiment preference': dict(C_embodiment=0.0, allowed_settings=FLOOR),
    'UEE without task preference':      dict(C_task=0.0, allowed_settings=FLOOR),
    'No preferences (epistemic only), floor': dict(C_task=0.0, C_embodiment=0.0, allowed_settings=FLOOR),
    'No preferences, no floor':         dict(C_task=0.0, C_embodiment=0.0, allowed_settings=None),
    'Naive objective + floor':          dict(**NAIVE, allowed_settings=FLOOR),
    'Naive objective, no floor':        dict(**NAIVE, allowed_settings=None),
}
out, per = {}, {}
for name, kw in conds.items():
    E, succ, low, mid, matched = [], [], [], [], []
    for i in range(N_OPS):
        w = int(np.random.default_rng([SEED, i]).integers(U.NW))
        h = U.run_uee(U.build_agent(**kw), U.Operator(np.random.default_rng([SEED, i, 0]), w=w), T=T, dwell=5)
        g = np.array([U.SETTINGS[s][0] for s in h['s']])
        E.append(h['e'][60:].mean()); succ.append(h['succ'].mean()); low.append(np.mean(g == 0)); mid.append(np.mean(g[60:] == 1))
        best_m = U.SETTINGS[U.best_setting(w)][1]
        matched.append(np.mean([U.SETTINGS[s][1] == best_m for s in h['s'][60:]]))
    per[name] = dict(E=E, succ=succ)
    out[name] = dict(E=float(np.mean(E)), E_sd=float(np.std(E, ddof=1)), succ=float(np.mean(succ)), succ_sd=float(np.std(succ, ddof=1)),
                     lowgain=float(np.mean(low)), midgain_last60=float(np.mean(mid)), matched_last60=float(np.mean(matched)))
    print(f"{name:36s} E {out[name]['E']:.3f}  succ {out[name]['succ']:.3f}  lowest gain {out[name]['lowgain']:.0%}  "
          f"mid gain (last 60 s) {out[name]['midgain_last60']:.0%}  matched mapping {out[name]['matched_last60']:.0%}")
ref = per['UEE (preferences + floor)']
for name in list(conds)[1:]:
    for k in ['E', 'succ']:
        x, y = np.array(per[name][k]), np.array(ref[k]); d = x - y
        out[name][f'{k}_minus_full'] = dict(diff=float(d.mean()), dz=float(d.mean() / d.std(ddof=1)) if d.std() > 0 else 0.0,
                                            p=float(stats.wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0)
json.dump(out, open('results/ablation.json', 'w'), indent=2)
