"""Robustness of the UEE to mismatch between its population model and the operator.
Each experiment changes one property of the simulated operators; the controller is unchanged.
Outcome: embodiment level in the last 60 s for UEE and for fixed Mechanical Transparency (MT),
paired on the same operators. Writes results/robustness.json."""
import json
import numpy as np
from scipy import stats
import uee_sim as U

N, T, SEED = 200, 120, 7
FLOOR = U.FLOOR
DEFAULT = U.DEFAULT


def compare(label, op_kwargs=None, agent_kwargs=None, w_switch=False, calib=None):
    op_kwargs = op_kwargs or {}; agent_kwargs = agent_kwargs or {}
    eu, em, pw_end, tu, tm, tp = [], [], [], [], [], []
    for i in range(N):
        r = np.random.default_rng([SEED, i])
        w = int(r.integers(U.NW))
        kw = dict(op_kwargs)
        if w_switch:
            kw['w_switch'] = (60, U.NW - 1 - w if w != U.NW // 2 else int(r.choice([0, U.NW - 1])))   # mirror type half-way through (balanced -> a random extreme)
        op = U.Operator(np.random.default_rng([SEED, i, 1]), w=w, **kw)
        ak = dict(agent_kwargs)
        if calib is not None:                                  # session-start cue-conflict calibration: prior mass `calib` on the true weighting
            ak['w_prior'] = [calib if k == w else (1 - calib) / (U.NW - 1) for k in range(U.NW)]
        h = U.run_uee(U.build_agent(allowed_settings=FLOOR, **ak), op, T=T, dwell=5)
        eu.append(h['e'][60:].mean()); pw_end.append(h['p_w_true'][-1]); tu.append(h['e']); tp.append(h['p_w_true'])
        op = U.Operator(np.random.default_rng([SEED, i, 1]), w=w, **kw)
        hm = U.run_schedule(op, [DEFAULT] * T); em.append(hm['e'][60:].mean()); tm.append(hm['e'])
    eu, em = np.array(eu), np.array(em)
    d = eu - em
    out = dict(label=label, UEE=float(eu.mean()), MT=float(em.mean()), diff=float(d.mean()),
               ci95=[float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(N)), float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(N))],
               dz=float(d.mean() / d.std(ddof=1)), p=float(stats.wilcoxon(eu, em).pvalue),
               belief_true_w_end=float(np.mean(pw_end)))
    print(f"{label:45s} UEE {out['UEE']:.2f}  MT {out['MT']:.2f}  diff {out['diff']:+.2f} dz {out['dz']:.2f}  P(W) {out['belief_true_w_end']:.2f}")
    TRAJ[label.strip()] = dict(uee=np.mean(tu, 0), mt=np.mean(tm, 0), p_w=np.mean(tp, 0))
    return out


TRAJ = {}
R = []
R.append(compare('matched model (reference)'))
for nz in [0.05, 0.15, 0.30, 0.45]:
    R.append(compare(f'signal noise {nz:.2f} (model assumes 0.10)', dict(noise=nz)))
R.append(compare('gaze lead uninformative, controller unaware', dict(ghlt_w_slope=0.0)))
R.append(compare('gaze lead uninformative, controller calibrated', dict(ghlt_w_slope=0.0), agent_kwargs=dict(ghlt_w_slope=0.0)))
R.append(compare('  + session-start calibration prior 0.6', dict(ghlt_w_slope=0.0), agent_kwargs=dict(ghlt_w_slope=0.0), calib=0.6))
R.append(compare('  + session-start calibration prior 0.8', dict(ghlt_w_slope=0.0), agent_kwargs=dict(ghlt_w_slope=0.0), calib=0.8))
R.append(compare('pupil ignores sensory conflict', dict(pupil_conflict=0.0)))
R.append(compare('pupil overreacts to conflict (x2)', dict(pupil_conflict=0.8)))
R.append(compare('operators adapt 2x slower', dict(rate_scale=0.5)))
R.append(compare('operators adapt 1.6x faster', dict(rate_scale=1.6)))
R.append(compare('operators lapse 2x as often', dict(lapse_scale=2.0)))
R.append(compare('operators lapse half as often', dict(lapse_scale=0.5)))
R.append(compare('weighting flips at 60 s, no drift', w_switch=True))
for dr in [0.005, 0.01, 0.02]:
    R.append(compare(f'weighting flips at 60 s, drift {dr:.3f}', agent_kwargs=dict(w_drift=dr), w_switch=True))
    R.append(compare(f'matched model, drift {dr:.3f} (cost check)', agent_kwargs=dict(w_drift=dr)))
json.dump(R, open('results/robustness.json', 'w'), indent=2)
np.savez('results/robustness_traj.npz', **{f'{k}|{m}': v[m] for k, v in TRAJ.items() for m in v})
