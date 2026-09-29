"""Main experiment for the UEE preprint. Run: python run_experiments.py  (writes results/)."""
import json, os
import numpy as np
from scipy import stats
import uee_sim as U

N_OPS, T, SEED = 300, 120, 2026
FLOOR, DEFAULT = U.FLOOR, U.DEFAULT
os.makedirs('results', exist_ok=True)


ORACLE = {w: U.best_setting(w) for w in range(U.NW)}


def oracle_setting(w):
    return ORACLE[w]


def session_rng(i):
    # same operator (parameters and noise stream) in every condition -> paired comparison
    return np.random.default_rng([SEED, i, 0])


def matched(w, s):
    return U.SETTINGS[s][1] == U.SETTINGS[ORACLE[w]][1]      # the mapping that suits this operator type


conds = ['MT', 'Yoked', 'UEE', 'Oracle', 'NaivePE', 'UEE_nodwell', 'UEE_utility', 'TS', 'UCB']
res = {c: dict(E=[], E_all=[], succ=[], switches=[], lowgain=[], matched=[], t_matched=[]) for c in conds}
traj = {c: [] for c in conds}
uee_schedules, ws, pw, wmap = [], [], [], []


def record(c, h, w):
    s = h['s']
    traj[c].append(h['e'])
    res[c]['E'].append(float(h['e'][60:].mean())); res[c]['succ'].append(float(h['succ'].mean()))
    res[c]['switches'].append(int((np.diff(s) != 0).sum()))
    res[c]['lowgain'].append(float(np.mean([U.SETTINGS[x][0] == 0 for x in s])))
    res[c]['matched'].append(float(np.mean([matched(w, x) for x in s[60:]])))
    res[c]['E_all'].append(float(h['e'].mean()))
    ok = [matched(w, x) for x in s]                   # first second from which the mapping stays matched
    res[c]['t_matched'].append(float(next((t for t in range(len(ok)) if all(ok[t:])), len(ok))))


for i in range(N_OPS):
    w = int(np.random.default_rng([SEED, i]).integers(U.NW)); ws.append(w)
    h = U.run_uee(U.build_agent(allowed_settings=FLOOR), U.Operator(session_rng(i), w=w), T=T, dwell=5)
    uee_schedules.append(h['s']); pw.append(h['p_w_true']); wmap.append(h['w_map'] == w); record('UEE', h, w)
    h = U.run_uee(U.build_agent(allowed_settings=FLOOR), U.Operator(session_rng(i), w=w), T=T, dwell=1)
    record('UEE_nodwell', h, w)
    # expected-utility controller: same inference and preferences, no information-gain term
    h = U.run_uee(U.build_agent(allowed_settings=FLOOR, info_gain=False), U.Operator(session_rng(i), w=w), T=T, dwell=5)
    record('UEE_utility', h, w)
    # model-free bandits over the admissible settings (reward: task success + high-embodiment signatures)
    for c in ('TS', 'UCB'):
        h = U.run_bandit(U.Operator(session_rng(i), w=w), c.lower(), np.random.default_rng([SEED, i, 7]), T=T, dwell=5)
        record(c, h, w)
    # naive prediction-error minimiser: prefers small pupils only, no task or embodiment preference, no floor
    h = U.run_uee(U.build_agent(C_task=0.0, C_embodiment=0.0, C_pupil=2.0, allowed_settings=None), U.Operator(session_rng(i), w=w), T=T, dwell=5)
    record('NaivePE', h, w)
    for c, sched in [('MT', [DEFAULT] * T), ('Oracle', [oracle_setting(w)] * T)]:
        record(c, U.run_schedule(U.Operator(session_rng(i), w=w), sched), w)

# Yoked: each operator receives the UEE schedule recorded for the previous operator
for i in range(N_OPS):
    record('Yoked', U.run_schedule(U.Operator(session_rng(i), w=ws[i]), uee_schedules[i - 1]), ws[i])

ws = np.array(ws)
summary = {}
for c in conds:
    summary[c] = {k: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1))) for k, v in res[c].items()}
    summary[c]['E_by_W'] = [float(np.mean(np.array(res[c]['E'])[ws == k])) for k in range(U.NW)]


def paired(a, b, k):
    x, y = np.array(res[a][k]), np.array(res[b][k]); d = x - y
    return dict(diff=float(d.mean()), ci95=[float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))), float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)))],
                dz=float(d.mean() / d.std(ddof=1)), p=float(stats.wilcoxon(x, y).pvalue))


tests = {f'{a}_vs_{b}_{k}': paired(a, b, k) for a, b in [('UEE', 'MT'), ('UEE', 'Yoked'), ('UEE', 'NaivePE'), ('UEE', 'Oracle'),
                                                          ('Yoked', 'MT'), ('UEE_nodwell', 'UEE'), ('Oracle', 'UEE'), ('UEE', 'UEE_utility'), ('UEE', 'TS'), ('UEE', 'UCB')] for k in ['E', 'E_all', 'succ']}
by_w = {}
for k in range(U.NW):
    x, y = np.array(res['UEE']['E'])[ws == k], np.array(res['MT']['E'])[ws == k]; d = x - y
    by_w[k] = dict(n=int((ws == k).sum()), diff=float(d.mean()), ci95=[float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                                                                       float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)))])
tests['UEE_vs_MT_E_by_W'] = by_w
pw, wmap = np.array(pw), np.array(wmap)
summary['belief_true_W'] = dict(t10=float(pw[:, 9].mean()), t30=float(pw[:, 29].mean()), t60=float(pw[:, 59].mean()), t120=float(pw[:, -1].mean()),
                                map_correct_t10=float(wmap[:, 9].mean()), map_correct_t30=float(wmap[:, 29].mean()), map_correct_t60=float(wmap[:, 59].mean()))
json.dump(dict(summary=summary, tests=tests, n_ops=N_OPS, T=T, seed=SEED, weighting_counts=np.bincount(ws).tolist()),
          open('results/summary.json', 'w'), indent=2)
np.save('results/pw.npy', pw)
np.savez('results/traj.npz', **{c: np.array(v) for c, v in traj.items()})
json.dump(dict(w=ws.tolist(), uee_schedules=[x.tolist() for x in uee_schedules], **{c: res[c] for c in conds}), open('results/per_operator.json', 'w'))
print(json.dumps(dict(summary=summary, tests=tests), indent=1))
