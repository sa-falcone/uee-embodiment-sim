"""Revision analyses for the THMS submission. Run: python analyses_v2.py  (writes results/analyses_v2.json).

A. Does the controller need embodiment signals?  A world in which task success carries no information about
   embodiment, with full or absent embodiment signals, and controllers with both preferences, task only, or
   embodiment only.
B. Rule-based baselines of the kind used in adaptive automation, with the same per-second signals:
   (i) a threshold rule that maps gaze-hand lead time directly to a mapping; (ii) a trigger rule that changes the
   mapping whenever the embodiment signatures fall below a threshold. Both tuned on the main operators.
C. Variation between random seeds: the main comparisons (10 seeds x 300 operators) and key robustness scenarios
   (10 seeds x 200 operators).
D. Sensitivity to the controller's design choices (preference strengths, dwell, prior, floor, assumed noise).
"""
import json, itertools
from multiprocessing import Pool
import numpy as np
import uee_sim as U

T = 120
FLOOR, DEFAULT = U.FLOOR, U.DEFAULT
ORACLE = {w: U.best_setting(w) for w in range(U.NW)}
HIGH = {m: 2 * U.NM + m for m in range(U.NM)}          # high-gain setting for each mapping


def summarize(h, w):
    s = np.asarray(h['s'])
    return dict(E=float(np.mean(h['e'][60:])), E_all=float(np.mean(h['e'])), succ=float(np.mean(h['succ'])),
                switches=int((np.diff(s) != 0).sum()))


# ---------------------------------------------------------------- rule-based controllers
def run_rule_ghlt(op, lo, hi, window, dwell=5):
    """Threshold rule: mean GHLT bin over the last `window` s -> visual (< lo), proprioceptive (> hi), else mixed; high gain."""
    s = DEFAULT; hist = dict(s=[], e=[], succ=[]); g = []
    for t in range(T):
        obs = op.step(s); g.append(obs[2])
        hist['s'].append(s); hist['e'].append(op.e); hist['succ'].append(obs[4] == 0)
        if t % dwell == dwell - 1:
            m = np.mean(g[-window:])
            s = HIGH[0] if m < lo else HIGH[2] if m > hi else HIGH[1]
    return {k: np.array(v) for k, v in hist.items()}


def run_rule_trigger(op, theta, window, dwell=5):
    """Trigger rule: embodiment index = mean of (low GTE, high LAT) over the last `window` s; if it falls below theta,
    move to the next mapping (visual -> mixed -> proprioceptive -> visual); high gain throughout."""
    m = U.SETTINGS[DEFAULT][1]; s = DEFAULT; hist = dict(s=[], e=[], succ=[]); idx = []; since = 0
    for t in range(T):
        obs = op.step(s); idx.append(((obs[1] == 0) + (obs[3] == 2)) / 2.0); since += 1
        hist['s'].append(s); hist['e'].append(op.e); hist['succ'].append(obs[4] == 0)
        if t % dwell == dwell - 1 and since >= window and np.mean(idx[-window:]) < theta:
            m = (m + 1) % U.NM; s = HIGH[m]; since = 0
    return {k: np.array(v) for k, v in hist.items()}


# ---------------------------------------------------------------- one session
def session(job):
    """job = (key, seed_tuple, w, op_kwargs, ctrl, ctrl_kwargs)"""
    key, seed, w, op_kwargs, ctrl, ck = job
    op = U.Operator(np.random.default_rng(seed), w=w, **op_kwargs)
    if ctrl == 'uee':
        ck = dict(ck); dwell = ck.pop('dwell', 5); calib = ck.pop('calib', None)
        if calib is not None:
            ck['w_prior'] = [calib if k == w else (1 - calib) / (U.NW - 1) for k in range(U.NW)]
        ck.setdefault('allowed_settings', FLOOR)
        h = U.run_uee(U.build_agent(**ck), op, T=T, dwell=dwell)
    elif ctrl == 'fixed':
        s = ORACLE[w] if ck['s'] == 'oracle' else DEFAULT if ck['s'] == 'MT' else ck['s']
        h = U.run_schedule(op, [s] * T)
    elif ctrl == 'schedule':
        h = U.run_schedule(op, ck['schedule'])
    elif ctrl in ('ts', 'ucb'):
        h = U.run_bandit(op, ctrl, np.random.default_rng(list(seed) + [7]), T=T, dwell=5)
    elif ctrl == 'rule_ghlt':
        h = run_rule_ghlt(op, **ck)
    elif ctrl == 'rule_trigger':
        h = run_rule_trigger(op, **ck)
    out = summarize(h, w); out['key'] = key; out['w'] = w
    if ctrl == 'uee':
        out['schedule'] = [int(x) for x in h['s']]
    return out


def operators(seed, n):
    return [(i, int(np.random.default_rng([seed, i]).integers(U.NW))) for i in range(n)]


def run_jobs(pool, jobs):
    res = {}
    for r in pool.imap_unordered(session, jobs, chunksize=8):
        res.setdefault(r['key'], []).append(r)
    return res


def paired(res, a, b, k='E'):
    """Mean paired difference a - b over operators (matched by operator index stored in key order)."""
    x = np.array([r[k] for r in sorted(res[a], key=lambda r: r['i'])]); y = np.array([r[k] for r in sorted(res[b], key=lambda r: r['i'])])
    d = x - y; n = len(d)
    return dict(diff=float(d.mean()), ci95=[float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(n)), float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(n))],
                dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float('nan'), a=float(x.mean()), b=float(y.mean()))


def tag(res):
    """Attach the operator index (stored in the key as 'name|i') and regroup by condition name."""
    out = {}
    for key, rows in res.items():
        name, i = key.rsplit('|', 1)
        for r in rows:
            r['i'] = int(i); out.setdefault(name, []).append(r)
    return out


def scenario_jobs(name, seed, n, op_kwargs, ctrl, ck, op_seed_extra=1):
    return [(f'{name}|{i}', (seed, i, op_seed_extra), w, op_kwargs, ctrl, ck) for i, w in operators(seed, n)]


if __name__ == '__main__':
    OUT = {}
    with Pool() as pool:
        # ======================================================== B0. tune the rules on the main operators (seed 2026)
        MAIN, N_MAIN = 2026, 300
        grid_g = [dict(lo=lo, hi=hi, window=wd) for lo in [0.6, 0.7, 0.8, 0.9] for hi in [1.1, 1.2, 1.3, 1.4] for wd in [5, 10, 20, 60]]
        grid_t = [dict(theta=th, window=wd) for th in [0.2, 0.3, 0.4, 0.5, 0.6] for wd in [5, 10, 20]]
        jobs = []
        for k, ck in enumerate(grid_g):
            jobs += [(f'g{k}|{i}', (MAIN, i, 0), w, {}, 'rule_ghlt', ck) for i, w in operators(MAIN, N_MAIN)]
        for k, ck in enumerate(grid_t):
            jobs += [(f't{k}|{i}', (MAIN, i, 0), w, {}, 'rule_trigger', ck) for i, w in operators(MAIN, N_MAIN)]
        res = tag(run_jobs(pool, jobs))
        best_g = grid_g[max(range(len(grid_g)), key=lambda k: np.mean([r['E'] for r in res[f'g{k}']]))]
        best_t = grid_t[max(range(len(grid_t)), key=lambda k: np.mean([r['E'] for r in res[f't{k}']]))]
        OUT['rule_tuning'] = dict(best_ghlt=best_g, best_trigger=best_t,
                                  ghlt_grid=[dict(ck, E=float(np.mean([r['E'] for r in res[f'g{k}']]))) for k, ck in enumerate(grid_g)],
                                  trigger_grid=[dict(ck, E=float(np.mean([r['E'] for r in res[f't{k}']]))) for k, ck in enumerate(grid_t)])
        print('best GHLT rule', best_g, ' best trigger rule', best_t, flush=True)

        # ======================================================== C1. main comparisons over 10 seeds (seed 2026 = paper)
        seeds = [2026] + list(range(3001, 3010))
        main_conds = [('UEE', 'uee', {}), ('EU', 'uee', dict(info_gain=False)), ('NaivePE', 'uee', dict(C_task=0.0, C_embodiment=0.0, C_pupil=2.0, allowed_settings=None)),
                      ('MT', 'fixed', dict(s='MT')), ('Oracle', 'fixed', dict(s='oracle')), ('TS', 'ts', {}), ('UCB', 'ucb', {}),
                      ('RuleGHLT', 'rule_ghlt', best_g), ('RuleTrigger', 'rule_trigger', best_t)]
        per_seed = []
        for sd in seeds:
            jobs = []
            for name, ctrl, ck in main_conds:
                jobs += [(f'{name}|{i}', (sd, i, 0), w, {}, ctrl, ck) for i, w in operators(sd, N_MAIN)]
            res = tag(run_jobs(pool, jobs))
            # yoked: previous operator's UEE schedule
            sched = {r['i']: r['schedule'] for r in res['UEE']}
            jobs = [(f'Yoked|{i}', (sd, i, 0), w, {}, 'schedule', dict(schedule=sched[(i - 1) % N_MAIN])) for i, w in operators(sd, N_MAIN)]
            res.update(tag(run_jobs(pool, jobs)))
            row = dict(seed=sd, mean={c: dict(E=float(np.mean([r['E'] for r in res[c]])), succ=float(np.mean([r['succ'] for r in res[c]])),
                                             switches=float(np.mean([r['switches'] for r in res[c]]))) for c in res},
                       diffs={f'UEE-{b}': paired(res, 'UEE', b) for b in ['MT', 'Yoked', 'Oracle', 'EU', 'TS', 'UCB', 'NaivePE', 'RuleGHLT', 'RuleTrigger']})
            row['diffs']['UEE-NaivePE_succ'] = paired(res, 'UEE', 'NaivePE', 'succ')
            per_seed.append(row)
            print(f"seed {sd}: UEE-MT {row['diffs']['UEE-MT']['diff']:+.3f}  UEE-Oracle {row['diffs']['UEE-Oracle']['diff']:+.3f}  "
                  f"UEE-RuleGHLT {row['diffs']['UEE-RuleGHLT']['diff']:+.3f}  UEE-RuleTrigger {row['diffs']['UEE-RuleTrigger']['diff']:+.3f}", flush=True)
        OUT['main_seeds'] = per_seed

        # ======================================================== A. embodiment signals when task success says nothing about embodiment
        SEED_A, N = 13, 200
        A = []
        for task_e in [1.0, 0.0]:
            for ss in [1.0, 0.0]:
                ok = dict(task_e_effect=task_e, signal_strength=ss)
                ak = dict(task_e_effect=task_e, signal_strength=ss)
                conds = [('both', 'uee', dict(ak)), ('task_only', 'uee', dict(ak, C_embodiment=0.0)), ('embodiment_only', 'uee', dict(ak, C_task=0.0)),
                         ('MT', 'fixed', dict(s='MT')), ('Oracle', 'fixed', dict(s='oracle'))]
                jobs = []
                for name, ctrl, ck in conds:
                    jobs += scenario_jobs(name, SEED_A, N, ok, ctrl, ck)
                res = tag(run_jobs(pool, jobs))
                row = dict(task_e_effect=task_e, signal_strength=ss,
                           **{c: dict(E=float(np.mean([r['E'] for r in res[c]])), succ=float(np.mean([r['succ'] for r in res[c]]))) for c in res},
                           diffs={f'{c}-MT': paired(res, c, 'MT') for c in ['both', 'task_only', 'embodiment_only']})
                A.append(row)
                print(f"[A] task_e {task_e} signals {ss}: " + '  '.join(f"{c} {row['diffs'][c + '-MT']['diff']:+.2f}" for c in ['both', 'task_only', 'embodiment_only']), flush=True)
        OUT['embodiment_signals'] = A

        # ======================================================== C2. key robustness scenarios over 10 seeds, with the rules
        g0 = dict(ghlt_w_slope=0.0)
        scen = [('matched', {}, {}, None), ('noise 0.45', dict(noise=0.45), {}, None),
                ('GHLT uninformative, unaware', g0, {}, None), ('GHLT uninformative, calibrated', g0, g0, None),
                ('GHLT uninformative, calibrated, prior 0.8', g0, g0, 0.8),
                ('signals 0', dict(signal_strength=0.0), {}, None), ('additive cues', dict(cue_combination='additive'), {}, None)]
        rseeds = [7] + list(range(4001, 4010))
        R = []
        for name, ok, ak, cal in scen:
            vals = {k: [] for k in ['UEE-MT', 'RuleGHLT-MT', 'RuleTrigger-MT', 'Oracle-MT']}
            for sd in rseeds:
                ck = dict(ak); ck['calib'] = cal
                jobs = (scenario_jobs('UEE', sd, N, ok, 'uee', ck) + scenario_jobs('MT', sd, N, ok, 'fixed', dict(s='MT')) +
                        scenario_jobs('Oracle', sd, N, ok, 'fixed', dict(s='oracle')) + scenario_jobs('RuleGHLT', sd, N, ok, 'rule_ghlt', best_g) +
                        scenario_jobs('RuleTrigger', sd, N, ok, 'rule_trigger', best_t))
                res = tag(run_jobs(pool, jobs))
                for c in ['UEE', 'RuleGHLT', 'RuleTrigger', 'Oracle']:
                    vals[f'{c}-MT'].append(paired(res, c, 'MT')['diff'])
            row = dict(scenario=name, **{k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), min=float(np.min(v)), max=float(np.max(v))) for k, v in vals.items()})
            R.append(row)
            print(f"[C2] {name:42s} UEE-MT {row['UEE-MT']['mean']:+.2f} (SD {row['UEE-MT']['sd']:.2f})  RuleGHLT {row['RuleGHLT-MT']['mean']:+.2f}  "
                  f"RuleTrigger {row['RuleTrigger-MT']['mean']:+.2f}  Oracle {row['Oracle-MT']['mean']:+.2f}", flush=True)
        OUT['robustness_seeds'] = R

        # ======================================================== D. sensitivity to design choices
        S = []
        variants = [('reference', {}), ('C_task 1', dict(C_task=1.0)), ('C_task 3', dict(C_task=3.0)), ('C_embodiment 0.5', dict(C_embodiment=0.5)),
                    ('C_embodiment 1.5', dict(C_embodiment=1.5)), ('dwell 3 s', dict(dwell=3)), ('dwell 10 s', dict(dwell=10)),
                    ('flat embodiment prior', dict(e_prior=[1, 1, 1])), ('no gain floor', dict(allowed_settings=list(range(U.NS)))),
                    ('assumed noise 0.20', dict(noise=0.2)), ('assumed noise 0.05', dict(noise=0.05))]
        for name, ck in variants:
            vals, sw, lowg = [], [], []
            for sd in rseeds[:5]:
                res = tag(run_jobs(pool, scenario_jobs('UEE', sd, N, {}, 'uee', ck) + scenario_jobs('MT', sd, N, {}, 'fixed', dict(s='MT'))))
                vals.append(paired(res, 'UEE', 'MT')['diff']); sw.append(np.mean([r['switches'] for r in res['UEE']]))
            row = dict(variant=name, diff_mean=float(np.mean(vals)), diff_sd=float(np.std(vals, ddof=1)), switches=float(np.mean(sw)))
            S.append(row)
            print(f"[D] {name:24s} UEE-MT {row['diff_mean']:+.2f} (SD {row['diff_sd']:.2f})  switches {row['switches']:.1f}", flush=True)
        OUT['sensitivity'] = S

    json.dump(OUT, open('results/analyses_v2.json', 'w'), indent=2)
    print('done', flush=True)
