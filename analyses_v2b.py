"""Follow-up to analyses_v2.py: widen the tuning grids of the two rule-based baselines (the best values of the first
grid lay on its edge), re-evaluate the rules on the same seeds and scenarios, and inspect the controller's choices in
the world where neither task success nor the signals carry information about embodiment.
Writes results/analyses_v2b.json. Rule-minus-MT differences are on the same operators as in analyses_v2.json, so
UEE-minus-rule = (UEE - MT) - (rule - MT)."""
import json
from multiprocessing import Pool
import numpy as np
import uee_sim as U
from analyses_v2 import session, operators, run_jobs, tag, paired, scenario_jobs, FLOOR

if __name__ == '__main__':
    OUT = {}
    MAIN, N_MAIN, N = 2026, 300, 200
    with Pool() as pool:
        grid_g = [dict(lo=lo, hi=hi, window=wd) for lo in [0.7, 0.8, 0.9] for hi in [1.1, 1.2, 1.3] for wd in [30, 60, 90, 120]]
        grid_t = [dict(theta=th, window=wd) for th in [0.02, 0.05, 0.1, 0.15, 0.2, 0.25] for wd in [10, 20, 30, 45, 60]]
        jobs = []
        for k, ck in enumerate(grid_g):
            jobs += [(f'g{k}|{i}', (MAIN, i, 0), w, {}, 'rule_ghlt', ck) for i, w in operators(MAIN, N_MAIN)]
        for k, ck in enumerate(grid_t):
            jobs += [(f't{k}|{i}', (MAIN, i, 0), w, {}, 'rule_trigger', ck) for i, w in operators(MAIN, N_MAIN)]
        res = tag(run_jobs(pool, jobs))
        sg = sorted([dict(ck, E=float(np.mean([r['E'] for r in res[f'g{k}']]))) for k, ck in enumerate(grid_g)], key=lambda r: -r['E'])
        st = sorted([dict(ck, E=float(np.mean([r['E'] for r in res[f't{k}']]))) for k, ck in enumerate(grid_t)], key=lambda r: -r['E'])
        best_g = {k: sg[0][k] for k in ('lo', 'hi', 'window')}; best_t = {k: st[0][k] for k in ('theta', 'window')}
        OUT['tuning'] = dict(best_ghlt=best_g, best_trigger=best_t, top_ghlt=sg[:5], top_trigger=st[:5])
        print('best GHLT', sg[:3], '\nbest trigger', st[:3], flush=True)

        # main seeds: rule - MT
        seeds = [2026] + list(range(3001, 3010))
        M = []
        for sd in seeds:
            jobs = []
            for name, ctrl, ck in [('MT', 'fixed', dict(s='MT')), ('RuleGHLT', 'rule_ghlt', best_g), ('RuleTrigger', 'rule_trigger', best_t)]:
                jobs += [(f'{name}|{i}', (sd, i, 0), w, {}, ctrl, ck) for i, w in operators(sd, N_MAIN)]
            res = tag(run_jobs(pool, jobs))
            M.append(dict(seed=sd, **{f'{c}-MT': paired(res, c, 'MT')['diff'] for c in ['RuleGHLT', 'RuleTrigger']},
                          **{f'{c}_succ': float(np.mean([r['succ'] for r in res[c]])) for c in ['RuleGHLT', 'RuleTrigger']},
                          **{f'{c}_switches': float(np.mean([r['switches'] for r in res[c]])) for c in ['RuleGHLT', 'RuleTrigger']}))
        OUT['main_seeds'] = M
        print('main: RuleGHLT-MT %+.3f  RuleTrigger-MT %+.3f' % (np.mean([m['RuleGHLT-MT'] for m in M]), np.mean([m['RuleTrigger-MT'] for m in M])), flush=True)

        # robustness scenarios: rule - MT
        g0 = dict(ghlt_w_slope=0.0)
        scen = [('matched', {}), ('noise 0.45', dict(noise=0.45)), ('GHLT uninformative', g0),
                ('signals 0', dict(signal_strength=0.0)), ('additive cues', dict(cue_combination='additive'))]
        rseeds = [7] + list(range(4001, 4010))
        R = []
        for name, ok in scen:
            v = {'RuleGHLT-MT': [], 'RuleTrigger-MT': []}
            for sd in rseeds:
                res = tag(run_jobs(pool, scenario_jobs('MT', sd, N, ok, 'fixed', dict(s='MT')) + scenario_jobs('RuleGHLT', sd, N, ok, 'rule_ghlt', best_g) +
                                   scenario_jobs('RuleTrigger', sd, N, ok, 'rule_trigger', best_t)))
                for c in ['RuleGHLT', 'RuleTrigger']:
                    v[f'{c}-MT'].append(paired(res, c, 'MT')['diff'])
            R.append(dict(scenario=name, **{k: dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1))) for k, x in v.items()}))
            print(f"[R] {name:22s} RuleGHLT {R[-1]['RuleGHLT-MT']['mean']:+.2f}  RuleTrigger {R[-1]['RuleTrigger-MT']['mean']:+.2f}", flush=True)
        OUT['robustness'] = R

    # inspect the no-information world (task success and signals uninformative about embodiment)
    from collections import Counter
    ch = Counter(); e_end = []
    for i in range(40):
        w = int(np.random.default_rng([13, i]).integers(U.NW))
        op = U.Operator(np.random.default_rng([13, i, 1]), w=w, task_e_effect=0.0, signal_strength=0.0)
        h = U.run_uee(U.build_agent(allowed_settings=FLOOR, task_e_effect=0.0, signal_strength=0.0), op, T=120, dwell=5)
        ch.update([U.SETTINGS[s] for s in h['s'][60:]])
    tot = sum(ch.values())
    OUT['no_information_settings_last60'] = {f'g{g}m{m}': round(c / tot, 3) for (g, m), c in ch.most_common()}
    print('no-information world, settings in last 60 s (gain, mapping):', OUT['no_information_settings_last60'], flush=True)
    json.dump(OUT, open('results/analyses_v2b.json', 'w'), indent=2)
    print('done', flush=True)
