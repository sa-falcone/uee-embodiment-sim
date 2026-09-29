"""Does the epistemic (information-gain) term matter? UEE vs the same controller without it (expected utility only),
paired on the same operators, in the scenarios where learning cue weighting is hardest. Writes results/epistemic.json.
(In the main experiment the two are identical on all 300 operators; see results/summary.json.)"""
import json
import numpy as np, uee_sim as U, zlib
OUT = []
def cmp(label, mk_op, agent_kw, n=200):
    d, eu, ev = [], [], []
    for i in range(n):
        op_a, op_b = mk_op(i), mk_op(i)
        a = U.run_uee(U.build_agent(allowed_settings=U.FLOOR, **agent_kw), op_a)['e']
        b = U.run_uee(U.build_agent(allowed_settings=U.FLOOR, info_gain=False, **agent_kw), op_b)['e']
        eu.append(a[60:].mean()); ev.append(b[60:].mean()); d.append(a[60:].mean() - b[60:].mean())
    d = np.array(d); h = 1.96 * d.std(ddof=1) / np.sqrt(n)
    OUT.append(dict(label=label, UEE=float(np.mean(eu)), utility_only=float(np.mean(ev)), diff=float(d.mean()),
                    ci95=[float(d.mean() - h), float(d.mean() + h)], n_differing=int((d != 0).sum())))
    print(f'{label:50s} UEE {np.mean(eu):.3f} utility-only {np.mean(ev):.3f} diff {d.mean():+.3f} [{d.mean()-h:+.3f},{d.mean()+h:+.3f}] ops differing {(d!=0).sum()}', flush=True)
# robustness seeds: GHLT uninformative, calibrated / unaware
def rob(kw):
    def mk(i):
        w = int(np.random.default_rng([7, i]).integers(U.NW)); return U.Operator(np.random.default_rng([7, i, 1]), w=w, **kw)
    return mk
cmp('GHLT uninformative, controller calibrated', rob(dict(ghlt_w_slope=0.0)), dict(ghlt_w_slope=0.0))
cmp('GHLT uninformative, controller unaware', rob(dict(ghlt_w_slope=0.0)), {})
cmp('signal noise 0.45', rob(dict(noise=0.45)), {})
lab = 'continuous cue weighting'
def mkc(i):
    r = np.random.default_rng([11, zlib.crc32(lab.encode()) % 10**6, i]); p = float(r.uniform(-1, 1))
    return U.Operator(np.random.default_rng([11, zlib.crc32(lab.encode()) % 10**6, i, 1]), w_pos=p)
cmp(lab, mkc, {})
json.dump(OUT, open('results/epistemic.json', 'w'), indent=2)
