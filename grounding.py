"""Parameter grounding: compare quantities implied by the population model with published statistics
from the author's own studies, and test the controller under the departures those statistics suggest.
Writes results/grounding.json.

Published statistics used (all from published papers; no raw data):
  - Falcone et al. 2022a (TMB, N = 28): suppressing tactile feedback lowered ownership by 0.68 and agency by 0.30
    of the drop caused by breaking visuo-proprioceptive synchrony (ratio of mixed-model t * SE); no cue interactions;
    SoE-performance correlation R^2 = 0.13 (ownership) and 0.18 (agency), i.e. r = 0.36 and 0.42 across participants.
  - Rubber-hand illusion develops within seconds of synchronous stimulation (Ehrsson et al. 2004); used as a plausibility check only.
  - Pupil (Falcone et al. 2022b): direction only;
    effect magnitudes are reported at the session level and do not identify per-second informativeness, so
    the controller is tested with signals weaker than assumed (signal_strength < 1).
"""
import json
import numpy as np
from scipy import stats
import uee_sim as U

out = {}

# ---------------------------------------------------------------- 1. relative weight of the two cue manipulations
B = U.embodiment_transition()
def E(w, g, m, B=B):
    return U.stationary_E(w, g * U.NM + m, B)

rows = []
for w in range(U.NW):
    g_best, m_best = U.SETTINGS[U.best_setting(w)]
    e_best = E(w, 2, m_best)
    m_worst = min(range(U.NM), key=lambda m: U.match(w, m))
    rows.append(dict(w=w, best=e_best,
                     drop_gain_mid=e_best - E(w, 1, m_best), drop_gain_low=e_best - E(w, 0, m_best),
                     drop_mapping_worst=e_best - E(w, 2, m_worst)))
non_bal = [r for r in rows if r['w'] != 2]                     # balanced operators have no strongly mismatched mapping
ratio_mid = np.mean([r['drop_gain_mid'] / r['drop_mapping_worst'] for r in non_bal])
ratio_low = np.mean([r['drop_gain_low'] / r['drop_mapping_worst'] for r in non_bal])
out['cue_ratio'] = dict(per_type=rows, ratio_gain_mid_over_mapping=float(ratio_mid), ratio_gain_low_over_mapping=float(ratio_low),
                        published_ratio_ownership=0.68, published_ratio_agency=0.30)
print(f"[1] gain drop / mapping drop: mid gain {ratio_mid:.2f}, low gain {ratio_low:.2f}  (published 0.30-0.68)")

# ---------------------------------------------------------------- 2. interaction between the two cues (2 x 2 per type)
def interaction(Bm):
    vals = []
    for w in [0, 1, 3, 4]:
        m_best = U.SETTINGS[U.best_setting(w)][1]
        m_worst = min(range(U.NM), key=lambda m: U.match(w, m))
        hh, hl = E(w, 2, m_best, Bm), E(w, 2, m_worst, Bm)          # high gain: matched / mismatched mapping
        lh, ll = E(w, 0, m_best, Bm), E(w, 0, m_worst, Bm)          # low gain
        main_gain, main_map = (hh + hl - lh - ll) / 2, (hh + lh - hl - ll) / 2
        vals.append(dict(w=w, main_gain=main_gain, main_mapping=main_map, interaction=(hh - hl) - (lh - ll)))
    return vals
Badd = U.embodiment_transition(cue_combination='additive')
ip, ia = interaction(B), interaction(Badd)
out['interaction'] = dict(product=ip, additive=ia)
f = lambda v: np.mean([abs(x['interaction']) / (abs(x['main_gain']) + abs(x['main_mapping'])) for x in v])
print(f"[2] |interaction| relative to the main effects: product model {f(ip):.2f}, additive model {f(ia):.2f}")

# ---------------------------------------------------------------- 3. embodiment-performance correlation across people
def cue_study(n, rng_seed, trial=60):
    """Each synthetic participant experiences all nine settings for `trial` s in random order, as in the
    within-subject cue study; returns per-person mean embodiment and mean task success."""
    rng = np.random.default_rng(rng_seed)
    me, ms = [], []
    for i in range(n):
        op = U.Operator(np.random.default_rng([rng_seed, i]), w=int(rng.integers(U.NW)))
        es, ss = [], []
        for s in rng.permutation(U.NS):
            for _ in range(trial):
                obs = op.step(int(s)); es.append(op.e); ss.append(obs[4] == 0)
        me.append(np.mean(es)); ms.append(np.mean(ss))
    return np.array(me), np.array(ms)

me, ms = cue_study(2000, 31)
r_pop = stats.pearsonr(me, ms)[0]
rs = [stats.pearsonr(*cue_study(28, 1000 + k))[0] for k in range(200)]
out['soe_performance'] = dict(r_population=float(r_pop), r_n28_median=float(np.median(rs)), r_n28_iqr=[float(np.percentile(rs, 25)), float(np.percentile(rs, 75))],
                              published_r=[0.36, 0.42])
print(f"[3] r(embodiment, success) across people: {r_pop:.2f} (N = 28 studies: median {np.median(rs):.2f}, IQR {np.percentile(rs, 25):.2f}-{np.percentile(rs, 75):.2f}); published 0.36-0.42")

# ---------------------------------------------------------------- 4. time course: onset of high embodiment
t_high = []
for i in range(2000):
    w = int(np.random.default_rng([41, i]).integers(U.NW))
    op = U.Operator(np.random.default_rng([41, i, 1]), w=w)
    s = U.best_setting(w); t = 0
    while op.e < U.NE - 1 and t < 600:
        op.step(s); t += 1
    t_high.append(t)
t_high = np.array(t_high)
out['onset'] = dict(median=float(np.median(t_high)), iqr=[float(np.percentile(t_high, 25)), float(np.percentile(t_high, 75))],
                    mean=float(t_high.mean()), reference='rubber hand illusion develops within seconds (Ehrsson et al. 2004)')
print(f"[4] time to high embodiment at the suitable setting: median {np.median(t_high):.0f} s (IQR {np.percentile(t_high, 25):.0f}-{np.percentile(t_high, 75):.0f}); RHI develops within seconds")

# ---------------------------------------------------------------- 5. controller under weaker signals and additive cues
N, T, SEED = 200, 120, 13
DEFAULT, FLOOR = U.DEFAULT, U.FLOOR

def compare(label, op_kwargs=None, agent_kwargs=None, calib=None):
    op_kwargs = op_kwargs or {}; agent_kwargs = agent_kwargs or {}
    eu, em, eo, pw = [], [], [], []
    B_op = U.embodiment_transition(cue_combination=op_kwargs.get('cue_combination', 'product'))
    oracle = {w: max(range(U.NS), key=lambda s: (round(U.stationary_E(w, s, B_op), 9), U.SETTINGS[s][0])) for w in range(U.NW)}
    mt = max([s for s in range(U.NS) if U.SETTINGS[s][0] == 2], key=lambda s: np.mean([U.stationary_E(w, s, B_op) for w in range(U.NW)]))
    for i in range(N):
        w = int(np.random.default_rng([SEED, i]).integers(U.NW))
        mk = lambda: U.Operator(np.random.default_rng([SEED, i, 1]), w=w, **op_kwargs)
        ak = dict(agent_kwargs)
        if calib is not None:                                  # session-start calibration: prior mass `calib` on the true weighting
            ak['w_prior'] = [calib if k == w else (1 - calib) / (U.NW - 1) for k in range(U.NW)]
        h = U.run_uee(U.build_agent(allowed_settings=FLOOR, **ak), mk(), T=T, dwell=5)
        eu.append(h['e'][60:].mean()); pw.append(h['p_w_true'][-1])
        em.append(U.run_schedule(mk(), [mt] * T)['e'][60:].mean())
        eo.append(U.run_schedule(mk(), [oracle[w]] * T)['e'][60:].mean())
    eu, em, eo = map(np.array, (eu, em, eo))
    d = eu - em
    ci = [float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(N)), float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(N))]
    r = dict(label=label, UEE=float(eu.mean()), MT=float(em.mean()), Oracle=float(eo.mean()), diff=float(d.mean()), ci95=ci,
             dz=float(d.mean() / d.std(ddof=1)), share_of_oracle_gain=float(d.mean() / (eo - em).mean()), belief_true_w_end=float(np.mean(pw)))
    print(f"[5] {label:52s} UEE {r['UEE']:.2f}  MT {r['MT']:.2f}  Oracle {r['Oracle']:.2f}  diff {r['diff']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]  "
          f"share {r['share_of_oracle_gain']:.0%}  P(W) {r['belief_true_w_end']:.2f}", flush=True)
    return r

R = [compare('matched model (reference)')]
for ss in [0.75, 0.5, 0.25, 0.0]:
    R.append(compare(f'signals {ss:.2f} as strong, controller unaware', dict(signal_strength=ss)))
for ss in [0.5, 0.25]:
    R.append(compare(f'signals {ss:.2f} as strong, controller calibrated', dict(signal_strength=ss), dict(signal_strength=ss)))
R.append(compare('signals 0.50 as strong + noise 0.30, unaware', dict(signal_strength=0.5, noise=0.30)))
R.append(compare('additive cue combination (operators)', dict(cue_combination='additive')))
R.append(compare('additive cues + signals 0.50, unaware', dict(cue_combination='additive', signal_strength=0.5)))
# gaze lead carries no information about cue weighting AND the embodiment signals are weaker than assumed
g0 = dict(ghlt_w_slope=0.0)
R.append(compare('gaze lead uninformative, calibrated (reference)', g0, g0))
for ss in [0.5, 0.25, 0.0]:
    R.append(compare(f'gaze lead uninformative + signals {ss:.2f}, calibrated', dict(g0, signal_strength=ss), dict(g0, signal_strength=ss)))
for cal in [0.6, 0.8]:
    R.append(compare(f'  same at signals 0.50, + calibration prior {cal}', dict(g0, signal_strength=0.5), dict(g0, signal_strength=0.5), calib=cal))
out['robustness'] = R
json.dump(out, open('results/grounding.json', 'w'), indent=2)
