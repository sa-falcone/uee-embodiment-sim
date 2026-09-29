"""
Universal Embodiment Engine (UEE) - discrete Active Inference simulation for the preprint.

The UEE is an Active Inference agent whose "world" is the operator. It never observes the
operator's own sensory predictions; it infers a hidden operator state from implicit signals
and task outcome, and chooses bounded device settings by minimising expected free energy.

Hidden state factors (controller's generative model)
    f0  E  embodiment level              {0 low, 1 mid, 2 high}
    f1  W  visuo-proprioceptive weighting, 5 graded types at positions -1, -0.6, 0, +0.6, +1
           (vision-dominant, mildly vision, balanced, mildly proprioceptive, proprioception-dominant)
    f2  S  current device setting        9 = feedback gain g {0 low, 1 mid, 2 high} x mapping m {0 visual, 1 mixed, 2 proprioceptive}
Observation modalities (each discretised to a few bins)
    o0  pupil diameter      {small, medium, large}   rises with low E AND with sensory conflict
    o1  gaze transition entropy {low, medium, high}  rises with low E
    o2  gaze-hand lead time {short, medium, long}    informative about W (and weakly about E)
    o3  look-ahead ratio    {low, medium, high}      rises with E
    o4  task outcome        {success, error}         depends on feedback gain g and on E
Control: one action u in {0..8} = "set the device to setting u". The action drives two factors:
S (S' = u) and E (E' ~ B(E' | E, W, u)), so that the one-step expected free energy sees the effect
of the chosen setting on embodiment, exactly as in the simulated operators.

The simulated operators (generative process) share this structure but differ individually
(cue weighting, adaptation rates, signal noise); the controller only knows the population model.
"""
import numpy as np
from pymdp.legacy.agent import Agent
from pymdp.legacy import utils

NE, NW = 3, 5
NG, NM = 3, 3
NS = NG * NM
SETTINGS = [(g, m) for g in range(NG) for m in range(NM)]      # index s -> (g, m)
W_POS = np.array([-1.0, -0.6, 0.0, 0.6, 1.0])                   # cue weighting: -1 vision-dominant ... +1 proprioception-dominant
M_POS = np.array([-1.0, 0.0, 1.0])                               # mapping: visual, mixed, proprioceptive
MATCH_WIDTH = 0.5                                                # sigma of the match function
GAIN_FACTOR = np.array([0.2, 0.7, 1.0])                          # kappa(g): how much cue information each gain delivers
LAPSE = 0.03                                                     # spontaneous loss of one embodiment level per second (population value)
P_SUCCESS_GAIN = [0.45, 0.70, 0.85]
P_SUCCESS_E = [-0.10, 0.0, 0.10]
W_NAMES = ["vision-dominant", "mildly vision", "balanced", "mildly proprio.", "proprio.-dominant"]
M_NAMES = ["visual", "mixed", "proprio."]
FLOOR = [s for s, (g, m) in enumerate(SETTINGS) if g > 0]         # feedback-gain floor: never the lowest gain


def match_gauss(pos, m, width=MATCH_WIDTH):
    """Population match function: mu = exp(-(pos_W - pos_m)^2 / (2 sigma^2))."""
    return float(np.exp(-0.5 * ((pos - M_POS[m]) / width) ** 2))


def match_triangular(pos, m, half_width=1.0):
    """Alternative (out-of-family) match function: mu = max(0, 1 - |pos_W - pos_m| / half_width)."""
    return float(max(0.0, 1.0 - abs(pos - M_POS[m]) / half_width))


def match_wide(pos, m):
    """Alternative (out-of-family) match function: a Gaussian 1.6 times wider than the controller assumes."""
    return match_gauss(pos, m, width=0.8)


def match(w, m):
    """mu(W, m) = exp(-(pos_W - pos_m)^2 / (2 sigma^2)): how well mapping m suits an operator with weighting w."""
    return match_gauss(W_POS[w], m)


def conflict(w, s):
    """c(W, S): sensory conflict the operator experiences, strong feedback delivered through the wrong mapping."""
    g, m = SETTINGS[s]
    return GAIN_FACTOR[g] * (1.0 - match(w, m))


def softbins(mu, width=0.55, n=3):
    """Discretise a latent value mu in [0,1] into n ordered bins (centres 0, 0.5, 1) with a Gaussian kernel."""
    centres = np.linspace(0, 1, n)
    p = np.exp(-0.5 * ((centres - mu) / width) ** 2)
    return p / p.sum()


# ------------------------------------------------------------------ model components
def embodiment_transition(up_rate=0.25, down_rate=0.25, low_gain_penalty=0.25, lapse=LAPSE, w_pos=None, match_fn=match_gauss):
    """B_E with shape (E', E, W, S): embodiment rises with matched, informative feedback and
    falls with mismatch, with feedback so low that the body has too few cues (additive cue model), and by
    spontaneous lapses at rate `lapse`, so that no level is absorbing."""
    w_pos = W_POS if w_pos is None else np.atleast_1d(w_pos)
    B = np.zeros((NE, NE, len(w_pos), NS))
    for e in range(NE):
        for w in range(len(w_pos)):
            for s in range(NS):
                g, m = SETTINGS[s]
                mu = match_fn(w_pos[w], m)
                p_up = min(up_rate * mu * GAIN_FACTOR[g], 0.95)
                p_down = min(down_rate * (1 - mu) + (low_gain_penalty if g == 0 else 0.0) + lapse, 0.95)
                if e == NE - 1:
                    p_up = 0.0
                if e == 0:
                    p_down = 0.0
                B[e, e, w, s] = 1.0 - p_up - p_down
                if e < NE - 1:
                    B[e + 1, e, w, s] = p_up
                if e > 0:
                    B[e - 1, e, w, s] = p_down
    return B


def stationary_E(w, s, B=None):
    """Long-run mean embodiment level (0-2) of an operator of type w held at setting s."""
    B = embodiment_transition() if B is None else B
    P = B[:, :, w, s]
    vals, vecs = np.linalg.eig(P)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))]); pi = pi / pi.sum()
    return float(pi @ np.arange(NE))


def best_setting(w):
    """Oracle: the setting with the highest long-run embodiment for type w (ties -> higher gain)."""
    return max(range(NS), key=lambda s: (round(stationary_E(w, s), 9), SETTINGS[s][0]))


def best_fixed_setting(type_probs=None):
    """Mechanical Transparency: the single high-gain setting with the highest long-run embodiment averaged over the
    population (type_probs over the five types; uniform by default)."""
    p = np.ones(NW) / NW if type_probs is None else np.asarray(type_probs, float)
    return max([s for s in range(NS) if SETTINGS[s][0] == 2], key=lambda s: sum(p[w] * stationary_E(w, s) for w in range(NW)))


def best_setting_for(pos, match_fn=match_gauss):
    """Oracle for an arbitrary operator: best setting for cue-weighting position `pos` under match function `match_fn`."""
    B = embodiment_transition(w_pos=[pos], match_fn=match_fn)
    return max(range(NS), key=lambda s: (round(stationary_E(0, s, B), 9), SETTINGS[s][0]))


def best_fixed_setting_for(positions, match_fn=match_gauss):
    """MT for an arbitrary population of cue-weighting positions."""
    B = embodiment_transition(w_pos=positions, match_fn=match_fn)
    return max([s for s in range(NS) if SETTINGS[s][0] == 2], key=lambda s: np.mean([stationary_E(k, s, B) for k in range(len(positions))]))


def likelihoods(noise=0.1, pupil_conflict=0.4, ghlt_w_slope=0.35, w_pos=None, match_fn=match_gauss):
    """A matrices, each with shape (obs, E, W, S). `noise` mixes the four implicit signals toward uniform;
    the task outcome is not mixed, so its success probability is exactly the one stated in the paper."""
    w_pos = W_POS if w_pos is None else np.atleast_1d(w_pos)
    A = utils.obj_array(5)
    shape = (3, NE, len(w_pos), NS)
    pupil, gte, ghlt, lat = (np.zeros(shape) for _ in range(4))
    task = np.zeros((2, NE, len(w_pos), NS))
    for e in range(NE):
        for w in range(len(w_pos)):
            for s in range(NS):
                g, m = SETTINGS[s]
                en = e / (NE - 1)
                c = GAIN_FACTOR[g] * (1.0 - match_fn(w_pos[w], m))
                pupil[:, e, w, s] = softbins(np.clip(0.75 - 0.5 * en + pupil_conflict * c, 0, 1))
                gte[:, e, w, s] = softbins(1.0 - en)
                ghlt[:, e, w, s] = softbins(np.clip(0.5 + ghlt_w_slope * w_pos[w] + 0.2 * (en - 0.5), 0, 1), width=0.35)
                lat[:, e, w, s] = softbins(en)
                p_succ = P_SUCCESS_GAIN[g] + P_SUCCESS_E[e]
                task[:, e, w, s] = [p_succ, 1 - p_succ]
    for i, a in enumerate([pupil, gte, ghlt, lat]):
        A[i] = (1 - noise) * a + noise / a.shape[0]
    A[4] = task
    return A


DEFAULT = best_fixed_setting()                                    # high gain, mapping best for the population


def build_agent(C_task=2.0, C_embodiment=1.0, C_pupil=0.0, allowed_settings=None, noise=0.1, w_drift=0.0, w_prior=None,
                ghlt_w_slope=0.35, info_gain=True):
    """Construct the UEE. C_* are log-preferences; allowed_settings implements the feedback floor;
    info_gain=False removes the epistemic term (expected-utility controller with the same inference)."""
    A = likelihoods(noise=noise, ghlt_w_slope=ghlt_w_slope)
    B = utils.obj_array(3)
    B[0] = embodiment_transition()                                  # (E', E, W, u): the chosen setting drives embodiment
    B[1] = ((1 - w_drift) * np.eye(NW) + w_drift / NW)[:, :, None]  # weighting stable within a session (w_drift > 0 lets beliefs track change)
    B[2] = np.zeros((NS, NS, NS))
    for u in range(NS):
        B[2][u, :, u] = 1.0                                          # action u sets the device to setting u
    C = utils.obj_array(5)
    C[0] = C_pupil * np.array([1.0, 0.0, -1.0])                      # (naive controller) prefer small pupils
    C[1] = C_embodiment * np.array([1.0, 0.0, -1.0])                 # prefer structured gaze (low GTE)
    C[2] = np.zeros(3)
    C[3] = C_embodiment * np.array([-1.0, 0.0, 1.0])                 # prefer anticipatory look-ahead (high LAT)
    C[4] = C_task * np.array([1.0, -1.0])                            # prefer task success
    D = utils.obj_array(3)
    D[0] = np.array([0.5, 0.35, 0.15])
    D[1] = np.ones(NW) / NW if w_prior is None else np.asarray(w_prior, float) / np.sum(w_prior)  # w_prior: session-start calibration
    D[2] = np.zeros(NS); D[2][DEFAULT] = 1.0                           # sessions start at the default setting
    allowed = list(range(NS)) if allowed_settings is None else list(allowed_settings)
    policies = [np.array([[a, 0, a]]) for a in allowed]              # one action drives E (factor 0) and S (factor 2)
    agent = Agent(A=A, B=B, C=C, D=D,
                  A_factor_list=[[0, 1, 2]] * 5,
                  B_factor_list=[[0, 1], [1], [2]],
                  num_controls=[NS, 1, NS], control_fac_idx=[0, 2],
                  policies=policies, policy_len=1, use_states_info_gain=info_gain,
                  action_selection="deterministic")
    return agent


def choose_setting(agent, current):
    """argmin_u G(u). pymdp breaks exact ties with the unseeded global RNG; here ties are broken
    deterministically in favour of keeping the current setting, otherwise the lowest index."""
    q_pi, _ = agent.infer_policies()
    best = np.flatnonzero(np.abs(q_pi - q_pi.max()) <= 1e-8)
    options = [int(agent.policies[k][0, 2]) for k in best]
    return current if current in options else options[0]


# ------------------------------------------------------------------ simulated operator
class Operator:
    """Generative process: one simulated person with individual parameters."""

    def __init__(self, rng, w=None, noise=None, rate_scale=1.0, lapse_scale=1.0, pupil_conflict=0.4, ghlt_w_slope=0.35, w_switch=None,
                 w_pos=None, match_fn=match_gauss):
        """w: one of the five types. w_pos (optional): a continuous cue-weighting position in [-1, 1] instead of a type;
        match_fn: the operator's own match function (the controller always assumes match_gauss)."""
        self.rng = rng
        self.w = rng.integers(NW) if w is None else w
        self.e = 0
        self.t = 0
        self.w_switch = w_switch                     # (time, new weighting) for a mid-session change
        up = 0.25 * rng.uniform(0.7, 1.3) * rate_scale
        down = 0.25 * rng.uniform(0.7, 1.3) * rate_scale
        lapse = LAPSE * rng.uniform(0.5, 1.5) * lapse_scale
        own = dict(w_pos=None if w_pos is None else [w_pos], match_fn=match_fn)
        self.B = embodiment_transition(up_rate=up, down_rate=down, lapse=lapse, **own)
        self.A = likelihoods(noise=rng.uniform(0.05, 0.25) if noise is None else noise,
                             pupil_conflict=pupil_conflict, ghlt_w_slope=ghlt_w_slope, **own)
        if w_pos is None:
            self.k = self.w                          # index into the operator's own matrices
        else:
            self.k = 0
            self.w = int(np.argmin(np.abs(W_POS - w_pos)))   # nearest type, used only to report beliefs

    def step(self, s):
        if self.w_switch is not None and self.t == self.w_switch[0]:
            self.w = self.k = self.w_switch[1]
        self.t += 1
        self.e = self.rng.choice(NE, p=self.B[:, self.e, self.k, s])
        obs = [int(self.rng.choice(a.shape[0], p=a[:, self.e, self.k, s])) for a in self.A]
        return obs


# ------------------------------------------------------------------ session runners
def run_uee(agent, op, T=120, dwell=5, s0=None):
    s = DEFAULT if s0 is None else s0
    hist = dict(s=[], e=[], succ=[], w_map=[], p_w_true=[], w_true=[])
    for t in range(T):
        obs = op.step(s)                               # E_t ~ B(E_{t-1}, W, s), o_t ~ A(E_t, W, s)
        qs = agent.infer_states(obs)                   # prior: D at t = 0, then B applied with the setting of this step
        hist['s'].append(s); hist['e'].append(op.e); hist['succ'].append(obs[4] == 0)
        hist['w_map'].append(int(np.argmax(qs[1]))); hist['p_w_true'].append(float(qs[1][op.w])); hist['w_true'].append(int(op.w))
        if t % dwell == dwell - 1:                     # choose a new setting at most every `dwell` steps
            s = choose_setting(agent, s)
        agent.action = np.array([float(s), 0.0, float(s)])   # the setting that will be applied during the next step
    return {k: np.array(v) for k, v in hist.items()}


def run_schedule(op, schedule):
    hist = dict(s=[], e=[], succ=[])
    for s in schedule:
        obs = op.step(int(s))
        hist['s'].append(int(s)); hist['e'].append(op.e); hist['succ'].append(obs[4] == 0)
    return {k: np.array(v) for k, v in hist.items()}


def bandit_reward(obs):
    """Per-second reward observed by the bandits: task success plus the two high-embodiment signatures, in [0, 1]."""
    return ((obs[4] == 0) + (obs[1] == 0) + (obs[3] == 2)) / 3.0


def run_bandit(op, kind, rng, T=120, dwell=5, allowed=None, c=0.1):
    """Model-free adaptive baseline: each admissible setting is an arm; a new arm is chosen every `dwell` s.
    kind='ts': Thompson sampling with Beta posteriors on the per-second reward (fractional updates);
    kind='ucb': UCB1 on the mean per-second reward, trying untried arms first (c = 0.1, tuned on the main operators)."""
    arms = list(FLOOR if allowed is None else allowed)
    a, b = np.ones(NS), np.ones(NS)                  # Beta(1, 1) priors
    n, tot = np.zeros(NS), np.zeros(NS)
    s = DEFAULT
    hist = dict(s=[], e=[], succ=[])
    for t in range(T):
        obs = op.step(s)
        r = bandit_reward(obs)
        a[s] += r; b[s] += 1 - r; n[s] += 1; tot[s] += r
        hist['s'].append(s); hist['e'].append(op.e); hist['succ'].append(obs[4] == 0)
        if t % dwell == dwell - 1:
            if kind == 'ts':
                s = max(arms, key=lambda k: rng.beta(a[k], b[k]))
            else:
                untried = [k for k in arms if n[k] == 0]
                N = n[arms].sum()
                s = untried[0] if untried else max(arms, key=lambda k: tot[k] / n[k] + c * np.sqrt(2 * np.log(N) / n[k]))
    return {k: np.array(v) for k, v in hist.items()}
