# UEE embodiment simulation

Code and results for the preprint

> Sara Falcone. *Embodiment-aware control by inference over the operator: a simulation study.* 2026. [arXiv link to be added]

The Universal Embodiment Engine (UEE) is a controller that infers a device operator's hidden embodiment level and
visuo-proprioceptive cue weighting from implicit gaze and pupil signals and task outcome, and chooses bounded device
settings (feedback gain × visuo-motor mapping) under explicit preferences. It is implemented as a discrete Active
Inference agent with [`inferactively-pymdp`](https://github.com/infer-actively/pymdp) 1.0.4 (classic interface,
`pymdp.legacy`), and tested on heterogeneous synthetic operators.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py      # main experiment
python ablation.py             # preferences and safeguards
python robustness.py           # model-operator mismatch
python generalization.py       # out-of-family operators and population sweep
python epistemic.py            # does the information-gain term matter?
python figures.py              # regenerates figures/ from results/
```

Each script runs in a few minutes on a laptop (tested with Python 3.14). All random seeds are fixed and ties in action
selection are broken deterministically, so the results in `results/` are reproduced exactly.

## Files

| File | What it does | Writes |
|---|---|---|
| `uee_sim.py` | Controller's generative model, synthetic operators, session runners (UEE and bandits) | — |
| `run_experiments.py` | 300 operators × 9 conditions × 120 s, paired (seed 2026) | `results/summary.json`, `per_operator.json`, `pw.npy`, `traj.npz` |
| `ablation.py` | Which preferences, information gain and the feedback floor matter (same operators) | `results/ablation.json` |
| `robustness.py` | 20 mismatch scenarios, 200 operators each (seed 7) | `results/robustness.json`, `robustness_traj.npz` |
| `generalization.py` | Continuous cue weighting, other match functions, population sweep (seed 11) | `results/generalization.json` |
| `epistemic.py` | UEE vs the same controller without the information-gain term, hardest scenarios | `results/epistemic.json` |
| `figures.py` | Figures 1–4 of the paper | `figures/*.pdf`, `figures/*.png` |

## Conditions in the main experiment

Mechanical Transparency (best fixed setting for the population), a yoked schedule (another operator's UEE settings),
the UEE with and without its 5-s dwell, an expected-utility controller (same inference, no information-gain term),
Thompson sampling and UCB bandits, a naive prediction-error minimizer, and an Oracle that knows the operator.

## Main results (embodiment level 0–2, last 60 s of a 120-s session; 300 operators)

| Condition | Embodiment, last 60 s | Embodiment, session | Task success | Setting changes |
|---|---|---|---|---|
| Mechanical Transparency | 0.73 ± 0.68 | 0.69 ± 0.63 | 0.82 ± 0.07 | 0 |
| Yoked schedule | 0.78 ± 0.81 | 0.75 ± 0.69 | 0.82 ± 0.08 | 1.5 ± 1.0 |
| Naive PE minimizer | 1.35 ± 0.63 | 1.19 ± 0.53 | 0.74 ± 0.14 | 9.0 ± 3.9 |
| **UEE** | 1.68 ± 0.34 | 1.51 ± 0.31 | 0.90 ± 0.04 | 1.5 ± 1.0 |
| UEE without dwell | 1.68 ± 0.35 | 1.53 ± 0.32 | 0.90 ± 0.04 | 3.4 ± 2.7 |
| Expected-utility controller | 1.68 ± 0.34 | 1.51 ± 0.31 | 0.90 ± 0.04 | 1.5 ± 1.0 |
| Thompson sampling | 0.85 ± 0.49 | 0.73 ± 0.32 | 0.75 ± 0.05 | 18.5 ± 2.4 |
| UCB | 1.30 ± 0.62 | 0.99 ± 0.43 | 0.80 ± 0.07 | 10.3 ± 3.9 |
| Oracle | 1.69 ± 0.33 | 1.61 ± 0.29 | 0.91 ± 0.04 | 0 |

The benefit comes from Bayesian inference over the operator's hidden state with explicit preferences: the
expected-utility controller matches the UEE exactly, and the model-free bandits fall well short.

## Caveat

The synthetic operators are close relatives of the controller's model. The simulation tests the controller's logic and
its failure modes, not whether real gaze and pupil signals behave this way; that requires studies with people.

## License

MIT (see `LICENSE`). If you use this code, please cite the preprint (see `CITATION.cff`).
