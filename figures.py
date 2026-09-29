"""Figures for the UEE preprint. Drawn at the printed text width (15.5 cm) with 7-8 pt text, so that
they are included at 100% scale. Neutral palette: baselines in greys, the UEE in one accent colour, the naive
controller in a second; conditions are also distinguished by line style and marker.
Run after run_experiments.py and robustness.py:  python figures.py"""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import uee_sim as U

INK, MUTED, GRID = "#1F1F1F", "#6B6B6B", "#E6E6E6"
ACCENT, ACCENT2 = "#B8322A", "#6A51A3"          # UEE, naive controller (validated pair)
MT_C, YOKE_C, ORACLE_C = "#7A7A7A", "#A9A9A9", "#1F1F1F"
W_CM, cm = 15.5, 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.it": "Helvetica Neue:italic",
                     "mathtext.rm": "Helvetica Neue"})

S = json.load(open('results/summary.json'))['summary']
traj = np.load('results/traj.npz'); pw = np.load('results/pw.npy')
per = json.load(open('results/per_operator.json')); R = {r['label'].strip(): r for r in json.load(open('results/robustness.json'))}
RT = np.load('results/robustness_traj.npz')
WS = np.array(per['w'])

COND = {  # key: (label, colour, linestyle, marker, linewidth)
    'MT':      ('Mechanical Transparency', MT_C, '-', 's', 1.1),
    'Yoked':   ('Yoked schedule', YOKE_C, (0, (4, 1.5)), 'v', 1.1),
    'NaivePE': ('Naive PE minimizer', ACCENT2, (0, (1, 1.2)), 'X', 1.3),
    'Oracle':  ('Oracle', ORACLE_C, (0, (5, 1.5, 1, 1.5)), 'D', 1.0),
    'UEE':     ('UEE', ACCENT, '-', 'o', 1.6),
    'TS':      ('Thompson sampling', "#008C7A", (0, (3, 1, 1, 1)), 'P', 1.1),
    'UCB':     ('UCB', "#C47F00", (0, (6, 2)), '^', 1.1),
}
BANDITS = ['TS', 'UCB']
ORDER = ['MT', 'Yoked', 'NaivePE', 'Oracle', 'UEE']


def style(ax, grid='y'):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)


def label(ax, L, T, x=-0.16):
    ax.text(x, 1.06, L, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)
    ax.text(x + 0.07, 1.06, T, transform=ax.transAxes, fontsize=8, va="bottom", ha="left", color=INK)


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)


# ======================================================================= Figure 1: generative model + problem structure
fig = plt.figure(figsize=(W_CM * cm, 6.2 * cm))
ax = fig.add_axes([0.0, 0.0, 0.56, 1.0]); ax.set_xlim(0, 9.3); ax.set_ylim(-0.1, 6.2); ax.axis("off")
R_N = 0.42


def node(x, y, txt, kind="hidden", sub=None):
    if kind == "action":
        ax.add_patch(Rectangle((x - R_N, y - R_N), 2 * R_N, 2 * R_N, fc="white", ec=ACCENT, lw=1.1))
        col = ACCENT
    else:
        ax.add_patch(Circle((x, y), R_N, fc="#D9D9D9" if kind == "obs" else "white", ec=INK, lw=0.9))
        col = INK
    ax.text(x, y, txt, ha="center", va="center", fontsize=9, color=col)
    if sub:
        ax.text(x, y - R_N - 0.12, sub, ha="center", va="top", fontsize=6.5, color=MUTED, linespacing=1.1)


def edge(p, q, col=INK, rad=0.0, lw=0.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=col, shrinkA=13, shrinkB=13,
                                 connectionstyle=f"arc3,rad={rad}"))


xE0, xE1, yE = 1.2, 4.6, 3.0
node(xE0, yE, r"$E_{t}$", sub="embodiment")
node(xE1, yE, r"$E_{t+1}$")
node(2.9, 4.75, r"$W$", sub=None); ax.text(2.35, 4.75, "cue weighting\n(fixed in session)", fontsize=6.5, color=MUTED, va="center", ha="right")
node(2.9, 1.25, r"$u_t$", kind="action"); ax.text(2.35, 1.25, "device\nsetting", ha="right", va="center", fontsize=6.5, color=ACCENT)
node(xE1, 1.25, r"$S_{t+1}$")
node(7.0, 3.0, r"$o_{t+1}$", kind="obs")
ax.text(7.55, 3.0, "pupil\nGTE\nGHLT\nLAT\ntask", fontsize=6.5, color=MUTED, va="center", linespacing=1.15)
edge((xE0, yE), (xE1, yE)); edge((2.9, 4.75), (xE1, yE)); edge((2.9, 1.25), (xE1, yE), col=ACCENT); edge((2.9, 1.25), (xE1, 1.25), col=ACCENT)
edge((xE1, yE), (7.0, 3.0)); edge((xE1, 1.25), (7.0, 3.0)); edge((2.9, 4.75), (7.0, 3.0), rad=-0.25)
ax.text(0.05, 6.15, "a", fontsize=9, fontweight="bold", va="top"); ax.text(0.38, 6.15, "Controller's generative model (one step)", fontsize=8, va="top")
ax.text(0.15, 0.05, r"$u_t = \arg\min_{u \in \mathcal{U}}\, G(u)$, every 5 s; $\mathcal{U}$: 3 gains × 3 mappings, lowest gain excluded", fontsize=6.6, color=ACCENT, va="bottom")

# panel b: long-run embodiment for each setting (rows) and operator type (columns)
from matplotlib.colors import LinearSegmentedColormap
axb = fig.add_axes([0.665, 0.12, 0.25, 0.72])
stat = np.array([[U.stationary_E(w, s) for w in range(U.NW)] for s in range(U.NS)])
cmap = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", "#E9B8B2", ACCENT, "#5A1712"])
axb.imshow(stat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
for s_ in range(U.NS):
    for w in range(U.NW):
        v = stat[s_, w]
        axb.text(w, s_, f"{v:.1f}", ha="center", va="center", fontsize=5.8, color="white" if v > 1.15 else INK)
for w in range(U.NW):
    b_ = U.best_setting(w)
    axb.add_patch(Rectangle((w - 0.5, b_ - 0.5), 1, 1, fill=False, ec=INK, lw=0.9, ls=(0, (2, 1))))
axb.add_patch(Rectangle((-0.5, U.DEFAULT - 0.5), U.NW, 1, fill=False, ec=INK, lw=1.3))
axb.axhline(2.5, color="white", lw=2)
axb.set_xticks(range(U.NW)); axb.set_xticklabels(["vis.", "mild\nvis.", "bal.", "mild\nprop.", "prop."], fontsize=6.3)
axb.set_yticks(range(U.NS)); axb.set_yticklabels([f"{['low', 'mid', 'high'][g]} · {['visual', 'mixed', 'prop.'][m]}" for g, m in U.SETTINGS], fontsize=6.3)
axb.tick_params(length=0); [axb.spines[k].set_visible(False) for k in axb.spines]
axb.set_xlabel("operator's cue weighting W", labelpad=3)
axb.text(U.NW - 0.35, U.DEFAULT, "MT", fontsize=6.3, va="center", ha="left", color=INK)
axb.text(U.NW - 0.35, 1.0, "below\nfloor", fontsize=6.3, va="center", ha="left", color=MUTED)
axb.text(-3.3, -1.1, "b", fontsize=9, fontweight="bold", va="bottom", transform=axb.transData)
axb.text(-2.85, -1.1, r"Long-run embodiment (0–2)", fontsize=8, va="bottom", transform=axb.transData)
save(fig, "fig1_model")

# ======================================================================= Figure 2: main results
fig, axs = plt.subplots(2, 2, figsize=(W_CM * cm, 10.2 * cm), gridspec_kw=dict(hspace=0.62, wspace=0.30, top=0.84, bottom=0.09, left=0.08, right=0.98))
a, b, c, d = axs.ravel()

t = np.arange(1, pw.shape[1] + 1); m = pw.mean(0); e = 1.96 * pw.std(0, ddof=1) / np.sqrt(pw.shape[0])
a.fill_between(t, m - e, m + e, color=ACCENT, alpha=0.18, lw=0); a.plot(t, m, color=ACCENT, lw=1.6)
a.axhline(1 / U.NW, ls=(0, (1, 1.5)), color=MUTED, lw=0.8); a.text(118, 1 / U.NW + 0.02, "chance", ha="right", va="bottom", fontsize=6.5, color=MUTED)
for tt in (10, 30):
    a.plot(tt, m[tt - 1], 'o', color=ACCENT, ms=3.5, mec="white", mew=0.5)
    a.annotate(f"{m[tt - 1]:.2f} at {tt} s", (tt, m[tt - 1]), xytext=(8, -9), textcoords="offset points", fontsize=6.8, color=INK)
a.set_xlabel("time (s)"); a.set_ylabel("belief in true type"); a.set_ylim(0, 1.03); a.set_xlim(0, 120); a.set_xticks([0, 30, 60, 90, 120]); style(a)
label(a, "A", "Inferring the operator", x=-0.2)

k = 5
for key in ['MT', 'Yoked', 'NaivePE', 'Oracle', 'UEE']:
    lab, col, ls, mk, lw = COND[key]
    y = np.convolve(traj[key].mean(0), np.ones(k) / k, mode='valid'); x = np.arange(len(y)) + k // 2 + 1
    b.plot(x, y, ls=ls, color=col, lw=lw, marker=mk, markevery=(10, 25), ms=3.2, mec="white", mew=0.4)
b.axvspan(60, 120, color="#F4F4F4", lw=0, zorder=0); b.text(90, 0.08, "outcome window", ha="center", fontsize=6.5, color=MUTED)
b.set_xlabel("time (s)"); b.set_ylabel("embodiment level (0–2)"); b.set_ylim(0, 2.05); b.set_xlim(0, 120); b.set_xticks([0, 30, 60, 90, 120]); style(b)
label(b, "B", "Embodiment over the session", x=-0.2)

for key in ORDER + BANDITS:
    lab, col, ls, mk, lw = COND[key]
    E = per[key]['E']; sc = per[key]['succ']
    c.errorbar(np.mean(E), np.mean(sc), xerr=ci(E), yerr=ci(sc), fmt=mk, color=col, ms=5 if key != 'Oracle' else 6.5, capsize=2, lw=0.9,
               mec="white" if key != 'Oracle' else col, mfc=col if key != 'Oracle' else "none", mew=0.6 if key != 'Oracle' else 1.0, zorder=3 if key == 'UEE' else 2)
c.set_xlabel("embodiment level, last 60 s"); c.set_ylabel("task success rate"); c.margins(x=0.12, y=0.15); style(c)
label(c, "C", "Embodiment and task success", x=-0.2)

offs = {'MT': -0.28, 'Yoked': -0.14, 'NaivePE': 0.0, 'Oracle': 0.14, 'UEE': 0.28}
for key in ORDER:
    lab, col, ls, mk, lw = COND[key]
    E = np.array(per[key]['E'])
    for w in range(U.NW):
        v = E[WS == w]
        d.errorbar(w + offs[key], v.mean(), yerr=ci(v) if v.std() > 0 else None, fmt=mk, color=col, ms=4.2, capsize=1.2, lw=0.8,
                   mec="white" if key != 'Oracle' else col, mfc=col if key != 'Oracle' else "none", mew=0.5 if key != 'Oracle' else 0.9)
d.set_xticks(range(U.NW)); d.set_xticklabels([f"{n}\n({np.sum(WS == k)})" for k, n in enumerate(["vis.", "mild vis.", "bal.", "mild prop.", "prop."])], fontsize=6.5)
d.set_ylabel("embodiment level, last 60 s"); d.set_ylim(-0.08, 2.12); d.set_xlim(-0.5, U.NW - 0.5); style(d); d.tick_params(axis='x', length=0)
label(d, "D", "By operator type", x=-0.2)

handles = [Line2D([0], [0], color=COND[k][1], ls=COND[k][2], marker=COND[k][3], lw=COND[k][4], ms=4,
                  mfc=COND[k][1] if k != 'Oracle' else "none", mec=COND[k][1] if k == 'Oracle' else "white", label=COND[k][0] + (" (C only)" if k in BANDITS else "")) for k in ["MT", "UEE", "Yoked", "TS", "NaivePE", "UCB", "Oracle"]]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.02), columnspacing=1.3, handlelength=2.6, handletextpad=0.5)
save(fig, "fig2_results")

# ======================================================================= Figure 3: where the controller fails
fig, (a, b) = plt.subplots(1, 2, figsize=(W_CM * cm, 5.4 * cm), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.55, left=0.30, right=0.98, top=0.84, bottom=0.2))
rows = [("matched model (reference)", 'matched model (reference)'),
        ("signal noise 0.45 (assumed 0.10)", 'signal noise 0.45 (model assumes 0.10)'),
        ("operators lapse 2× as often", 'operators lapse 2x as often'),
        ("GHLT uninformative, controller unaware", 'gaze lead uninformative, controller unaware'),
        ("GHLT uninformative, controller calibrated", 'gaze lead uninformative, controller calibrated'),
        ("W mirrored at 60 s, no drift in model", 'weighting flips at 60 s, no drift'),
        ("W mirrored at 60 s, drift 0.01 in model", 'weighting flips at 60 s, drift 0.010')]
ref = R['matched model (reference)']['diff']
for i, (lab, key) in enumerate(rows):
    r = R[key]; y = -i
    reliable = r['ci95'][0] > 0
    col = ACCENT if reliable else MUTED
    a.errorbar(r['diff'], y, xerr=[[r['diff'] - r['ci95'][0]], [r['ci95'][1] - r['diff']]], fmt='o', color=col, ms=4, capsize=2, lw=0.9,
               mfc=col if reliable else "white", mew=0.9)
a.axvline(0, color=INK, lw=0.7); a.axvline(ref, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
a.set_yticks([-i for i in range(len(rows))]); a.set_yticklabels([r[0] for r in rows]); a.tick_params(axis='y', length=0)
a.set_xlim(min(-0.3, min(R[k]['ci95'][0] for _, k in rows) - 0.05), max(R[k]['ci95'][1] for _, k in rows) + 0.08); a.set_ylim(-len(rows) + 0.4, 0.6); a.set_xlabel("UEE − MT, embodiment last 60 s")
for s in ("top", "right", "left"):
    a.spines[s].set_visible(False)
a.xaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
a.text(-0.95, 1.08, "A", transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
a.text(-0.87, 1.08, "Benefit under model–operator mismatch", transform=a.transAxes, fontsize=8, va="bottom")
a.text(-1.02, 1.1, "", transform=a.transAxes)

k = 5
sm = lambda v: (np.arange(len(v) - k + 1) + k // 2 + 1, np.convolve(v, np.ones(k) / k, mode='valid'))
for key, col, ls, lab in [('weighting flips at 60 s, no drift|mt', MT_C, '-', "MT"),
                          ('weighting flips at 60 s, no drift|uee', ACCENT, (0, (1, 1.2)), "UEE, no drift"),
                          ('weighting flips at 60 s, drift 0.010|uee', ACCENT, '-', "UEE, drift 0.01")]:
    x, y = sm(RT[key]); b.plot(x, y, color=col, ls=ls, lw=1.4 if 'uee' in key else 1.1)
b.axvline(60, color=INK, lw=0.6, ls=(0, (3, 2))); b.text(58.5, 2.22, "W mirrored", fontsize=6.5, color=INK, ha="right", va="top")
b.set_xlabel("time (s)"); b.set_ylabel("embodiment level (0–2)"); b.set_ylim(0, 2.3); b.set_yticks([0, 0.5, 1, 1.5, 2]); b.set_xlim(0, 120); b.set_xticks([0, 30, 60, 90, 120]); style(b)
b.text(45, 0.7, "MT", ha="center", va="top", fontsize=6.8, color=INK)
b.annotate("UEE, no drift", xy=(74, 0.16), xytext=(6, 0.2), ha="left", va="center", fontsize=6.8, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=2, shrinkB=2))
b.text(118, 2.08, "UEE, drift 0.01", ha="right", va="bottom", fontsize=6.8, color=INK)
label(b, "B", "Unmodelled change in cue weighting", x=-0.25)
save(fig, "fig3_robustness")

# ======================================================================= Figure 4: population sweep and out-of-family operators
G = json.load(open('results/generalization.json'))
fig, (a, b) = plt.subplots(1, 2, figsize=(W_CM * cm, 5.6 * cm), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.95, left=0.08, right=0.98, top=0.84, bottom=0.2))
sw = G['sweep']; x = [r['p_balanced'] for r in sw]
for key, fld in [('MT', 'MT'), ('Oracle', 'Oracle'), ('UEE', 'UEE')]:
    lab, col, ls, mk, lw = COND[key]
    a.plot(x, [r[fld] for r in sw], ls=ls, color=col, lw=lw, marker=mk, ms=4 if key != 'Oracle' else 4.5,
           mfc=col if key != 'Oracle' else "none", mec=col if key == 'Oracle' else "white", mew=0.5 if key != 'Oracle' else 0.9)
a.text(0.35, 1.93, "UEE ≈ Oracle", ha="center", va="bottom", fontsize=6.8, color=INK)
a.text(0.02, sw[0]['MT'] - 0.1, "MT", ha="left", va="top", fontsize=6.8, color=INK)
a.set_xlabel("fraction of balanced operators"); a.set_ylabel("embodiment level, last 60 s"); a.set_ylim(0, 2.1); a.set_xlim(-0.03, 1.03); style(a)
label(a, "A", "Benefit depends on population variety", x=-0.28)
oof = G['out_of_family']
names = ["five types, Gaussian (in family)", "continuous cue weighting", "triangular match", "match 1.6× wider", "continuous + triangular"]
for i, r in enumerate(oof):
    y = -i
    for fld, cfld, off, key in [('Oracle_minus_MT', 'Oracle_minus_MT_ci', 0.14, 'Oracle'), ('UEE_minus_MT', 'UEE_minus_MT_ci', -0.14, 'UEE')]:
        lab, col, ls, mk, lw = COND[key]
        v, (lo, hi) = r[fld], r[cfld]
        b.errorbar(v, y + off, xerr=[[v - lo], [hi - v]], fmt=mk, color=col, ms=4 if key == 'UEE' else 4.5, capsize=1.5, lw=0.9,
                   mfc=col if key == 'UEE' else "none", mec=col if key == 'Oracle' else "white", mew=0.5 if key == 'UEE' else 0.9)
b.axvline(0, color=INK, lw=0.7)
b.set_yticks([-i for i in range(len(oof))]); b.set_yticklabels(names); b.tick_params(axis='y', length=0)
b.set_ylim(-len(oof) + 0.45, 0.55); b.set_xlim(-0.05, 1.25); b.set_xlabel("gain over MT, embodiment last 60 s")
for sp in ("top", "right", "left"):
    b.spines[sp].set_visible(False)
b.xaxis.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True)
b.legend(handles=[Line2D([0], [0], color=ACCENT, marker='o', ls='none', ms=4, mec="white", label="UEE"),
                  Line2D([0], [0], color=ORACLE_C, marker='D', ls='none', ms=4.5, mfc="none", label="Oracle")],
         loc="lower right", frameon=False, handletextpad=0.3)
b.text(-0.78, 1.08, "B", transform=b.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
b.text(-0.70, 1.08, "Operators outside the model family", transform=b.transAxes, fontsize=8, va="bottom")
save(fig, "fig4_generalization")
print("figures written")
