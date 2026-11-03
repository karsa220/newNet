r"""ACSSR module figure for the DPRASR-Net paper.

Generates acssr.pdf (vector) -- drop it next to the .tex and
\includegraphics[width=\columnwidth]{acssr.pdf} picks it up.
Edit labels/colors here and re-run:  python acssr_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
})

SCALE_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
GRAY, ORANGE, DARK = "#555555", "#D95F02", "#333333"

fig, ax = plt.subplots(figsize=(7.2, 5.35))
ax.set_xlim(0, 11.0)
ax.set_ylim(0, 8.15)
ax.set_aspect("equal")
ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=1.4, alpha=1.0, z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.07",
        fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))


def txt(x, y, s, fs=10, color=DARK, **kw):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs,
            color=color, zorder=6, **kw)


def arrow(p, q, color=GRAY, lw=1.5, z=2, ms=12):
    ax.annotate("", xy=q, xytext=p, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0))


def seg(pts, color=GRAY, lw=1.4, z=2):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw,
            solid_capstyle="round", zorder=z)


# ------------------------------------------------------------------ geometry
rows = [4.5, 3.6, 2.7, 1.8, 0.9]          # F_1 (top) .. F_5 (bottom), box bottom y
FX, FW, FH = 0.32, 1.13, 0.60             # input feature-map boxes
OTX = 7.4                                  # x of the ⊗ nodes / w_k column
WXL, WW, WH = 7.04, 0.72, 0.38            # w_k boxes
COLX = 8.6                                 # collector vertical
SGX = 8.95                                 # Σ node
FOUT = (9.28, 2.60, 1.32, 0.80)           # F_out box

# ------------------------------------------------------------ input features
for i, y in enumerate(rows):
    rbox(FX, y, FW, FH, SCALE_COLORS[i], SCALE_COLORS[i], alpha=0.32)
    txt(FX + FW / 2, y + FH / 2, f"$F_{{{i+1}}}$", 13)
txt(FX + FW / 2, 0.50, "encoder scales ($K{=}5$)", 9)

# taps to Concat: left-side riser bus at x=0.18 (crosses nothing)
for y in rows:
    seg([(FX, y + FH / 2), (0.18, y + FH / 2), (0.18, 6.70)])
arrow((0.18, 6.70), (2.16, 6.70))

# ------------------------------------------------------------- top chain
rbox(2.20, 6.30, 1.40, 0.80, "white", DARK)
txt(2.90, 6.90, "Concat", 12.5, fontweight="bold")
txt(2.90, 6.53, r"$KC{\times}H{\times}W$", 9.5)
arrow((3.62, 6.70), (3.90, 6.70))

rbox(3.94, 6.20, 2.36, 1.00, "white", DARK)
txt(5.12, 6.95, "Bottleneck", 12.5, fontweight="bold")
txt(5.12, 6.62, r"$1{\times}1$ Conv $\downarrow r$ · BN · GELU", 9.5)
txt(5.12, 6.38, r"$1{\times}1$ Conv $\uparrow r$", 9.5)
arrow((6.32, 6.70), (6.68, 6.70))

rbox(6.70, 6.20, 3.80, 1.00, "white", DARK)
txt(8.60, 6.95, r"Softmax over scale axis $k$", 12, fontweight="bold")
txt(8.60, 6.64, r"scores reshaped to $K{\times}C{\times}H{\times}W$", 9)
txt(8.60, 6.40, "(per-pixel, per-channel)", 9, style="italic")

# ------------------------------------------- softmax output trunk into w stack
seg([(OTX, 6.20), (OTX, rows[4] + 0.62 + WH)], ORANGE, 1.7, z=2)

# ------------------------------------- w_k maps, ⊗ nodes, branches, collector
for i, y in enumerate(rows):
    # routing-weight map (same colour as its scale)
    rbox(WXL, y + 0.62, WW, WH, SCALE_COLORS[i], SCALE_COLORS[i], alpha=0.45)
    txt(OTX, y + 0.62 + WH / 2, f"$w_{{{i+1}}}$", 10.5)
    arrow((OTX, y + 0.62), (OTX, y + 0.46), ORANGE, 1.3, ms=9)

    # ⊗ node on the branch line
    ax.add_patch(Circle((OTX, y + 0.30), 0.115, fc="white", ec="black",
                        lw=1.2, zorder=4))
    txt(OTX, y + 0.293, r"$\otimes$", 11)

    # feature branch F_k -> ⊗
    arrow((FX + FW, y + FH / 2), (OTX - 0.13, y + FH / 2))

    # ⊗ -> collector -> Σ
    seg([(OTX + 0.115, y + 0.30), (COLX, y + 0.30), (COLX, 3.0)])

arrow((COLX, 3.0), (SGX - 0.17, 3.0))
ax.add_patch(Circle((SGX, 3.0), 0.165, fc="white", ec="black", lw=1.3, zorder=4))
txt(SGX, 2.99, r"$\Sigma$", 13)
txt(SGX, 2.64, "weighted sum", 8.5, style="italic")

# --------------------------------------------------------------------- output
arrow((SGX + 0.17, 3.0), (FOUT[0] - 0.02, 3.0))
rbox(*FOUT, "#E9E9E9", DARK)
txt(FOUT[0] + FOUT[2] / 2, 3.10, r"$F_{out}$", 13.5, fontweight="bold")
txt(FOUT[0] + FOUT[2] / 2, 2.80, r"$C{\times}H{\times}W$", 9.5)

# --------------------------------------------------------------------- legend
seg([(0.35, 7.80), (0.90, 7.80)], GRAY, 1.6)
ax.text(1.00, 7.80, "feature path", fontsize=9, va="center", ha="left", color=DARK)
seg([(0.35, 7.52), (0.90, 7.52)], ORANGE, 1.7)
ax.text(1.00, 7.52, "routing weights", fontsize=9, va="center", ha="left", color=DARK)

fig.savefig("/workspace/acssr.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("/workspace/acssr_preview.png", dpi=180, facecolor="white")
print("saved acssr.pdf / acssr_preview.png")
