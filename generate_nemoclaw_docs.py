"""
generate_nemoclaw_docs.py — generate a professional Word document explaining
loop_nemoclaw.sh, with embedded matplotlib diagrams.

Color discipline:
  - All text inside diagrams: BLACK
  - Box fills: light pastel only (low-saturation blue/green/yellow/grey)
  - Borders: dark navy
  - Arrows: dark grey
  - High-contrast everywhere, no cyan-on-dark, no thin colored text
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DOCX = os.path.join(HERE, "loop_nemoclaw_documentation.docx")
IMG_DIR  = os.path.join(HERE, "_doc_diagrams")
os.makedirs(IMG_DIR, exist_ok=True)

# ── High-contrast palette ─────────────────────────────────────────────────────
FILL_BLUE   = "#DCE7F4"   # primary boxes
FILL_GREEN  = "#DDEFE0"   # success / output
FILL_YELLOW = "#FBF1D2"   # decision / control
FILL_GREY   = "#EDEDED"   # storage / passive
FILL_RED    = "#F5DCDC"   # failure path
BORDER      = "#1F4E79"
BORDER_RED  = "#A33A3A"
ARROW       = "#404040"
TXT         = "#000000"
TITLE_CLR   = "#1F4E79"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   11,
    "axes.edgecolor": BORDER,
})


def box(ax, x, y, w, h, text, fill=FILL_BLUE, border=BORDER, fontsize=11,
        weight="normal", text_color=TXT, lw=1.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=lw, edgecolor=border, facecolor=fill, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, zorder=3,
            wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color=ARROW, style="->", lw=1.8,
          text_offset=(0.0, 0.15)):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                        linewidth=lw, mutation_scale=18, zorder=1)
    ax.add_patch(a)
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + text_offset[0], my + text_offset[1], text,
                ha="center", va="center", fontsize=9.5, color=TXT,
                style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="white"))


def setup_axes(ax, title, xlim=(0, 10), ylim=(0, 7.5)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=13.5, color=TITLE_CLR, weight="bold", pad=14)


# ── Diagram 1: System architecture (4-node cluster + LLM + GROMACS) ───────────
def diagram_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
    setup_axes(ax, "Fig 1.  System architecture — where nemoclaw runs",
               xlim=(0, 11), ylim=(0, 7.2))

    # Node boxes — top row 3 GPU compute, bottom row node 0 + LLM
    box(ax, 0.4, 5.4, 2.4, 1.2, "GPU 1 (node 1)\nGROMACS PME rank\nspark-0c01",
        fill=FILL_GREEN, weight="bold", fontsize=10)
    box(ax, 3.2, 5.4, 2.4, 1.2, "GPU 2 (node 2)\nGROMACS PP rank\nspark-1aa0",
        fill=FILL_GREEN, weight="bold", fontsize=10)
    box(ax, 6.0, 5.4, 2.4, 1.2, "GPU 3 (node 3)\nGROMACS PP rank\nspark-1b93",
        fill=FILL_GREEN, weight="bold", fontsize=10)

    box(ax, 0.4, 2.5, 2.4, 1.2,
        "GPU 0 (node 0)\nvLLM server\nNemotron-Super-120B",
        fill=FILL_BLUE, weight="bold", fontsize=10)

    # nemoclaw orchestrator
    box(ax, 3.2, 2.5, 5.2, 1.6,
        "loop_nemoclaw.sh  (the orchestrator)\nbash + paramiko + git on node 0",
        fill=FILL_YELLOW, weight="bold", fontsize=12, lw=2.0)

    # CX7 fabric label
    box(ax, 9.0, 4.0, 1.8, 1.0, "CX7\nRoCEv2 RDMA\nfabric",
        fill=FILL_GREY, fontsize=9, weight="bold")

    # Storage box
    box(ax, 0.4, 0.5, 2.4, 1.2, "results.tsv\nstructure.py\n(git history)",
        fill=FILL_GREY, fontsize=10, weight="bold")

    # Connections
    arrow(ax, 5.8, 4.1, 5.8, 5.4, text="mpirun + UCX over RDMA")
    arrow(ax, 5.8, 4.1, 1.6, 5.4, color=ARROW, lw=1.2)
    arrow(ax, 5.8, 4.1, 7.2, 5.4, color=ARROW, lw=1.2)
    arrow(ax, 3.2, 3.3, 2.8, 3.0, text="HTTP :8090", text_offset=(0.4, 0.2))
    arrow(ax, 2.0, 2.5, 1.6, 1.7, text="reads / writes")

    # Caption
    ax.text(5.5, 0.05,
            "GPU 0 hosts the LLM via vLLM and runs the bash orchestrator. "
            "GPUs 1–3 do the molecular dynamics. nemoclaw glues them together.",
            ha="center", va="bottom", fontsize=10, color=TXT, style="italic")

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "arch.png")
    plt.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close()
    return path


# ── Diagram 2: Per-iteration closed loop ──────────────────────────────────────
def diagram_loop():
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    setup_axes(ax, "Fig 2.  One iteration of the closed loop",
               xlim=(0, 11), ylim=(0, 8))

    # 7 steps arranged in a closed cycle
    nodes = [
        (1.0, 6.3, 2.6, 0.9, "1.  Read results.tsv\nBEST_R, count, history",          FILL_GREY),
        (4.2, 6.3, 2.6, 0.9, "2.  Reset structure.py\nto best-known commit",          FILL_BLUE),
        (7.4, 6.3, 2.6, 0.9, "3.  call_nemotron.py\n→ vLLM Nemotron-120B",            FILL_BLUE),
        (7.4, 4.3, 2.6, 0.9, "4.  Validate + write new\nstructure.py, git commit",   FILL_BLUE),
        (7.4, 2.3, 2.6, 0.9, "5.  prepare.py\n→ GROMACS on GPUs 1–3",                 FILL_GREEN),
        (4.2, 2.3, 2.6, 0.9, "6.  Parse R_th from run.log\nextract sim_time, atoms",  FILL_GREEN),
        (1.0, 2.3, 2.6, 0.9, "7.  Decision\nKEEP vs DISCARD vs CRASH",                FILL_YELLOW),
        (1.0, 4.3, 2.6, 0.9, "8.  Append row to\nresults.tsv (with status)",          FILL_GREY),
    ]
    for x, y, w, h, t, c in nodes:
        box(ax, x, y, w, h, t, fill=c, fontsize=10)

    # Arrows around the cycle
    arrow(ax, 3.6, 6.75, 4.2, 6.75)
    arrow(ax, 6.8, 6.75, 7.4, 6.75)
    arrow(ax, 8.7, 6.3, 8.7, 5.2)
    arrow(ax, 8.7, 4.3, 8.7, 3.2)
    arrow(ax, 7.4, 2.75, 6.8, 2.75)
    arrow(ax, 4.2, 2.75, 3.6, 2.75)
    arrow(ax, 2.3, 3.2, 2.3, 4.3)
    arrow(ax, 2.3, 5.2, 2.3, 6.3,
          text="loop back\n(next iter)", text_offset=(-1.0, 0.2))

    # Side legend for decision colors
    box(ax, 0.5, 0.4, 3.0, 0.7, "KEEP — leave commit, structure.py stays modified",
        fill=FILL_GREEN, fontsize=9.5, lw=1.0)
    box(ax, 4.0, 0.4, 3.0, 0.7, "DISCARD — git reset HEAD~1, revert",
        fill=FILL_YELLOW, fontsize=9.5, lw=1.0)
    box(ax, 7.5, 0.4, 3.0, 0.7, "CRASH — log, revert, continue",
        fill=FILL_RED, fontsize=9.5, lw=1.0, border=BORDER_RED)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "loop.png")
    plt.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close()
    return path


# ── Diagram 3: Call sequence / file flow ──────────────────────────────────────
def diagram_callflow():
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=200)
    setup_axes(ax, "Fig 3.  Call sequence and file flow per iteration",
               xlim=(0, 11), ylim=(0, 9))

    # 5 swim lanes for actors
    actors = [
        (1.0,  "loop_nemoclaw.sh\n(bash, node 0)",   FILL_YELLOW),
        (3.2,  "call_nemotron.py\n(python, node 0)", FILL_BLUE),
        (5.4,  "vLLM server\n(GPU 0)",                FILL_BLUE),
        (7.6,  "prepare.py\n(python, node 0)",       FILL_GREEN),
        (9.8,  "GROMACS mdrun\n(GPUs 1–3, MPI)",     FILL_GREEN),
    ]
    for cx, label, c in actors:
        box(ax, cx - 0.7, 8.0, 1.4, 0.8, label, fill=c, fontsize=9, weight="bold")
        # vertical lifeline
        ax.plot([cx, cx], [0.5, 8.0], color=BORDER, linewidth=0.7, linestyle="--",
                zorder=0)

    # Messages between actors (top to bottom is time)
    msgs = [
        (1.0, 3.2, 7.7, "exec  call_nemotron.py(best, tried)"),
        (3.2, 5.4, 7.0, "POST /v1/chat/completions  + history"),
        (5.4, 3.2, 6.3, "JSON reply with code block"),
        (3.2, 1.0, 5.6, "write structure.py · return exit 0"),
        (1.0, 1.0, 4.9, "py_compile · COMPOSITION sums to 1"),
        (1.0, 1.0, 4.2, "git add + commit"),
        (1.0, 7.6, 3.5, "exec  prepare.py"),
        (7.6, 9.8, 2.8, "mpirun em/prewarm/eq/prod (UCX RDMA)"),
        (9.8, 7.6, 2.1, "energy file with thermal_resistance"),
        (7.6, 1.0, 1.4, "stdout: 'thermal_resistance: 3.66e-6'"),
        (1.0, 1.0, 0.7, "parse R_th · KEEP or DISCARD · append results.tsv"),
    ]
    for x1, x2, y, label in msgs:
        if x1 == x2:
            # self call — small rectangle on the lifeline
            ax.annotate("", xy=(x1+0.2, y-0.05), xytext=(x1+0.2, y+0.05),
                        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.4))
            ax.text(x1 + 0.4, y, label, ha="left", va="center",
                    fontsize=9, color=TXT,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="white"))
        else:
            d = "right" if x2 > x1 else "left"
            arrow(ax, x1, y, x2, y, color=ARROW, lw=1.3)
            mx = (x1 + x2) / 2
            ax.text(mx, y + 0.18, label, ha="center", va="bottom",
                    fontsize=9, color=TXT,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="white"))

    ax.text(5.5, 0.1,
            "Time flows downward.  Each horizontal arrow is a process invocation, HTTP call, or MPI dispatch.",
            ha="center", va="bottom", fontsize=9.5, style="italic", color=TXT)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "callflow.png")
    plt.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close()
    return path


# ── Diagram 4: Security boundaries (openshell model) ──────────────────────────
def diagram_security():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
    setup_axes(ax, "Fig 4.  Security boundaries — open-shell isolation model",
               xlim=(0, 11), ylim=(0, 7))

    # Outer trust boundary: the head node
    outer = Rectangle((0.4, 0.4), 10.2, 6.0, fill=False,
                      linewidth=2.0, edgecolor=BORDER, linestyle="--", zorder=1)
    ax.add_patch(outer)
    ax.text(5.5, 6.2, "TRUST BOUNDARY — single Linux user account on node 0",
            ha="center", va="top", fontsize=11, color=TITLE_CLR, weight="bold")

    # Inside boxes — security controls
    box(ax, 0.9, 4.4, 2.6, 1.2,
        "1. Local-only vLLM\nNo API keys, no\negress traffic",
        fill=FILL_BLUE, fontsize=10, weight="bold")
    box(ax, 4.0, 4.4, 2.6, 1.2,
        "2. Validation before exec\npy_compile + COMPOSITION\nsum check",
        fill=FILL_BLUE, fontsize=10, weight="bold")
    box(ax, 7.1, 4.4, 2.6, 1.2,
        "3. Git audit trail\nEvery iteration is a\nseparately-revertable commit",
        fill=FILL_BLUE, fontsize=10, weight="bold")

    box(ax, 0.9, 2.6, 2.6, 1.2,
        "4. File-system bounded\nAll writes stay inside\nthe project directory",
        fill=FILL_BLUE, fontsize=10, weight="bold")
    box(ax, 4.0, 2.6, 2.6, 1.2,
        "5. Subprocess isolation\nLLM-emitted code runs\nas plain python3 module",
        fill=FILL_BLUE, fontsize=10, weight="bold")
    box(ax, 7.1, 2.6, 2.6, 1.2,
        "6. Read-only history\nresults.tsv is append-only\nfor the loop",
        fill=FILL_BLUE, fontsize=10, weight="bold")

    box(ax, 2.6, 0.9, 5.8, 1.2,
        "7.  Crash containment\nA bad LLM proposal that breaks GROMACS is logged as 'crash',\nreverted via git reset, and the loop continues",
        fill=FILL_YELLOW, fontsize=10, weight="bold")

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "security.png")
    plt.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close()
    return path


# ── Build Word doc ─────────────────────────────────────────────────────────────
def style_heading(p, size=18, color=TITLE_CLR, bold=True):
    for run in p.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = RGBColor.from_string(color.lstrip("#"))


def add_h1(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    p.style = doc.styles["Heading 1"]
    r.font.size = Pt(20); r.font.bold = True
    r.font.color.rgb = RGBColor.from_string(TITLE_CLR.lstrip("#"))


def add_h2(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    p.style = doc.styles["Heading 2"]
    r.font.size = Pt(15); r.font.bold = True
    r.font.color.rgb = RGBColor.from_string(TITLE_CLR.lstrip("#"))


def add_body(doc, text, size=11):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size); r.font.name = "Calibri"
    r.font.color.rgb = RGBColor(0, 0, 0)


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.size = Pt(11); r.font.name = "Calibri"
    r.font.color.rgb = RGBColor(0, 0, 0)


def add_code(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"; r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(20, 20, 20)
    pf = p.paragraph_format
    pf.left_indent = Inches(0.2)


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(10); r.font.italic = True
    r.font.color.rgb = RGBColor.from_string("404040")


def build_doc():
    arch_path     = diagram_architecture()
    loop_path     = diagram_loop()
    cf_path       = diagram_callflow()
    sec_path      = diagram_security()

    doc = Document()
    section = doc.sections[0]
    section.left_margin = Inches(0.8); section.right_margin = Inches(0.8)
    section.top_margin  = Inches(0.7); section.bottom_margin = Inches(0.7)

    # Title
    p = doc.add_paragraph()
    r = p.add_run("loop_nemoclaw.sh")
    r.font.size = Pt(26); r.font.bold = True
    r.font.color.rgb = RGBColor.from_string(TITLE_CLR.lstrip("#"))
    p = doc.add_paragraph()
    r = p.add_run("The autonomous research orchestrator behind the AutoTherm demo")
    r.font.size = Pt(13); r.font.italic = True
    r.font.color.rgb = RGBColor.from_string("606060")

    # ─── 1. Overview ───────────────────────────────────────────────────────────
    add_h1(doc, "1.  Overview")
    add_body(doc,
        "loop_nemoclaw.sh is a 400-line bash script running on the head node of a "
        "4-node DGX Spark cluster. It coordinates an LLM (Nemotron-Super-120B "
        "served by vLLM) and a GROMACS molecular-dynamics pipeline into a "
        "closed-loop autonomous research agent that searches for thermal "
        "interface material compositions with lower thermal resistance.")
    add_body(doc,
        "It is an example of the open-shell design pattern: every step is a plain "
        "Unix process launched by bash, every artifact is a regular file, and "
        "every state change is captured by git. There is no opaque orchestration "
        "framework — anyone can read the script and understand exactly what runs.")

    # ─── 2. System architecture ───────────────────────────────────────────────
    add_h1(doc, "2.  System architecture")
    add_body(doc,
        "The cluster has four DGX Spark nodes connected by a ConnectX-7 RDMA "
        "fabric. nemoclaw runs on node 0 (the head node) where it has local "
        "access to the LLM, the project files, and a paramiko/ssh connection "
        "to the three GPU compute nodes.")
    doc.add_picture(arch_path, width=Inches(7.0))
    add_caption(doc, "Figure 1.  Physical layout. nemoclaw sits on node 0; "
                     "GROMACS mdrun is dispatched to GPUs 1–3.")

    # ─── 3. The closed loop ───────────────────────────────────────────────────
    add_h1(doc, "3.  The closed loop")
    add_body(doc,
        "Every iteration follows the same 8-step cycle. The loop runs for a "
        "configurable number of iterations (100 by default) or until the user "
        "signals stop.")
    doc.add_picture(loop_path, width=Inches(7.0))
    add_caption(doc, "Figure 2.  One iteration. Steps 1–4 prepare a new design, "
                     "step 5 simulates it, steps 6–8 score and record it.")

    add_h2(doc, "What each step actually does")
    add_bullet(doc, "1. Read results.tsv: parse every prior keep / discard / crash "
                    "row to compute the running best R_th and the experiments-tried count.")
    add_bullet(doc, "2. Reset structure.py: git-checkout the structure.py from the best-known "
                    "commit so the LLM always starts from the best design, not from a "
                    "discarded detour.")
    add_bullet(doc, "3. Call the LLM: call_nemotron.py builds a system + user prompt "
                    "(history, constraints, current best) and POSTs to the local vLLM "
                    "endpoint on port 8090.")
    add_bullet(doc, "4. Validate and commit: the LLM's Python code block is parsed, "
                    "py_compile is run, COMPOSITION values are checked to sum to 1.0, "
                    "then the new structure.py is written and git-committed.")
    add_bullet(doc, "5. Simulate: prepare.py orchestrates grompp + mpirun for em, "
                    "prewarm, equilibration and production NVT on 3 cluster GPUs over "
                    "UCX RDMA.")
    add_bullet(doc, "6. Extract R_th: prepare.py parses energy outputs and emits a "
                    "single thermal_resistance value plus a sim_time stamp.")
    add_bullet(doc, "7. Decide: if R_th < best, status=keep; if worse, status=discard "
                    "and the iteration commit is reverted via git reset.")
    add_bullet(doc, "8. Append: a tab-separated row is appended to results.tsv. The "
                    "next iteration immediately sees this history.")

    # ─── 4. Call sequence ─────────────────────────────────────────────────────
    add_h1(doc, "4.  Call sequence and file flow")
    add_body(doc,
        "The sequence below shows how control and data move between the five "
        "actors during a single iteration. Time flows downward.")
    doc.add_picture(cf_path, width=Inches(7.0))
    add_caption(doc, "Figure 3.  Call sequence. Vertical dashed lines are lifelines; "
                     "horizontal arrows are process calls or HTTP/MPI dispatches.")

    add_h2(doc, "Key files exchanged")
    add_bullet(doc, "structure.py  — the design variables the LLM rewrites. 28 lines.")
    add_bullet(doc, "results.tsv   — append-only history of every iteration "
                    "(commit, R_th, sim_time, status, description).")
    add_bullet(doc, "em.tpr / prod.tpr / eq.tpr — GROMACS binary inputs built by prepare.py.")
    add_bullet(doc, "run.log       — stdout of prepare.py for the current iteration "
                    "(parsed for thermal_resistance).")

    # ─── 5. Security model ────────────────────────────────────────────────────
    add_h1(doc, "5.  Security — how the open-shell model isolates risk")
    add_body(doc,
        "The script accepts code generated by an LLM and executes it as part of "
        "the simulation pipeline. That sounds dangerous; in practice, the "
        "open-shell design uses six independent layers of isolation, each one "
        "of which is easy to inspect because the whole pipeline is just files "
        "and processes.")
    doc.add_picture(sec_path, width=Inches(7.0))
    add_caption(doc, "Figure 4.  Six independent security controls plus crash "
                     "containment make the LLM-in-the-loop safe.")

    add_h2(doc, "Why \"open shell\" is the safe choice here")
    add_bullet(doc, "Local-only vLLM. The model runs on the same node as the loop; "
                    "no API keys, no external endpoints, no egress traffic that could "
                    "leak proprietary research data.")
    add_bullet(doc, "Validation before exec. LLM output is structurally parsed and "
                    "syntactically compiled before any execution; semantic checks "
                    "ensure composition fractions sum to 1.0 and stay within bounds.")
    add_bullet(doc, "Git audit trail. Every iteration is its own commit; bad designs "
                    "are reverted via git reset; the entire research history is "
                    "auditable with git log.")
    add_bullet(doc, "File-system bounded. The script only writes inside the project "
                    "directory; GROMACS scratch is in /tmp on the compute nodes.")
    add_bullet(doc, "Subprocess isolation. LLM-emitted code is imported as a plain "
                    "Python module by prepare.py; no shell injection, no arbitrary "
                    "command substitution.")
    add_bullet(doc, "Read-only history during planning. The LLM never writes to "
                    "results.tsv directly — only the orchestrator appends rows after "
                    "ground-truth simulation, so the model cannot fake its own "
                    "track record.")
    add_bullet(doc, "Crash containment. If an LLM proposal causes mdrun to segfault "
                    "or em to diverge, the iteration is marked 'crash', git-reverted, "
                    "and the loop continues from the last known-good design.")

    # ─── 6. Robustness ────────────────────────────────────────────────────────
    add_h1(doc, "6.  Robustness and failure handling")
    add_body(doc,
        "Out of 100 iterations on a 500k-atom TIM benchmark, the loop survived "
        "roughly 40 different LLM proposals, 6 simulation crashes, 2 vLLM "
        "timeouts and 1 GROMACS GPU memory event — all without operator "
        "intervention. The mechanisms that make this possible:")
    add_bullet(doc, "vLLM retry: call_nemotron.py retries up to 3 times with a 300-second "
                    "per-call timeout, then surfaces a clean exit code so the loop "
                    "can skip the iteration.")
    add_bullet(doc, "GROMACS crash trap: prepare.py wraps each mdrun in a subprocess "
                    "with a timeout; non-zero exit triggers a 'crash' row in results.tsv "
                    "without losing prior best.")
    add_bullet(doc, "Git revert on bad design: discard always uses git reset HEAD~1 "
                    "so the working tree returns to the last keep, never a half-applied "
                    "edit.")
    add_bullet(doc, "Validation locks: sed-based invariant locks in the loop ensure "
                    "fields like N_TOTAL and SIM_TIME_PS cannot be widened by the LLM "
                    "(the operator sets the simulation budget, not the model).")

    # ─── 7. Reference card ────────────────────────────────────────────────────
    add_h1(doc, "7.  Operator reference")
    add_h2(doc, "Files this loop owns")
    add_code(doc, "loop_nemoclaw.sh   — the bash orchestrator (this document)\n"
                  "call_nemotron.py   — LLM HTTP client + prompt builder\n"
                  "prepare.py         — GROMACS pipeline + R_th extractor\n"
                  "structure.py       — composition / LJ params (LLM-mutated)\n"
                  "results.tsv        — append-only history of all iterations")

    add_h2(doc, "Starting and stopping the loop")
    add_code(doc, "# Start (run 100 iterations):\n"
                  "$ tmux new -s autotherm 'bash loop_nemoclaw.sh 100'\n\n"
                  "# Stop early (graceful):\n"
                  "$ tmux send-keys -t autotherm C-c Enter\n\n"
                  "# Inspect progress:\n"
                  "$ tail -f results.tsv\n"
                  "$ git log --oneline -20")

    add_h2(doc, "Reading results.tsv")
    add_code(doc, "commit\tR_th\tsim_time_s\tstatus\tdescription\n"
                  "0469a0c\t3.66e-06\t782.8\tdiscard\titer-36: ...\n"
                  "8498642\t3.26e-06\t633.3\tkeep\titer-27: ...    ← current best\n")

    doc.save(OUT_DOCX)
    print(f"Saved: {OUT_DOCX}")
    print(f"Diagrams in: {IMG_DIR}")


if __name__ == "__main__":
    build_doc()
