#!/bin/bash
# ============================================================
# loop_nemoclaw.sh — Autonomous TIM Research Loop using NemoClaw
#
# Runs entirely on node 0. GROMACS is distributed automatically
# by prepare.py to nodes 1-3 via mpirun over the CX7/mlx5
# InfiniBand fabric (N_RANKS=3, hostfile_gromacs lists nodes 1-3).
# Node 0 GPU is reserved for vLLM; nodes 1-3 GPUs for GROMACS.
#
# Security model (vs bare OpenCode in loop.sh):
#   - LLM agent runs inside NemoClaw sandbox (netns + Landlock + seccomp)
#   - Agent only sees sandbox workspace — cannot access host filesystem
#   - Agent cannot make network calls (netns blocks all non-proxy egress)
#   - Prompt base64-encoded over SSH to prevent shell injection
#   - TIM composition data never leaves node 0 host
#
# How it works each iteration:
#   1. Upload structure.py (baseline), results.tsv, program.md into sandbox
#   2. NemoClaw agent reads files, writes new structure.py in sandbox
#   3. Download structure.py from sandbox to host
#   4. Host validates, commits, runs prepare.py
#   5. prepare.py dispatches GROMACS MPI job to nodes 1-3 via CX7
#   6. Host collects result, updates results.tsv
#
# Prerequisites on node 0:
#   - NemoClaw installed; sandbox 'autotherm-agent' created and running
#     (NEMOCLAW_ENDPOINT_URL=http://172.17.0.1:8090/v1)
#   - vLLM serving Nemotron on port 8090 (port 8080 = OpenShell gateway)
#   - openshell gateway 'nemoclaw' configured
#   - SSH passwordless access from node 0 to nodes 1-3 (prepare.py uses SCP)
#   - hostfile_gromacs present: lists nodes 1-3 only
#
# Usage:
#   ./loop_nemoclaw.sh [max_iterations]
# ============================================================

REPO_DIR="/home/nvidia/autotherm"
PYTHON="python3"
NEMOCLAW="/home/nvidia/.local/bin/nemoclaw"
OPENSHELL="/home/nvidia/.local/bin/openshell"
VLLM_PORT=8090
MODEL="vllm//home/nvidia/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
MAX_ITER="${1:-100}"

GMX="/usr/local/gromacs/bin/gmx_mpi"

# ── NemoClaw sandbox config ───────────────────────────────────────────────────
SANDBOX_NAME="autotherm-agent"
SANDBOX_WORK="/sandbox/.openclaw/workspace"
SSH_PROXY="${OPENSHELL} ssh-proxy --gateway-name nemoclaw --name ${SANDBOX_NAME}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
OPENCLAW_ENV="env HOME=/sandbox XDG_CONFIG_HOME=/tmp/.config"
OPENCLAW="/usr/local/bin/openclaw"

sbox()        { ssh ${SSH_OPTS} -o "ProxyCommand=${SSH_PROXY}" sandbox "$@"; }
sbox_upload() { ssh ${SSH_OPTS} -o "ProxyCommand=${SSH_PROXY}" sandbox "cat > $1"; }

cd "$REPO_DIR"

log() { echo "[$(date '+%H:%M:%S')] [node-0] $*"; }

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  MODE : SINGLE ORCHESTRATOR (node 0)                ║"
echo "║  LLM  : Nemotron-3-Super-120B  via  NemoClaw        ║"
echo "║  SIM  : GROMACS MPI on nodes 1-3 via CX7/mlx5      ║"
echo "║  SEC  : NemoClaw sandbox (netns + Landlock + seccomp)║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────────────
[ ! -f "$NEMOCLAW" ] && { log "ERROR: nemoclaw not found at $NEMOCLAW"; exit 1; }
[ ! -f "$OPENSHELL" ] && { log "ERROR: openshell not found at $OPENSHELL"; exit 1; }
[ ! -f "$GMX" ]       && { log "ERROR: gmx_mpi not found at $GMX"; exit 1; }
[ ! -f "$REPO_DIR/hostfile_gromacs" ] && { log "ERROR: hostfile_gromacs not found (nodes 1-3 list)"; exit 1; }

log "NemoClaw : $NEMOCLAW"
log "OpenShell: $OPENSHELL"
log "Model    : $MODEL"
log "GROMACS  : $GMX  ($($GMX --version 2>&1 | grep 'GROMACS version' | head -1))"

# ── Verify NemoClaw sandbox is accessible ─────────────────────────────────────
log "Checking NemoClaw sandbox '${SANDBOX_NAME}'..."
if ! sbox "${OPENCLAW_ENV} ${OPENCLAW} gateway health" > /dev/null 2>&1; then
    log "ERROR: NemoClaw gateway not responding."
    log "  Fix: nemoclaw ${SANDBOX_NAME} recover"
    log "  Then: openshell inference update --gateway-name nemoclaw --timeout 600"
    exit 1
fi
log "NemoClaw sandbox OK."

# ── Ensure vLLM on port 8090 (8080 reserved for OpenShell gateway) ────────────
if curl -sf http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
    log "vLLM already ready on :${VLLM_PORT}."
else
    log "vLLM not on :${VLLM_PORT} — starting Nemotron..."
    pkill -9 -f "vllm serve" 2>/dev/null; sleep 2
    /home/nvidia/vllm-env/bin/vllm serve \
        /home/nvidia/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
        --port ${VLLM_PORT} \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.85 \
        --compilation-config '{"mode": 0}' \
        >> /tmp/vllm_nemotron.log 2>&1 &
    log "vLLM started (PID $!) — waiting up to 35 min..."
    for i in $(seq 1 420); do
        curl -sf http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1 \
            && { log "vLLM ready."; break; }
        [ $((i % 12)) -eq 0 ] && log "  still waiting... ${i}/420 ($(( i*5/60 ))m elapsed)"
        sleep 5
    done
    curl -sf http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1 \
        || { log "ERROR: vLLM not ready after 35 min. Abort."; exit 1; }
fi

# ── Save baseline ──────────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/structure.py.baseline" ]; then
    cp "$REPO_DIR/structure.py" "$REPO_DIR/structure.py.baseline"
    log "Saved structure.py.baseline"
fi

# ── Main loop (runs entirely on node 0) ───────────────────────────────────────
for iter in $(seq 1 "$MAX_ITER"); do
    log "══════ Iteration $iter / $MAX_ITER ══════"

    cp "$REPO_DIR/structure.py.baseline" structure.py
    log "Reset structure.py to baseline."

    BEST_R=$(awk -F'\t' '$4 == "keep" {print $2}' results.tsv 2>/dev/null \
             | sort -g | head -1)
    BEST_R="${BEST_R:-2.28e-5}"
    TRIED=$(awk -F'\t' '$4 == "keep" || $4 == "discard"' results.tsv 2>/dev/null | wc -l || echo 0)
    log "Best thermal_resistance=$BEST_R  |  Total experiments=$TRIED"

    # ── Upload into NemoClaw sandbox workspace ────────────────────────────────
    log "Uploading files to sandbox..."
    sbox_upload "${SANDBOX_WORK}/structure.py" < structure.py \
        || { log "ERROR: upload structure.py failed"; continue; }
    sbox_upload "${SANDBOX_WORK}/results.tsv"  < results.tsv \
        || { log "ERROR: upload results.tsv failed"; continue; }
    sbox_upload "${SANDBOX_WORK}/program.md"   < "$REPO_DIR/program.md" \
        || { log "ERROR: upload program.md failed"; continue; }

    # ── Build prompt (sandbox paths only — agent cannot reference host paths) ─
    PROMPT="You are an AI research agent on a single NVIDIA DGX Spark.

You are a Computational Materials Scientist optimizing a Thermal Interface Material (TIM).
Current best thermal_resistance = ${BEST_R} m²K/W. LOWER is BETTER.

Steps you MUST follow:
1. Read ${SANDBOX_WORK}/structure.py
2. Read ${SANDBOX_WORK}/results.tsv
3. Read ${SANDBOX_WORK}/program.md
4. Choose ONE specific change to structure.py that has not been tried yet.
5. Write the complete new structure.py using the bash tool:
   cat > ${SANDBOX_WORK}/structure.py << 'PYEOF'
   [complete new python file]
   PYEOF
6. Stop. Do NOT run simulations. Do NOT modify any file other than ${SANDBOX_WORK}/structure.py.

IMPORTANT: COMPOSITION values must sum to exactly 1.0."

    # ── Run NemoClaw agent (base64-encoded to prevent shell injection) ─────────
    log "Running NemoClaw agent (iter $iter)..."
    PROMPT_B64=$(printf '%s' "$PROMPT" | base64 -w0)
    SESSION_ID="autotherm-iter-${iter}"

    AGENT_OUT=$(sbox "printf '%s' '${PROMPT_B64}' | base64 -d > /tmp/prompt.txt && \
        ${OPENCLAW_ENV} ${OPENCLAW} agent --agent main \
            --session-id '${SESSION_ID}' \
            --message \"\$(cat /tmp/prompt.txt)\" \
            --json --timeout 600" 2>&1)
    AGENT_EXIT=$?

    if [ $AGENT_EXIT -ne 0 ]; then
        log "WARNING: NemoClaw agent exited $AGENT_EXIT — skipping."
        log "Last output: $(echo "$AGENT_OUT" | grep -v '^$' | tail -3)"
        cp "$REPO_DIR/structure.py.baseline" structure.py
        continue
    fi

    # ── Download modified structure.py from sandbox ───────────────────────────
    log "Downloading structure.py from sandbox..."
    sbox "cat ${SANDBOX_WORK}/structure.py" > structure.py \
        || { log "ERROR: download failed"; cp "$REPO_DIR/structure.py.baseline" structure.py; continue; }

    # ── Validate ──────────────────────────────────────────────────────────────
    if ! $PYTHON -m py_compile structure.py 2>/tmp/syntax_err.txt; then
        log "WARNING: syntax error — skipping. $(cat /tmp/syntax_err.txt)"
        cp "$REPO_DIR/structure.py.baseline" structure.py; continue
    fi

    if git diff --quiet structure.py; then
        log "WARNING: structure.py unchanged — skipping."
        continue
    fi

    $PYTHON -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('s', 'structure.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
total = sum(s.COMPOSITION.values())
if abs(total - 1.0) > 1e-6:
    print(f'INVALID: composition sums to {total:.6f}'); sys.exit(1)
print('OK')
" 2>/tmp/val_err.txt
    if [ $? -ne 0 ]; then
        log "WARNING: validation failed: $(cat /tmp/val_err.txt)"
        cp "$REPO_DIR/structure.py.baseline" structure.py; continue
    fi

    # ── Commit ────────────────────────────────────────────────────────────────
    HEADLINE=$(head -10 structure.py | grep -i '#\|"""\|COMPOSITION' | head -1 \
               | sed 's/[#"*]*//g' | xargs)
    git add structure.py
    git commit -m "iter-${iter}: ${HEADLINE:-TIM optimization}" \
        || { log "Commit failed — skipping."; continue; }
    COMMIT=$(git rev-parse --short HEAD)
    log "Committed $COMMIT"

    # ── GROMACS simulation (prepare.py dispatches to nodes 1-3 via mpirun/CX7) ─
    log "Running GROMACS on nodes 1-3 via CX7/mlx5 (iter $iter)..."
    $PYTHON prepare.py 2>&1 | tee run.log
    SIM_EXIT=${PIPESTATUS[0]}
    log "Simulation complete (exit=$SIM_EXIT)."

    if [ "$SIM_EXIT" -ne 0 ]; then
        CRASH_MSG=$(grep -m1 "Error\|INVALID\|ERROR\|Traceback" run.log 2>/dev/null || echo "unknown")
        log "WARNING: Simulation CRASHED — $CRASH_MSG"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$COMMIT" "N/A" "N/A" "crash" "iter-${iter}: $CRASH_MSG" >> results.tsv
        git reset HEAD~1 2>/dev/null || true
        cp "$REPO_DIR/structure.py.baseline" structure.py; continue
    fi

    THERMAL_R=$(grep -o 'thermal_resistance:[[:space:]]*[-0-9.eE+]*' run.log \
                | sed 's/thermal_resistance:[[:space:]]*//' | head -1)
    SIM_TIME=$(grep -o 'sim_time_s:[[:space:]]*[0-9.]*' run.log \
               | sed 's/sim_time_s:[[:space:]]*//' | head -1)
    NANOTUBE_GEOM=$(grep -m1 '^nanotube_geometry: ' run.log \
                    | sed 's/^nanotube_geometry: //' || true)

    if [ -z "$THERMAL_R" ]; then
        log "WARNING: thermal_resistance not found in run.log — skipping."
        log "Last 5 lines: $(tail -5 run.log | tr '\n' '|')"
        git reset HEAD~1 2>/dev/null || true
        cp "$REPO_DIR/structure.py.baseline" structure.py; continue
    fi

    log "thermal_resistance=$THERMAL_R  best=$BEST_R  sim_time=${SIM_TIME}s"

    if [ -n "$NANOTUBE_GEOM" ]; then
        printf '%s\t%s\n' "$iter" "$NANOTUBE_GEOM" >> "$REPO_DIR/nanotube_geometry_history.tsv"
    fi

    BETTER=$($PYTHON -c "
try:    print('yes' if float('${THERMAL_R}') < float('${BEST_R}') else 'no')
except: print('no')
")

    if [ "$BETTER" = "yes" ]; then
        STATUS="keep"
        log "NEW BEST — keeping."
        if [ -n "$NANOTUBE_GEOM" ]; then
            echo "$NANOTUBE_GEOM" > "$REPO_DIR/nanotube_geometry_best.json"
        fi
    else
        STATUS="discard"
        log "No improvement — reverting."
        git reset HEAD~1
        cp "$REPO_DIR/structure.py.baseline" structure.py
    fi

    if [ ! -f "$REPO_DIR/nanotube_geometry_baseline.json" ] && [ -n "$NANOTUBE_GEOM" ]; then
        echo "$NANOTUBE_GEOM" > "$REPO_DIR/nanotube_geometry_baseline.json"
        log "Saved baseline nanotube geometry."
    fi

    DESC="iter-${iter}: ${HEADLINE:-TIM optimization}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$COMMIT" "${THERMAL_R}" "${SIM_TIME:-?}" "$STATUS" "$DESC" >> results.tsv

    log "Iteration $iter complete."
    echo ""
done

log "Loop complete — $MAX_ITER iterations done."

# ── Manufacturing Report ───────────────────────────────────────────────────────
log "══════ Generating Manufacturing Formulation Report ══════"

BEST_COMMIT=$(awk -F'\t' '$4 == "keep" {print $1, $2}' results.tsv 2>/dev/null \
              | sort -k2 -g | head -1 | awk '{print $1}')
BEST_R=$(awk -F'\t' '$4 == "keep" {print $2}' results.tsv 2>/dev/null | sort -g | head -1)

if [ -z "$BEST_COMMIT" ]; then
    log "No successful iterations — using baseline."; BEST_COMMIT="HEAD"; BEST_R="N/A"
fi

log "Best commit: $BEST_COMMIT  R_th=$BEST_R m²K/W"

BEST_STRUCT="/tmp/best_structure.py"
git show "${BEST_COMMIT}:structure.py" > "$BEST_STRUCT" 2>/dev/null || cp structure.py "$BEST_STRUCT"

REPORT_FILE="$REPO_DIR/manufacturing_report.txt"
REPORT_DATE=$(date '+%Y-%m-%d %H:%M:%S')

REPORT_PROMPT="You are a materials scientist preparing a manufacturing specification for a Thermal Interface Material (TIM).

The AI research loop has completed ${MAX_ITER} GROMACS molecular dynamics simulations to find the optimal TIM formulation.

Best result: thermal_resistance = ${BEST_R} m²K/W (lower is better; target for GPU TIMs: < 1e-5 m²K/W).
All experiment history is in ${SANDBOX_WORK}/results.tsv.

The winning formulation parameters are in ${SANDBOX_WORK}/structure.py — read that file first.

Generate a professional manufacturing specification report with these sections:

1. EXECUTIVE SUMMARY
   - Best thermal resistance achieved
   - Comparison to commercial TIMs (typical range: 1e-8 to 1e-6 m²K/W for indium sheets; 1e-6 to 1e-5 for phase-change TIMs)
   - Go/No-Go recommendation for prototype manufacturing

2. OPTIMAL FORMULATION
   - Composition by weight percent (derive from COMPOSITION fractions and LJ_PARAMS masses)
   - Composition by volume percent (derive from COMPOSITION fractions and sigma³ as proxy for molecular volume)
   - For each component: material identity (metal=InGa eutectic, filler=graphene/h-BN nanoflakes, binder=polysiloxane), supplier grade recommendations, purity requirements

3. INTERFACE CHEMISTRY
   - Cross-interaction analysis: which component pair has the strongest LJ coupling (lowest epsilon = weakest link in thermal chain)
   - Surface treatment recommendations to enhance weak interfaces
   - Wettability notes for InGa on graphene vs silicon vs copper surfaces

4. PROCESSING INSTRUCTIONS
   - Mixing sequence (add binder first, disperse filler, incorporate metal last — or as LJ params suggest)
   - Temperature during mixing (use TEMPERATURE parameter from structure.py as processing temp proxy)
   - Layer thickness target (derive from BOX_L / sqrt(N_TOTAL) as single-layer thickness estimate, in micrometers)
   - Curing / application method: stencil print vs syringe dispense vs pre-form

5. PREDICTED PERFORMANCE ENVELOPE
   - Thermal resistance at operating temperature (${BEST_R} m²K/W achieved)
   - Estimated conductivity range (λ = L/R)
   - Sensitivity: which parameter has the most impact on performance (based on results.tsv spread)

6. SCALE-UP NOTES
   - Batch size feasibility
   - QC test: 4-point thermal impedance measurement method
   - Shelf life estimate for InGa-based TIMs

Write in professional engineering report style. Be specific and quantitative wherever the simulation data supports it.
Output the report as plain text to stdout."

log "Uploading best structure.py and results.tsv to sandbox for report..."
sbox_upload "${SANDBOX_WORK}/structure.py" < "$BEST_STRUCT"
sbox_upload "${SANDBOX_WORK}/results.tsv"  < "$REPO_DIR/results.tsv"

PROMPT_B64=$(printf '%s' "$REPORT_PROMPT" | base64 -w0)
REPORT_OUT=$(sbox "printf '%s' '${PROMPT_B64}' | base64 -d > /tmp/report_prompt.txt && \
    ${OPENCLAW_ENV} ${OPENCLAW} agent --agent main \
        --session-id 'autotherm-report' \
        --message \"\$(cat /tmp/report_prompt.txt)\" \
        --json --timeout 600" 2>&1)

if [ $? -eq 0 ] && [ -n "$REPORT_OUT" ]; then
    echo "$REPORT_OUT" | tee "$REPORT_FILE"
else
    log "WARNING: NemoClaw report failed — writing fallback report."
    {
        echo "THERMAL INTERFACE MATERIAL — MANUFACTURING FORMULATION REPORT"
        echo "Generated: $REPORT_DATE"
        echo "Best formulation — commit: $BEST_COMMIT  R_th: $BEST_R m²K/W"
        echo ""
        echo "=== structure.py parameters ==="
        cat "$BEST_STRUCT"
        echo ""
        echo "=== SCALE-UP GUIDANCE ==="
        echo "Material identities:"
        echo "  metal  → InGa eutectic alloy (In₇₅Ga₂₅ by at%)  — order from Indium Corporation"
        echo "  filler → Graphene / h-BN nanoflakes (lateral size 1-5 µm, thickness 2-10 nm)"
        echo "  binder → Polydimethylsiloxane (PDMS) or phenyl-modified polysiloxane, 50-100 cP"
        echo ""
        $PYTHON - <<PYEOF
import importlib.util, math
spec = importlib.util.spec_from_file_location('s', '$BEST_STRUCT')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
comp = s.COMPOSITION
ljp  = s.LJ_PARAMS
total_mass = sum(comp[c] * ljp[c]['mass'] for c in comp)
print("Weight fractions (approx):")
for c in comp:
    wf = comp[c] * ljp[c]['mass'] / total_mass
    print(f"  {c:<8} {wf*100:.1f} wt%  (vol: {comp[c]*100:.1f}%)")
total_vol = sum(comp[c] * ljp[c]['sigma']**3 for c in comp)
print()
print("Volume fractions (sigma^3 proxy):")
for c in comp:
    vf = comp[c] * ljp[c]['sigma']**3 / total_vol
    print(f"  {c:<8} {vf*100:.1f} vol%")
n = s.N_TOTAL
avg_sig = sum(ljp[c]['sigma'] for c in ljp) / len(ljp)
box_l = (n * avg_sig**3 / 0.85)**(1/3)
layer_um = (box_l / math.sqrt(n)) * 1000
print(f"\nEstimated TIM layer thickness: {layer_um:.1f} µm")
print(f"Processing temperature: {s.TEMPERATURE:.0f} K ({s.TEMPERATURE-273.15:.0f} °C)")
PYEOF
        echo ""
        echo "=== Experiment history (keep entries only) ==="
        awk -F'\t' '$4 == "keep"' results.tsv 2>/dev/null | sort -t$'\t' -k2 -g \
            | awk -F'\t' '{printf "  %-10s  R=%-15s  t=%-8s  %s\n", $1, $2, $3, $5}'
    } | tee "$REPORT_FILE"
fi

git add "$REPORT_FILE" 2>/dev/null || true
git diff --cached --quiet || \
    git commit -m "manufacturing_report: thermal_resistance=${BEST_R}" 2>/dev/null || true
log "Done. Manufacturing report committed."
