#!/bin/bash
# ============================================================
# Autonomous Thermal Interface Material Research Loop
#
# Single-node usage:
#   ./loop.sh [max_iterations]
#
# Cluster usage (set by cluster_launch.sh):
#   CLUSTER_SIZE=4 NODE_ID=0 ./loop.sh [max_iterations]
# ============================================================

REPO_DIR="/home/nvidia/autotherm"
export PATH="/home/nvidia/.bun/bin:$PATH"
OPENCODE=$(find /home/nvidia/.local/bin /home/nvidia/.local/share/fnm /home/nvidia/.bun/install/global/node_modules/opencode-ai/bin -name opencode -type f 2>/dev/null | head -1)
MODEL="vllm//home/nvidia/models/Qwen3.5-122B-A10B-AWQ"
MAX_ITER="${1:-50}"
NODE_ID="${NODE_ID:-0}"
CLUSTER_SIZE="${CLUSTER_SIZE:-1}"
BRANCH="node-${NODE_ID}"
NODES=(0 1 2 3)

GMX="/usr/local/gromacs/bin/gmx_mpi"
PYTHON="python3"

cd "$REPO_DIR"

log() { echo "[$(date '+%H:%M:%S')] [node-${NODE_ID}] $*"; }

echo ""
echo "╔══════════════════════════════════════════════════════╗"
if [ "$CLUSTER_SIZE" -gt 1 ]; then
    echo "║  MODE : CLUSTER  (node ${NODE_ID} of $((CLUSTER_SIZE-1)), ${CLUSTER_SIZE} nodes total)         ║"
else
    echo "║  MODE : SINGLE NODE                                  ║"
fi
echo "║  GOAL : Minimize interfacial thermal resistance      ║"
echo "║  JUDGE: GROMACS 2025.1 GPU-accelerated NEMD          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Prerequisites ─────────────────────────────────────────────
[ -z "$OPENCODE" ] && { log "ERROR: opencode not found"; exit 1; }
[ ! -f "$GMX" ]    && { log "ERROR: gmx_mpi not found at $GMX. Build GROMACS first."; exit 1; }
log "OpenCode : $OPENCODE"
log "Model    : $MODEL"
log "GROMACS  : $GMX  ($($GMX --version 2>&1 | grep 'GROMACS version' | head -1))"

# ── Checkout node branch (cluster only) ──────────────────────
if [ "$CLUSTER_SIZE" -gt 1 ]; then
    git fetch origin 2>/dev/null || true
    if git show-ref --quiet "refs/remotes/origin/$BRANCH"; then
        git checkout -B "$BRANCH" "origin/$BRANCH" 2>/dev/null || git checkout "$BRANCH"
    else
        git checkout -B "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
    fi
fi

# ── Wait for vLLM ─────────────────────────────────────────────
log "Waiting for vLLM on :8080..."
for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1 && { log "vLLM ready."; break; }
    sleep 5
done
curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1 || { log "ERROR: vLLM not ready. Abort."; exit 1; }

# ── Cluster results sync ──────────────────────────────────────
sync_global_results() {
    if [ "$CLUSTER_SIZE" -le 1 ]; then return; fi
    git fetch origin 2>/dev/null || true
    > /tmp/global_results.tsv
    for n in "${NODES[@]}"; do
        git show "origin/node-${n}:results.tsv" 2>/dev/null \
            | grep -v '^commit	' >> /tmp/global_results.tsv || true
    done
    grep -v '^commit	' results.tsv 2>/dev/null >> /tmp/global_results.tsv || true
    sort -u -k1,1 /tmp/global_results.tsv \
        | { printf 'commit\tthermal_resistance\tsim_time_s\tstatus\tdescription\n'; cat; } \
        > results.tsv
    log "Synced: $(awk -F'\t' '$4 == "keep" || $4 == "discard"' results.tsv | wc -l) total experiments."
}

push_results() {
    if [ "$CLUSTER_SIZE" -le 1 ]; then return; fi
    git add results.tsv 2>/dev/null || true
    git diff --cached --quiet || \
        git commit -m "node-${NODE_ID}: results sync after iter-${iter}"
    git push origin "$BRANCH" 2>/dev/null || {
        git pull --rebase origin "$BRANCH" 2>/dev/null \
            && git push origin "$BRANCH" 2>/dev/null || true
    }
}

# ── Baseline structure.py ─────────────────────────────────────
if [ ! -f "$REPO_DIR/structure.py.baseline" ]; then
    cp "$REPO_DIR/structure.py" "$REPO_DIR/structure.py.baseline"
    log "Saved structure.py.baseline"
fi

# ── Main loop ─────────────────────────────────────────────────
for iter in $(seq 1 "$MAX_ITER"); do
    log "══════ Iteration $iter / $MAX_ITER ══════"

    # Reset to baseline each iteration (Qwen proposes from scratch based on results.tsv)
    cp "$REPO_DIR/structure.py.baseline" structure.py
    log "Reset structure.py to baseline."

    sync_global_results

    BEST_R=$(awk -F'\t' '$4 == "keep" {print $2}' results.tsv 2>/dev/null \
             | sort -n | head -1)
    BEST_R="${BEST_R:-9.99e-6}"
    TRIED=$(awk -F'\t' '$4 == "keep" || $4 == "discard"' results.tsv 2>/dev/null | wc -l || echo 0)
    log "Global best thermal_resistance=$BEST_R  |  Total experiments=$TRIED"

    if [ "$CLUSTER_SIZE" -gt 1 ]; then
        CONTEXT="You are AI research agent node-${NODE_ID} on a ${CLUSTER_SIZE}-node NVIDIA DGX Spark cluster.
NOTE: $((CLUSTER_SIZE-1)) other agents run simultaneously — check results.tsv and avoid duplicates."
    else
        CONTEXT="You are an AI research agent on a single NVIDIA DGX Spark."
    fi

    PROMPT="${CONTEXT}

You are a Computational Materials Scientist optimizing a Thermal Interface Material (TIM).
Current best thermal_resistance = ${BEST_R} m²K/W. LOWER is BETTER.

Steps you MUST follow:
1. Read ${REPO_DIR}/structure.py
2. Read ${REPO_DIR}/results.tsv
3. Read ${REPO_DIR}/program.md
4. Choose ONE specific change to structure.py that has not been tried yet.
5. Write the complete new structure.py using the bash tool:
   cat > ${REPO_DIR}/structure.py << 'PYEOF'
   [complete new python file]
   PYEOF
6. Stop. Do NOT run the simulation yourself. Do NOT modify prepare.py.

IMPORTANT: COMPOSITION values must sum to exactly 1.0.
Use bash to write the file (step 5). Do NOT use the edit tool."

    log "Running OpenCode agent..."
    "$OPENCODE" run --model "$MODEL" "$PROMPT"
    OPENCODE_EXIT=$?

    if [ $OPENCODE_EXIT -ne 0 ]; then
        log "WARNING: OpenCode exited $OPENCODE_EXIT — skipping."
        cp "$REPO_DIR/structure.py.baseline" structure.py
        continue
    fi

    if ! $PYTHON -m py_compile structure.py 2>/tmp/syntax_err.txt; then
        log "WARNING: syntax error in structure.py — skipping."
        log "$(cat /tmp/syntax_err.txt)"
        cp "$REPO_DIR/structure.py.baseline" structure.py
        continue
    fi

    if git diff --quiet structure.py; then
        log "WARNING: structure.py unchanged — skipping."
        continue
    fi

    # Quick parameter validation
    $PYTHON -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('s', 'structure.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
total = sum(s.COMPOSITION.values())
if abs(total - 1.0) > 1e-6:
    print(f'INVALID: composition sums to {total:.6f}')
    sys.exit(1)
print('OK')
" 2>/tmp/val_err.txt
    if [ $? -ne 0 ]; then
        log "WARNING: structure.py failed validation: $(cat /tmp/val_err.txt)"
        cp "$REPO_DIR/structure.py.baseline" structure.py
        continue
    fi

    HEADLINE=$(head -10 structure.py | grep -i '#\|"""\|COMPOSITION' | head -1 \
               | sed 's/[#"*]*//g' | xargs)
    git add structure.py
    git commit -m "node-${NODE_ID} iter-${iter}: ${HEADLINE:-TIM optimization}" \
        || { log "Commit failed — skipping."; continue; }
    COMMIT=$(git rev-parse --short HEAD)
    log "Committed $COMMIT"

    # ── Run GROMACS simulation ────────────────────────────────
    log "Running GROMACS simulation (GPU offload)..."
    $PYTHON prepare.py 2>&1 | tee run.log
    SIM_EXIT=${PIPESTATUS[0]}
    log "Simulation complete (exit=$SIM_EXIT)."

    if [ "$SIM_EXIT" -ne 0 ]; then
        CRASH_MSG=$(grep -m1 "Error\|INVALID\|ERROR\|Traceback" run.log 2>/dev/null || echo "unknown")
        log "WARNING: Simulation CRASHED — $CRASH_MSG"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$COMMIT" "N/A" "N/A" "crash" "node-${NODE_ID} iter-${iter}: $CRASH_MSG" >> results.tsv
        git reset HEAD~1 2>/dev/null || true
        cp "$REPO_DIR/structure.py.baseline" structure.py
        continue
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
        cp "$REPO_DIR/structure.py.baseline" structure.py
        continue
    fi

    log "thermal_resistance=$THERMAL_R  best=$BEST_R  sim_time=${SIM_TIME}s"

    # Save per-iteration geometry snapshot
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
        # Persist geometry for the new best iteration
        if [ -n "$NANOTUBE_GEOM" ]; then
            echo "$NANOTUBE_GEOM" > "$REPO_DIR/nanotube_geometry_best.json"
            log "Saved nanotube geometry for new best (iter $iter)."
        fi
    else
        STATUS="discard"
        log "No improvement — reverting."
        git reset HEAD~1
        cp "$REPO_DIR/structure.py.baseline" structure.py
    fi

    # Save baseline geometry once (iter 1 of node-0, first successful sim)
    if [ ! -f "$REPO_DIR/nanotube_geometry_baseline.json" ] && [ -n "$NANOTUBE_GEOM" ]; then
        echo "$NANOTUBE_GEOM" > "$REPO_DIR/nanotube_geometry_baseline.json"
        log "Saved nanotube geometry for baseline simulation."
    fi

    DESC="node-${NODE_ID} iter-${iter}: ${HEADLINE:-TIM optimization}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$COMMIT" "${THERMAL_R}" "${SIM_TIME:-?}" "$STATUS" "$DESC" \
        >> results.tsv

    push_results
    log "Iteration $iter complete."
    echo ""
done

log "Loop complete — $MAX_ITER iterations done on node-${NODE_ID}."

# ── Manufacturing Report ───────────────────────────────────────────────────────
# Only node-0 (or single-node mode) generates the final report.
if [ "${NODE_ID}" -eq 0 ]; then
    log "══════ Generating Manufacturing Formulation Report ══════"

    # Find best commit from results.tsv
    BEST_COMMIT=$(awk -F'\t' '$4 == "keep" {print $1, $2}' results.tsv 2>/dev/null \
                  | sort -k2 -n | head -1 | awk '{print $1}')
    BEST_R=$(awk -F'\t' '$4 == "keep" {print $2}' results.tsv 2>/dev/null \
             | sort -n | head -1)

    if [ -z "$BEST_COMMIT" ]; then
        log "No successful iterations found — using baseline structure."
        BEST_COMMIT="HEAD"
        BEST_R="N/A"
    fi

    log "Best commit: $BEST_COMMIT  thermal_resistance=$BEST_R m²K/W"

    # Check out best structure.py into a temp file
    BEST_STRUCT="/tmp/best_structure.py"
    git show "${BEST_COMMIT}:structure.py" > "$BEST_STRUCT" 2>/dev/null \
        || cp structure.py "$BEST_STRUCT"

    REPORT_FILE="$REPO_DIR/manufacturing_report.txt"
    REPORT_DATE=$(date '+%Y-%m-%d %H:%M:%S')

    REPORT_PROMPT="You are a materials scientist preparing a manufacturing specification for a Thermal Interface Material (TIM).

The AI research loop has completed ${MAX_ITER} GROMACS molecular dynamics simulations to find the optimal TIM formulation.

Best result: thermal_resistance = ${BEST_R} m²K/W (lower is better; target for GPU TIMs: < 1e-5 m²K/W).
All experiment history is in ${REPO_DIR}/results.tsv.

The winning formulation parameters are in this Python file:
$(cat $BEST_STRUCT)

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
   - Thermal resistance at operating temperature (${TEMPERATURE:-350} K)
   - Estimated conductivity range (λ = L/R)
   - Sensitivity: which parameter has the most impact on performance (based on results.tsv spread)

6. SCALE-UP NOTES
   - Batch size feasibility
   - QC test: 4-point thermal impedance measurement method
   - Shelf life estimate for InGa-based TIMs

Write in professional engineering report style. Be specific and quantitative wherever the simulation data supports it.
Output the report as plain text to stdout (it will be saved to a file).
Use bash to run: echo 'REPORT GENERATED' > /tmp/report_done.txt after writing."

    log "Invoking OpenCode for manufacturing report..."
    "$OPENCODE" run --model "$MODEL" "$REPORT_PROMPT" > "$REPORT_FILE" 2>&1
    REPORT_EXIT=$?

    if [ $REPORT_EXIT -eq 0 ] && [ -s "$REPORT_FILE" ]; then
        log "Manufacturing report saved to: $REPORT_FILE"
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║           MANUFACTURING FORMULATION REPORT                   ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo "Generated: $REPORT_DATE"
        echo "Best thermal_resistance: $BEST_R m²K/W  (commit: $BEST_COMMIT)"
        echo "Full report: $REPORT_FILE"
        echo "---"
        cat "$REPORT_FILE"
    else
        log "WARNING: OpenCode report generation failed (exit=$REPORT_EXIT). Generating fallback report."
        # Fallback: emit a structured summary from results.tsv + best structure.py directly
        {
            echo "THERMAL INTERFACE MATERIAL — MANUFACTURING FORMULATION REPORT"
            echo "Generated: $REPORT_DATE"
            echo "Research loop: $MAX_ITER iterations, GROMACS 2025.1 NVT-MD on 4x NVIDIA Blackwell GB10"
            echo ""
            echo "═══ BEST FORMULATION ═══"
            echo "Commit:              $BEST_COMMIT"
            echo "Thermal resistance:  $BEST_R m²K/W"
            echo ""
            echo "=== structure.py parameters ==="
            cat "$BEST_STRUCT"
            echo ""
            echo "=== Experiment history (keep entries only) ==="
            awk -F'\t' '$4 == "keep"' results.tsv 2>/dev/null \
                | sort -t$'\t' -k2 -n \
                | awk -F'\t' '{printf "  %-10s  R=%-15s  t=%-8s  %s\n", $1, $2, $3, $5}'
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
        } | tee "$REPORT_FILE"
        log "Fallback report saved to: $REPORT_FILE"
    fi

    # Commit the report to git
    git add "$REPORT_FILE" 2>/dev/null || true
    git diff --cached --quiet || \
        git commit -m "manufacturing_report: thermal_resistance=${BEST_R}" 2>/dev/null || true

    log "Done. Manufacturing report committed."
fi
