#!/bin/bash
# loop_1node.sh — Single-node TIM optimization loop (node 0 only)
# vLLM + GROMACS both run on node 0. Used for 1-node vs 4-node comparison.
# NEVER modifies /home/nvidia/autotherm/ — fully independent experiment.

REPO_DIR="/home/nvidia/autotherm_1node"
PYTHON="python3"
VLLM_PORT=8090
MAX_ITER="${1:-100}"
GMX="/usr/local/gromacs/bin/gmx_mpi"

cd "$REPO_DIR"

log() { echo "[$(date '+%H:%M:%S')] [1node] $*"; }

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  MODE : SINGLE NODE (node 0 only)                   ║"
echo "║  LLM  : Nemotron-3-Super-120B  via  vLLM :8090      ║"
echo "║  SIM  : GROMACS mpirun -n 1 on node 0               ║"
echo "║  NOTE : vLLM and GROMACS share node 0 resources     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

[ ! -f "$GMX" ] && { log "ERROR: gmx_mpi not found"; exit 1; }

if ! curl -sf http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
    log "ERROR: vLLM not running on :${VLLM_PORT}. Start it first."; exit 1
fi
log "vLLM ready on :${VLLM_PORT}."

if [ ! -f "$REPO_DIR/structure.py.baseline" ]; then
    cp "$REPO_DIR/structure.py" "$REPO_DIR/structure.py.baseline"
fi

for iter in $(seq 1 "$MAX_ITER"); do
    log "══════ Iteration $iter / $MAX_ITER ══════"

    BEST_COMMIT=$(awk -F'\t' '$4=="keep"{print $1}' results.tsv 2>/dev/null | tail -1)
    if [ -n "$BEST_COMMIT" ]; then
        git show ${BEST_COMMIT}:structure.py > structure.py 2>/dev/null \
            || cp "$REPO_DIR/structure.py.baseline" structure.py
        log "Reset structure.py to best known commit (${BEST_COMMIT})."
    else
        cp "$REPO_DIR/structure.py.baseline" structure.py
        log "Reset structure.py to baseline (no best yet)."
    fi

    BEST_R=$(awk -F'\t' '$4 == "keep" {print $2}' results.tsv 2>/dev/null | sort -g | head -1)
    BEST_R="${BEST_R:-2.28e-5}"
    TRIED=$(awk -F'\t' '$4 == "keep" || $4 == "discard"' results.tsv 2>/dev/null | wc -l || echo 0)
    log "Best thermal_resistance=$BEST_R  |  Total experiments=$TRIED"

    log "Running call_nemotron agent (iter $iter)..."
    python3 "$REPO_DIR/call_nemotron.py" "$REPO_DIR" "$BEST_R" "$TRIED" \
        2> >(while IFS= read -r line; do log "$line"; done)
    AGENT_EXIT=$?
    if [ $AGENT_EXIT -ne 0 ]; then
        log "WARNING: call_nemotron agent failed (exit=$AGENT_EXIT) — skipping."; continue
    fi

    if ! $PYTHON -m py_compile structure.py 2>/tmp/syntax_err_1node.txt; then
        log "WARNING: syntax error — skipping."
        cp "$REPO_DIR/structure.py.baseline" structure.py; continue
    fi
    if git diff --quiet structure.py; then
        log "WARNING: structure.py unchanged — skipping."; continue
    fi

    $PYTHON -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('s', 'structure.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
total = sum(s.COMPOSITION.values())
if abs(total - 1.0) > 1e-6:
    print(f'INVALID: {total:.6f}'); sys.exit(1)
print('OK')
" 2>/tmp/val_err_1node.txt || {
        log "WARNING: validation failed"
        cp "$REPO_DIR/structure.py.baseline" structure.py; continue
    }

    HEADLINE=$(head -10 structure.py | grep -i '#\|COMPOSITION' | head -1 | sed 's/[#"*]*//g' | xargs)
    git add structure.py
    git commit -m "iter-${iter}: ${HEADLINE:-TIM optimization}" \
        || { log "Commit failed — skipping."; continue; }
    COMMIT=$(git rev-parse --short HEAD)
    log "Committed $COMMIT"

    log "Running GROMACS single-node (iter $iter)..."
    $PYTHON prepare.py 2>&1 | tee run.log
    SIM_EXIT=${PIPESTATUS[0]}
    log "Simulation complete (exit=$SIM_EXIT)."

    if [ "$SIM_EXIT" -ne 0 ]; then
        CRASH_MSG=$(grep -m1 "Error\|INVALID\|ERROR\|Traceback" run.log 2>/dev/null || echo "unknown")
        log "WARNING: CRASHED — $CRASH_MSG"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$COMMIT" "N/A" "N/A" "crash" "iter-${iter}: $CRASH_MSG" >> results.tsv
        git reset HEAD~1 2>/dev/null || true
        cp "$REPO_DIR/structure.py.baseline" structure.py; continue
    fi

    THERMAL_R=$(grep -o 'thermal_resistance:[[:space:]]*[-0-9.eE+]*' run.log \
                | sed 's/thermal_resistance:[[:space:]]*//' | head -1)
    SIM_TIME=$(grep -o 'sim_time_s:[[:space:]]*[0-9.]*' run.log \
               | sed 's/sim_time_s:[[:space:]]*//' | head -1)

    if [ -z "$THERMAL_R" ]; then
        log "WARNING: thermal_resistance not found — skipping."
        git reset HEAD~1 2>/dev/null || true
        cp "$REPO_DIR/structure.py.baseline" structure.py; continue
    fi

    log "thermal_resistance=$THERMAL_R  best=$BEST_R  sim_time=${SIM_TIME}s"

    BETTER=$($PYTHON -c "
try:    print('yes' if float('${THERMAL_R}') < float('${BEST_R}') else 'no')
except: print('no')
")

    if [ "$BETTER" = "yes" ]; then
        STATUS="keep"
        log "NEW BEST — keeping."
    else
        STATUS="discard"
        log "No improvement — reverting."
        git reset HEAD~1
        cp "$REPO_DIR/structure.py.baseline" structure.py
    fi

    DESC="iter-${iter}: ${HEADLINE:-TIM optimization}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$COMMIT" "${THERMAL_R}" "${SIM_TIME:-?}" "$STATUS" "$DESC" >> results.tsv
    log "Iteration $iter complete."
    echo ""
done

log "Loop complete — $MAX_ITER iterations done."
