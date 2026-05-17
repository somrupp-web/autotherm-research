#!/bin/bash
# test_10M_1node.sh — runs on node 1 (10.137.203.184)
# Test 1: same config as benchmark — verify crash is reproducible
# Test 2: -tunepme no -dlb no — bypass PME load balancer, see if it still crashes
#
# Both reuse the prod.tpr from the preserved benchmark workdir.

set -u
WORKDIR=/tmp/bench_10000000_w7kt2ckd
GMX=/usr/local/gromacs/bin/gmx_mpi
LD="${LD_LIBRARY_PATH:-}"

cd "$WORKDIR" || { echo "no workdir"; exit 1; }

# Clean any leftover test outputs from prior attempts
rm -f test1.* test2.* 2>/dev/null

# ── Test 1: SAME CONFIG AS BENCHMARK (reproduce the crash) ────────────────────
echo "================================================================"
echo "TEST 1: same config as benchmark — looking for reproducible crash"
echo "================================================================"
date
T0=$(date +%s)
timeout 14400 mpirun -n 1 --mca pml ob1 --mca btl self,tcp \
    -x LD_LIBRARY_PATH="$LD" \
    $GMX mdrun -v -s prod.tpr -deffnm test1 \
    -nb gpu -pme gpu -bonded cpu -update gpu -gpu_id 0 -ntomp 4 2>&1 \
    | tee test1.stdout
RC1=${PIPESTATUS[0]}
T1=$(date +%s)
DUR1=$((T1 - T0))
echo "=== TEST 1 RESULT: exit=$RC1  duration=${DUR1}s ==="
if [ -f test1.log ]; then
    LAST_STEP=$(awk '/^[ ]+[0-9]+[ ]+[0-9.]+$/{last=$0} END{print last}' test1.log)
    echo "TEST 1 last step recorded: $LAST_STEP"
fi
echo ""

# ── Test 2: -tunepme no -dlb no (bypass load balancer) ────────────────────────
echo "================================================================"
echo "TEST 2: -tunepme no -dlb no -pme-order 4  (bypass PME balancer)"
echo "================================================================"
date
T0=$(date +%s)
timeout 14400 mpirun -n 1 --mca pml ob1 --mca btl self,tcp \
    -x LD_LIBRARY_PATH="$LD" \
    $GMX mdrun -v -s prod.tpr -deffnm test2 \
    -nb gpu -pme gpu -bonded cpu -update gpu -gpu_id 0 -ntomp 4 \
    -tunepme no -dlb no -pme-order 4 2>&1 \
    | tee test2.stdout
RC2=${PIPESTATUS[0]}
T1=$(date +%s)
DUR2=$((T1 - T0))
echo "=== TEST 2 RESULT: exit=$RC2  duration=${DUR2}s ==="
if [ -f test2.log ]; then
    LAST_STEP=$(awk '/^[ ]+[0-9]+[ ]+[0-9.]+$/{last=$0} END{print last}' test2.log)
    echo "TEST 2 last step recorded: $LAST_STEP"
fi
echo ""

echo "================================================================"
echo "SUMMARY"
echo "================================================================"
echo "Test 1 (same as benchmark):       exit=$RC1  duration=${DUR1}s"
echo "Test 2 (-tunepme no -dlb no):     exit=$RC2  duration=${DUR2}s"
date
