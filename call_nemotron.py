#!/usr/bin/env python3
"""
call_nemotron.py — Direct vLLM agent for TIM optimization.
Replaces OpenCode. No tool calls needed: Nemotron outputs structure.py as a code block.

Usage: python3 call_nemotron.py <repo_dir> <best_r> <tried>
  repo_dir : path to autotherm repo
  best_r   : current best thermal_resistance (e.g. 2.28e-5)
  tried    : number of experiments so far
Writes new structure.py on success. Exits 0 on success, 1 on failure.
"""

import sys, os, re, json, time
try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests

VLLM_URL = "http://localhost:8090/v1/chat/completions"
MODEL    = "/home/nvidia/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

def read_file(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[could not read {path}: {e}]"

def call_vllm(prompt, max_tokens=4096, temperature=0.7, retries=3):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(VLLM_URL, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[agent] attempt {attempt+1} failed: {e}", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(5)
    return None

def extract_code_block(text):
    """Strip <think> blocks then extract first python code block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    bt = chr(96) * 3
    for pattern in [bt + r"python\s*(.*?)" + bt, bt + r"\s*(.*?)" + bt]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None

def main():
    if len(sys.argv) < 4:
        print("Usage: call_nemotron.py <repo_dir> <best_r> <tried>", file=sys.stderr)
        sys.exit(1)

    repo, best_r, tried = sys.argv[1], sys.argv[2], sys.argv[3]

    structure = read_file(os.path.join(repo, "structure.py"))
    results   = read_file(os.path.join(repo, "results.tsv"))
    program   = read_file(os.path.join(repo, "program.md"))

    prompt = f"""You are a Computational Materials Scientist optimizing a Thermal Interface Material (TIM).
Current best thermal_resistance = {best_r} m²K/W. LOWER is BETTER.
Total experiments completed: {tried}

=== structure.py (baseline — reset each iteration, your job is to improve it) ===
{structure}

=== results.tsv (full experiment history: commit, R_th, sim_time, status, description) ===
{results}

=== program.md (optimization guide, parameter bounds, strategies) ===
{program}

Study the experiment history carefully. Choose ONE specific change that has NOT been tried yet and is likely to significantly reduce thermal_resistance based on the strategies in program.md.

Output ONLY the complete new structure.py file inside a single python code block. No explanation outside the block.

"""

    print("[agent] Calling Nemotron...", file=sys.stderr, flush=True)
    t0 = time.time()
    content = call_vllm(prompt)
    elapsed = time.time() - t0

    if content is None:
        print("[agent] ERROR: vLLM call failed after retries", file=sys.stderr)
        sys.exit(1)

    print(f"[agent] Response received in {elapsed:.1f}s ({len(content)} chars)", file=sys.stderr, flush=True)

    new_structure = extract_code_block(content)
    if new_structure is None:
        print("[agent] ERROR: no python code block found in response", file=sys.stderr)
        print("[agent] Response preview:", content[:500], file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join(repo, "structure.py")
    with open(out_path, "w") as f:
        f.write(new_structure + "\n")

    print(f"[agent] Wrote {len(new_structure)} chars to structure.py", file=sys.stderr, flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
