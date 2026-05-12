#!/usr/bin/env python3
"""
call_nemotron.py — Direct vLLM agent for TIM optimization.
Calls vLLM /v1/chat/completions directly. No tool calls needed.
Nemotron outputs the new structure.py as a plain python code block.

Usage: python3 call_nemotron.py <repo_dir> <best_r> <tried>
Writes structure.py on success. Exits 0 on success, 1 on failure.
"""

import sys, os, re, json, time

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests

VLLM_URL = "http://localhost:8080/v1/chat/completions"
MODEL    = "/home/nvidia/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

def read_file(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except Exception as e:
        return "[could not read {}: {}]".format(path, e)

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
            print("[agent] attempt {} failed: {}".format(attempt+1, e), file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(5)
    return None

def extract_code_block(text):
    # Strip Nemotron thinking tokens
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    # Try ```python block first, then plain ```
    m = re.search(r"```python[\s\S]*?\n([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\n([\s\S]*?)```", text)
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

    prompt = (
        "You are a Computational Materials Scientist optimizing a Thermal Interface Material (TIM).\n"
        "Current best thermal_resistance = {} m2K/W. LOWER is BETTER.\n"
        "Total experiments completed: {}\n\n"
        "=== structure.py (baseline, reset each iteration) ===\n"
        "{}\n\n"
        "=== results.tsv (full experiment history) ===\n"
        "{}\n\n"
        "=== program.md (optimization guide and parameter bounds) ===\n"
        "{}\n\n"
        "Study the history carefully. Choose ONE change that has NOT been tried yet "
        "and is likely to significantly reduce thermal_resistance based on program.md strategies.\n\n"
        "Output ONLY the complete new structure.py inside a python code block. "
        "No text outside the block.\n\n"
        "```python\n"
        "# complete new structure.py here\n"
        "```"
    ).format(best_r, tried, structure, results, program)

    print("[agent] Calling Nemotron...", file=sys.stderr, flush=True)
    t0 = time.time()
    content = call_vllm(prompt)
    elapsed = time.time() - t0

    if content is None:
        print("[agent] ERROR: vLLM call failed after retries", file=sys.stderr)
        sys.exit(1)

    print("[agent] Response in {:.1f}s ({} chars)".format(elapsed, len(content)),
          file=sys.stderr, flush=True)

    new_structure = extract_code_block(content)
    if new_structure is None:
        print("[agent] ERROR: no python code block found", file=sys.stderr)
        print("[agent] Preview:", content[:600], file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join(repo, "structure.py")
    with open(out_path, "w") as f:
        f.write(new_structure + "\n")

    print("[agent] Wrote {} chars to structure.py".format(len(new_structure)),
          file=sys.stderr, flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
