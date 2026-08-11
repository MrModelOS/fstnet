#!/usr/bin/env python3
"""ACSC - Asynchronous Self-Critique & Speculative Roy.

Parallel pipeline:
  Generator (FST-Net) -> code
  Adversarial (FST-Net) -> failing test case
  Runner (subprocess) -> pass/fail
If fail -> feed error back to Generator (loop max 3x).
"""
import subprocess, tempfile, os, json, time, threading, requests

FSTNET_URL = "http://localhost:8000/v1/generate"
TIMEOUT = 10  # seconds for test execution


def generate_code(prompt, max_tokens=128, temp=0.4):
    """Call FST-Net to generate code."""
    try:
        r = requests.post(FSTNET_URL, json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temp
        }, timeout=15)
        return r.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


def generate_test(code, func_name):
    """Ask FST-Net to generate a failing test case."""
    prompt = f"Given this code, write ONE edge-case input that would make it fail:\n{code}\n\nFailing input for {func_name}:"
    try:
        r = requests.post(FSTNET_URL, json={
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0.7
        }, timeout=10)
        return r.json().get("response", "").strip()
    except Exception:
        return ""


def run_python_test(code, test_input, func_name):
    """Execute code + test in subprocess, return (passed, stderr)."""
    wrapper = f"""
{code}

# Test
try:
    result = {func_name}({test_input})
    print(f"RESULT: {{result}}")
except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True, text=True, timeout=TIMEOUT
            )
            os.unlink(f.name)
            passed = result.returncode == 0
            return passed, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            os.unlink(f.name)
            return False, "TIMEOUT"
        except Exception as e:
            os.unlink(f.name)
            return False, str(e)


def acsc_loop(task, max_iterations=3):
    """Full ACSC pipeline: generate -> critique -> test -> refine."""
    log = []
    code = None

    for i in range(max_iterations):
        print(f"\n=== Iteration {i+1}/{max_iterations} ===", flush=True)

        if i == 0:
            prompt = f"Write a Python function: {task}"
        else:
            prompt = f"Fix this function. Previous error: {error}\n\nCode:\n{code}\n\nFixed version:"

        # Generate code
        t0 = time.time()
        code = generate_code(prompt, max_tokens=128, temp=0.4)
        gen_time = time.time() - t0
        print(f"Generated ({gen_time:.1f}s): {code[:100]}...", flush=True)
        log.append({"step": "generate", "code": code, "time": gen_time})

        if code.startswith("ERROR"):
            return {"success": False, "error": code, "log": log}

        # Extract function name
        func_name = "unknown"
        if "def " in code:
            func_name = code.split("def ")[1].split("(")[0].strip()

        # Generate adversarial test (parallel in thread)
        test_input = None
        def get_test():
            nonlocal test_input
            test_input = generate_test(code, func_name)

        t = threading.Thread(target=get_test)
        t.start()
        t.join(timeout=10)

        if not test_input:
            test_input = "None"  # fallback

        # Run test
        passed, output = run_python_test(code, test_input, func_name)
        print(f"Test '{test_input}': {'PASS' if passed else 'FAIL'}", flush=True)
        print(f"Output: {output[:100]}", flush=True)
        log.append({"step": "test", "input": test_input, "passed": passed, "output": output})

        if passed:
            return {"success": True, "code": code, "iterations": i + 1, "log": log}

        error = output[:200]

    return {"success": False, "code": code, "iterations": max_iterations, "log": log}


if __name__ == "__main__":
    result = acsc_loop("def factorial(n):")
    print(f"\n=== RESULT ===")
    print(f"Success: {result['success']}")
    print(f"Iterations: {result.get('iterations', 0)}")
    print(f"Code:\n{result.get('code', '')[:300]}")
