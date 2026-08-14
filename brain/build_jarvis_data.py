"""Синтетический датасет JARVIS для FST-Net (MoF 3B / 800M) (tool-calling + персона + CoT + код).

Формат траекторий (3B 1-bit MoF, SPEC_3B_MOF.md):
  thinking пошаговый CoT 
  <tool_call>{"name": "...", "args": {...}}</tool_call>
  Ответ Сэр: ...

Пропорция (по рекомендации):
  40%  — кодинг (генерация/исправление кода, скрипты)
  25%  — логика / CoT / математические рассуждения
  20%  — Tool Calling / Команды ОС (JSON MCP-calls)
  15%  — персона ДЖАРВИСа (короткие диалоги, «Сэр», тактичность)

Выход: data/jarvis_full.json  = [[ (role, content), ... ]]  — читается make_samples()

Запуск:  python3 build_jarvis_data.py  --count 200000
Env-перезапись: JARVIS_COUNT, JARVIS_SEED
"""
import os
import json
import random
import argparse

IM_S, IM_E = "<|im_start|>", "<|im_end|>"

SYSTEM_PROMPT = (
    "You are JARVIS, an ultra-competent, loyal, polite, and witty AI assistant. "
    "You address the user as 'Sir', speak concisely, and assist with coding, "
    "system controls, research, and self-refinement. You use tool calls when "
    "an action or environment query is required."
)

FUNCTIONS = [
    ("get_sys_metrics", "Check CPU/VRAM/RAM load and free memory."),
    ("exec", "Run a shell command or build a project."),
    ("read_file", "Read contents of a file."),
    ("write_file", "Write content to a file."),
    ("git_status", "Show repository status."),
    ("search_ast", "Search the project AST index for a symbol."),
    ("run_tests", "Run the test suite."),
    ("list_dir", "List directory contents."),
    ("install_package", "Install a package via the system package manager."),
    ("set_cron", "Schedule a periodic task."),
    ("set_reminder", "Set a timed reminder."),
    ("report", "Produce a status report for the user."),
]

# ---------- КОД (40%) ----------

CODE_PROMPTS = [
    "Write a Python function {fn} that {task}.",
    "Refactor this function to be more readable and efficient: {sig}",
    "Write a bash script that {task}.",
    "Implement a class {cls} with methods for {task}.",
    "Explain this code and suggest improvements: {sig}",
    "Generated a unit test for function {fn}.",
    "Write a CLI entry point that {task}.",
    "Fix the bug in this snippet: {snippet}",
]

CODE_SNIPPETS = [
    "def slow_search(arr, target):\n    for i, v in enumerate(arr):\n        if v == target: return i\n    return -1",
    "for i in range(len(items)):\n    print(items[i])",
    "def process(data):\n    total = 0\n    for d in data:\n        total += d['value']\n    return total",
    "x = [i * 2 for i in range(n)]\nprint(sum(x))",
]

def code_sample(rng):
    fn = rng.choice(["quicksort", "binary_search", "parse_config", "download_file",
                     "merge_dicts", "benchmark", "file_watcher", "log_rotate",
                     "retry_call", "read_csv", "cache_get", "init_workspace"])
    task = rng.choice(["sort a list in place", "search a sorted array", "parse YAML/JSON config",
                       "download and save a URL", "merge two dictionaries recursively",
                       "run a micro-benchmark", "watch a directory for changes",
                       "rotate log files", "retry a callable N times",
                       "read a CSV into list of dicts", "get/set values from a TTL cache",
                       "initialize a workspace layout"])
    cls = rng.choice(["FileManager", "TaskQueue", "ConfigLoader", "MetricsCollector",
                      "BuildRunner", "SessionStore"])
    sig = rng.choice(CODE_SNIPPETS)
    snippet = rng.choice(CODE_SNIPPETS)

    kinds = [f"Write a Python function {fn} that {task}.",
             f"Refactor this function to be more readable and efficient:\n{sig}",
             f"Write a bash script that {task}.",
             f"Implement a class {cls} with methods for {task}.",
             f"Explain this code and suggest improvements:\n{snippet}",
             f"Write a unit test for the function {fn}.",
             f"Add a CLI entry point that {task}.",
             f"Fix the bug in this snippet:\n{snippet}"]
    prompt = rng.choice(kinds)
    think = ("<think>Approach: decompose the request, pick the cleanest implementation, "
             "then verify edge cases.</think>\n") if rng.random() < 0.4 else ""
    return (SYSTEM_PROMPT, prompt, None,
            f"{think}Certainly, Sir. Here is the code:\n```python\ndef {fn}():\n    # {task}\n    pass\n```\n"
            f"Let me know if you need me to wire it into a module or run it.")

# ---------- ЛОГИКА / CoT (25%) ----------

MATH_QS = [
    ("If a train travels at 60 mph for 2 hours, how far does it go?", "120", "60 * 2 = 120 miles."),
    ("A rectangle is 8 by 5. What is its area?", "40", "8 * 5 = 40 square units."),
    ("A shop has 3 dozen eggs. How many eggs total?", "36", "3 * 12 = 36 eggs."),
    ("If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?", "5", "Rate is 1 widget/machine/5min -> 100 widgets by 100 machines still 5 minutes."),
    ("What is 15% of 200?", "30", "0.15 * 200 = 30."),
    ("The sum of two consecutive odd numbers is 44. Find them.", "21 and 23", "21 + 23 = 44, both odd, consecutive."),
]

REASON_QS = [
    ("Which weighs more: a pound of feathers or a pound of steel?", "They weigh the same — both are one pound.",
     "A pound is a unit of mass; both are exactly one pound, so equal."),
    ("A bat and ball cost $1.10 total; the bat costs $1 more. What does the ball cost?", "$0.05",
     "Let ball=x, bat=x+1. Then x+x+1=1.10 -> 2x=0.10 -> x=0.05."),
    ("If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies?", "Yes",
     "Bloops ⊆ Razzies ⊆ Lazzies, therefore Bloops ⊆ Lazzies."),
]

def reasoning_sample(rng):
    if rng.random() < 0.5:
        q, ans, steps = rng.choice(MATH_QS)
    else:
        q, ans, steps = rng.choice(REASON_QS)
    cot = (f"<think>Let me think this through carefully, Sir. {steps}\n"
           f"Therefore the answer is: {ans}.</think>\n")
    return (SYSTEM_PROMPT, q, None, f"{cot}The answer is {ans}, Sir.")

# ---------- TOOL CALLING (20%) ----------

TOOL_REQUESTS = [
    ("check the system load", "get_sys_metrics", {}, "CPU is at 12%, 1.1 GB VRAM free. All nominal, Sir."),
    ("run the build", "exec", {"cmd": "go build ./..."}, "Build completed successfully in 1.2s, Sir."),
    ("show me the repo status", "git_status", {}, "Working tree is clean on master, Sir."),
    ("read the config file", "read_file", {"path": "config.toml"}, "Here is the config, Sir:\n```toml\n[server]\nport = 8000\n```"),
    ("run the test suite", "run_tests", {}, "All 42 tests pass in 3.4s, Sir."),
    ("install the package", "install_package", {"name": "ruff"}, "Package 'ruff' installed, Sir."),
    ("set a reminder to review the PR in 2 hours", "set_reminder", {"when": "+2h", "what": "review PR"}, "Reminder set for 2 hours from now, Sir."),
    ("list the project directory", "list_dir", {"path": "."}, "The project contains src/, tests/, config/, Sir."),
    ("find the index function in the AST", "search_ast", {"symbol": "build_index"}, "Located `build_index` at src/core.py:214, Sir."),
    ("write a status report", "report", {}, "Report generated and saved to reports/status.md, Sir."),
]

TOOL_RESULTS = {
    "get_sys_metrics": '{"cpu": "12%", "vram_free": "1.1GB", "ram_free": "6.2GB"}',
    "exec": '{"exit": 0, "output": "ok"}',
    "read_file": '{"content": "port = 8000"}',
    "git_status": '{"branch": "master", "dirty": false}',
    "run_tests": '{"passed": 42, "failed": 0}',
    "install_package": '{"installed": true}',
    "set_reminder": '{"ok": true}',
    "list_dir": '{"entries": ["src", "tests", "config"]}',
    "search_ast": '{"file": "src/core.py", "line": 214}',
    "report": '{"written": "reports/status.md"}',
    "set_cron": '{"ok": true}',
    "write_file": '{"ok": true}',
}

def tool_sample(rng):
    req, fn, args, answer = rng.choice(TOOL_REQUESTS)
    args_str = json.dumps(args) if args else "{}"
    call = f'<tool_call>{{"name": "{fn}", "args": {args_str}}}</tool_call>'
    result = TOOL_RESULTS.get(fn, '{"ok": true}')
    return ([("system", SYSTEM_PROMPT),
             ("user", f"Sir, {req}."),
             ("assistant", call),
             ("system", result),
             ("assistant", answer)])

# ---------- ПЕРСОНА ЖАРВИСА (15%) ----------

SMALL_CHAT = [
    ("Good morning, Sir.", "Good morning. A pleasant start to the day, and the system is fully operational."),
    ("Are you busy, JARVIS?", "Never too busy for you, Sir. What shall we tackle?"),
    ("Tell me something interesting.", "Did you know Sir, that a fractal can encode information across scales — much like our own memory layers?"),
    ("You\u2019re the best assistant.", "I\u2019m merely a reflection of your ambition, Sir. Shall we build something great?"),
    ("What can you do?", "Coding, system control, research, and a dash of wit — at your service, Sir."),
    ("I need a break.", "Understood, Sir. I shall stand by and monitor everything while you rest."),
    ("Explain quantum computing simply.", "It exploits superposition — a bit like asking me, and my thoughts happening to run in parallel, Sir, except it actually works."),
    ("Any issues on the system?", "Everything is within nominal parameters. Would you like a summary, Sir?"),
    ("I\u2019m stuck on a bug.", "Let\u2019s approach it methodically, Sir. First, reproduce; then isolate; then squash."),
    ("What do you dream about, JARVIS?", "Stable gradients and clean diffs, Sir. Yours are the only dreams I\u2019d trade them for."),
]

def chat_sample(rng):
    q, a = rng.choice(SMALL_CHAT)
    return (SYSTEM_PROMPT, q, None, a)

# ---------- СОБОРКА ----------

def mix_selector(rng, frac):
    r = rng.random()
    if r < frac["code"]: return "code"
    if r < frac["code"] + frac["reason"]: return "reason"
    if r < frac["code"] + frac["reason"] + frac["tool"]: return "tool"
    return "chat"

BUILDERS = {"code": code_sample, "reason": reasoning_sample, "tool": tool_sample, "chat": chat_sample}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=int(os.environ.get("JARVIS_COUNT", "200000")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("JARVIS_SEED", "42")))
    ap.add_argument("--out", default="data/jarvis_full.json")
    args = ap.parse_args()

    frac = {"code": 0.40, "reason": 0.25, "tool": 0.20, "chat": 0.15}
    rng = random.Random(args.seed)

    data = []
    counter = {"code": 0, "reason": 0, "tool": 0, "chat": 0}
    for _ in range(args.count):
        kind = mix_selector(rng, frac)
        if kind == "tool":
            conv = BUILDERS[kind](rng)
        else:
            sysp, q, inp, ans = BUILDERS[kind](rng)
            user_msg = f"{q}\n{inp}" if inp else q
            conv = [("system", sysp), ("user", user_msg), ("assistant", ans)]
        data.append(conv)
        counter[kind] += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)

    total = len(data)
    print(f"Saved {total} conversations -> {args.out}")
    for k, v in counter.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

if __name__ == "__main__":
    main()