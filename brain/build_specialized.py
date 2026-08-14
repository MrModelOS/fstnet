"""Специализированный датасет JARVIS для Stage-2 обучения FST-Net (3B 1-bit MoF).

Формат траекторий (как jarvis_full.json):
  [[ (role, content), ... ]]  — роли system/user/assistant, читается make_samples().
  thinking: <think>...</think>, tool-calling: <tool_call>{...}</tool_call>, ответ «Сэр: ...».

Пропорция (500K):
  30%  — Linux/Bash: администрирование, скрипты, файловая система, процессы,
         сеть, пакеты, docker, git, cron/systemd, текст (awk/sed/grep), бэкапы.
  30%  — Multi-turn Tool-Calling: траектории с 2-4 вызовами инструментов
         (ОС + GGM — граф знаний), результат приходит system-сообщением.
  40%  — Synth-Math: продвинутая логика/математика с реально вычисленными
         ответами и пошаговым CoT.

Выход: data/jarvis_special.json  (валидный JSON-массив, инкрементальная запись).

Запуск:  python3 build_specialized.py  [--count 500000] [--seed 42]
Env-перезапись: JARVIS_COUNT, JARVIS_SEED
"""
import os
import json
import math
import random
import argparse

IM_S, IM_E = "<|im_start|>", "<|im_end|>"

SYSTEM_PROMPT = (
    "You are JARVIS, an ultra-competent, loyal, polite, and witty AI assistant. "
    "You address the user as 'Sir', speak concisely, and assist with coding, "
    "system controls, research, and self-refinement. You use tool calls when "
    "an action or environment query is required."
)

def think(text):
    return f"<think>{text}</think>\n"

def sir(text):
    return f"Certainly, Sir. {text}"

# ---------- Linux / Bash (30%) ----------

BASH_PROMPTS = [
    "Write a bash script that {task}.",
    "How would you {task}? Give the commands.",
    "Automate this: {task}.",
    "Give me a one-liner to {task}.",
    "Explain and provide a script that {task}.",
]

BASH_DOMAINS = [
    {  # файлы: поиск старых файлов и архивация
        "task": ["find and archive all files older than {days} days in {dir_}",
                 "move all files modified more than {days} days ago to {dir_}/old"],
        "script": "#!/usr/bin/env bash\nset -euo pipefail\n\ntarget={dir_}\ncutoff={days}\n\nfind \"$target\" -type f -mtime +{days} -print0 | while IFS= read -r -d '' f; do\n  # старые файлы -> архив\n  tar -czf \"${{f}}.tar.gz\" -C \"$(dirname \"$f\")\" \"$(basename \"$f\")\" && rm -f \"$f\"\ndone\necho \"Archived files older than {days} days under {dir_}\"",
        "check": "find -type f -mtime +N / tar -czf",
    },
    {  # логи: ротация по размеру
        "task": ["rotate log files larger than {size}M in {dir_}",
                 "truncate logs exceeding {size}M and keep the newest"],
        "script": "#!/usr/bin/env bash\nLOG_DIR={dir_}\nMAX={size}\n\nfind \"$LOG_DIR\" -name '*.log' -size +{size}M -print0 | while IFS= read -r -d '' f; do\n  mv \"$f\" \"${{f}}.$(date +%Y%m%d)\"\n  gzip \"${{f}}.$(date +%Y%m%d)\"\n  : > \"$f\"\ndone\necho \"Rotated logs > {size}M in $LOG_DIR\"",
        "check": "find -size +NM / gzip",
    },
    {  # процессы: убийство зависших
        "task": ["kill all processes named {proc} that are running longer than {sec}s",
                 "find and terminate runaway {proc} processes"],
        "script": "ps -eo pid,etime,comm | awk -v p={proc} -v lim={sec} '$3==p {{ split($2,a,\":\"); s=a[1]*3600+a[2]*60+a[3]; if (s>lim) print $1 }}' | xargs -r kill -9",
        "check": "ps / awk / xargs kill",
    },
    {  # сеть: прослушивающие порты
        "task": ["list which processes are listening on which ports",
                 "show open ports and their owning processes"],
        "script": "ss -tulnp | awk 'NR>1 {{ print $5, $7 }}'",
        "check": "ss -tulnp",
    },
    {  # пакеты: массовая установка
        "task": ["install a list of packages from a file {dir_}/pkgs.txt with apt",
                 "bulk-install the packages listed in {dir_}/pkgs.txt"],
        "script": "apt-get update && xargs -a {dir_}/pkgs.txt apt-get install -y",
        "check": "xargs -a / apt-get",
    },
    {  # docker: чистка
        "task": ["remove all stopped containers, dangling images and unused volumes",
                 "clean up docker resources that are no longer used"],
        "script": "docker container prune -f && docker image prune -a -f && docker volume prune -f && docker builder prune -f",
        "check": "docker prune",
    },
    {  # git: авто-коммит
        "task": ["commit all changes with a message based on the date",
                 "stage everything and commit with a timestamped message"],
        "script": "git add -A && git commit -m \"$(date +%Y-%m-%d) auto-commit\" && git push origin $(git branch --show-current)",
        "check": "git add / commit / push",
    },
    {  # cron: бэкап по расписанию
        "task": ["schedule a backup script {dir_}/backup.sh to run daily at {hour}:{min}",
                 "set up a cron job for {dir_}/backup.sh every day at {hour}:{min}"],
        "script": "(crontab -l 2>/dev/null | grep -v '{dir_}/backup.sh'; echo '{min} {hour} * * * {dir_}/backup.sh') | crontab -",
        "check": "crontab -l",
    },
    {  # текст: массовая замена
        "task": ["replace all occurrences of {a} with {b} in *.conf files under {dir_}",
                 "do an in-place sed replacement of {a} -> {b} for configs in {dir_}"],
        "script": "find {dir_} -name '*.conf' -exec sed -i 's/{a}/{b}/g' {{}} +",
        "check": "find -exec sed -i",
    },
    {  # бэкапы: rsync инкрементальный
        "task": ["back up {dir_}/src to {dir_}/backups using rsync with hard links",
                 "create an incremental backup of {dir_}/src via rsync --link-dest"],
        "script": "#!/usr/bin/env bash\nset -euo pipefail\nsrc={dir_}/src\ndst={dir_}/backups\nlast=$(ls -1t \"$dst\" | head -1)\nrsync -a --link-dest=\"$dst/$last\" \"$src\" \"$dst/$(date +%Y%m%d)\"",
        "check": "rsync --link-dest",
    },
    {  # перф: топ по памяти
        "task": ["show the top {n} processes by memory usage",
                 "display the {n} heaviest processes by RAM"],
        "script": "ps aux --sort=-%mem | head -n {n}",
        "check": "ps aux --sort=-%mem",
    },
    {  # systemd: сервис
        "task": ["restart and enable the {svc} service at boot",
                 "reload systemd and make {svc} start on boot"],
        "script": "systemctl daemon-reload && systemctl restart {svc} && systemctl enable {svc}",
        "check": "systemctl enable",
    },
]

def bash_sample(rng):
    d = rng.choice(BASH_DOMAINS)
    task_t = rng.choice(d["task"])
    ctx = {k: v for k, v in d.items()}
    fill = {
        "dir_": rng.choice(["/var/log", "/var/www", "/opt/data", "/home/user/projects", "/srv/app"]),
        "days": rng.randint(7, 120),
        "size": rng.randint(10, 500),
        "proc": rng.choice(["python3", "node", "chrome", "ffmpeg", "java", "rclone"]),
        "sec": rng.randint(120, 7200),
        "hour": rng.randint(0, 23),
        "min": rng.randint(0, 59),
        "a": rng.choice(["localhost", "127.0.0.1", "DEBUG", "http://"]),
        "b": rng.choice(["127.0.0.1", "localhost", "INFO", "https://"]),
        "n": rng.randint(5, 20),
        "svc": rng.choice(["nginx", "postgresql", "docker", "sshd", "prometheus"]),
    }
    prompt = rng.choice(BASH_PROMPTS).format(task=task_t.format(**fill))
    script = d["script"].format(**fill)
    check = d["check"]
    co = think(
        f"Approach: pick a safe, idempotent sequence of commands, quote paths, "
        f"and verify with a lightweight check ({check}).")
    ans = (f"{co}{sir(f'here is the script you need:\\n```bash\\n{script}\\n```')}\n"
           f"It is idempotent and safe to rerun. Shall I wire it into a cron job, Sir?")
    return ([("system", SYSTEM_PROMPT), ("user", prompt), ("assistant", ans)])

# ---------- Multi-turn Tool-Calling (30%) ----------

def _res(fn, args, rng):
    if fn == "get_sys_metrics":
        return '{"cpu": "11%", "vram_free": "1.1GB", "ram_free": "5.9GB"}'
    if fn == "exec":
        return '{"exit": 0, "output": "done"}'
    if fn == "read_file":
        return json.dumps({"path": args.get("path"), "content": "port = 8000\ndebug = false"})
    if fn == "write_file":
        return json.dumps({"path": args.get("path"), "ok": True, "bytes": 214})
    if fn == "list_dir":
        return json.dumps({"path": args.get("path"), "entries": ["src", "tests", "config", "README.md"]})
    if fn == "git_status":
        return '{"branch": "master", "dirty": false, "ahead": 2}'
    if fn == "run_tests":
        return '{"passed": 42, "failed": 0, "duration_s": 3.4}'
    if fn == "install_package":
        return json.dumps({"name": args.get("name"), "installed": True, "version": "1.2.3"})
    if fn == "set_cron":
        return json.dumps({"ok": True, "entry": f"0 2 * * * {args.get('cmd')}"})
    if fn == "ggm_search":
        return json.dumps({"query": args.get("query"), "hits": ["node-127", "node-204", "node-88"]})
    if fn == "ggm_store":
        return json.dumps({"ok": True, "id": f"e-{rng.randint(1000, 9999)}", "fact": args.get("fact")})
    if fn == "ggm_link":
        return json.dumps({"ok": True, "edges": 1, "rel": args.get("rel")})
    if fn == "ggm_query":
        return json.dumps({"entity": args.get("entity"), "facts": ["created 2024-01", "owner: Sir", "repo: fstnet"]})
    return '{"ok": true}'

def call(fn, args, r):
    s = json.dumps(args) if args else "{}"
    return f'<tool_call>{{"name": "{fn}", "args": {s}}}</tool_call>'

def result(fn, args, r):
    return _res(fn, args, r)

SCENARIOS = [
    {  # дебаг падающего теста
        "user": "One of the tests keeps failing, Sir. Find out which and fix it.",
        "steps": [("list_dir", {"path": "tests"}),
                  ("read_file", {"path": "tests/test_api.py"}),
                  ("run_tests", {})],
        "final": "The failure was an uninitialized fixture in test_api.py. I fixed it and all 42 tests pass now, Sir.",
    },
    {  # занятое место
        "user": "Disk is nearly full. Diagnose and clean it up, Sir.",
        "steps": [("exec", {"cmd": "df -h /"}),
                  ("exec", {"cmd": "du -sh /var/log/*"}),
                  ("exec", {"cmd": "rm -f /var/log/*.gz"})],
        "final": "Cleared 4.2GB of rotated logs, Sir. Disk usage is back to 58%.",
    },
    {  # деплой
        "user": "Deploy the current branch and verify, Sir.",
        "steps": [("git_status", {}),
                  ("exec", {"cmd": "go build ./..."}),
                  ("exec", {"cmd": "systemctl restart app"})],
        "final": "Build succeeded and the app is back up on master, Sir. All health checks green.",
    },
    {  # исследование через GGM
        "user": "Remind me how the memory layout works, Sir.",
        "steps": [("ggm_search", {"query": "memory layout"}),
                  ("ggm_query", {"entity": "fstnet"})],
        "final": "From the graph: memory layout uses 1-bit fields over a frozen base — details in node-127, Sir.",
    },
    {  # установка тулчейна
        "user": "Install the build toolchain and confirm it works, Sir.",
        "steps": [("install_package", {"name": "clang"}),
                  ("exec", {"cmd": "clang --version"})],
        "final": "Toolchain installed, version 1.2.3 confirmed, Sir.",
    },
    {  # cron-бэкап
        "user": "Schedule nightly backups, Sir.",
        "steps": [("write_file", {"path": "backup.sh"}),
                  ("set_cron", {"cmd": "backup.sh"})],
        "final": "backup.sh written and scheduled for 02:00 nightly, Sir.",
    },
    {  # сохранение знания в граф
        "user": "Remember that I prefer Go for new services, Sir.",
        "steps": [("ggm_store", {"fact": "Sir prefers Go for new services"}),
                  ("ggm_link", {"rel": "prefers", "entity": "Go"})],
        "final": "Stored and linked in the knowledge graph, Sir. I will not forget.",
    },
    {  # отчёт
        "user": "Give me a summary of the project state, Sir.",
        "steps": [("git_status", {}),
                  ("run_tests", {}),
                  ("get_sys_metrics", {})],
        "final": "Branch master clean (2 commits ahead), 42/42 tests pass, CPU 11% — all nominal, Sir.",
    },
]

def tool_sample(rng):
    s = rng.choice(SCENARIOS)
    conv = [("system", SYSTEM_PROMPT), ("user", s["user"])]
    for fn, args in s["steps"]:
        conv.append(("assistant", call(fn, args, rng)))
        conv.append(("system", result(fn, args, rng)))
    conv.append(("assistant", s["final"]))
    return conv

# ---------- Synth-Math / логика (40%) ----------

def math_sample(rng):
    kind = rng.random()
    if kind < 0.20:
        return _arith(rng)
    if kind < 0.35:
        return _percent(rng)
    if kind < 0.50:
        return _avg(rng)
    if kind < 0.65:
        return _seq(rng)
    if kind < 0.80:
        return _system(rng)
    if kind < 0.90:
        return _workrate(rng)
    return _prob(rng)

def _answer(q, ans, steps, rng):
    co = think(f"Let me reason step by step, Sir. {steps} Therefore: {ans}.")
    return ([("system", SYSTEM_PROMPT), ("user", q), ("assistant", f"{co}The answer is {ans}, Sir.")])

def _arith(rng):
    a, b, c, d = (rng.randint(2, 99) for _ in range(4))
    ans = a * b + c * d
    q = f"Evaluate {a} × {b} + {c} × {d}. Show the arithmetic."
    st = f"{a}×{b} = {a*b}; {c}×{d} = {c*d}; sum = {a*b} + {c*d} = {ans}."
    return _answer(q, ans, st, rng)

def _percent(rng):
    p = rng.randint(3, 75)
    n = rng.randint(40, 900)
    ans = round(p * n / 100, 2)
    q = f"What is {p}% of {n}?"
    st = f"{p}/100 × {n} = {p*n}/100 = {ans}."
    return _answer(q, ans, st, rng)

def _avg(rng):
    k = rng.randint(4, 8)
    nums = [rng.randint(10, 990) for _ in range(k)]
    s = sum(nums)
    ans = round(s / k, 2)
    q = f"Find the average of {', '.join(map(str, nums))}."
    st = f"Sum = {s}; count = {k}; average = {s}/{k} = {ans}."
    return _answer(q, ans, st, rng)

def _seq(rng):
    a0 = rng.randint(1, 20)
    dd = rng.randint(2, 9)
    n = rng.randint(5, 20)
    s = n * (2 * a0 + (n - 1) * dd) // 2
    q = f"The first term of an arithmetic sequence is {a0}, common difference {dd}. "
    q += f"What is the sum of the first {n} terms?"
    st = f"Sum = n/2 × (2a + (n−1)d) = {n}/2 × ({2*a0} + {n-1}×{dd}) = {s}."
    return _answer(q, s, st, rng)

def _system(rng):
    x, y = rng.randint(1, 12), rng.randint(1, 12)
    p, qq = rng.randint(2, 4), rng.randint(2, 4)
    eq1 = f"x + y = {x + y}"
    eq2 = f"{p}x + {qq}y = {p*x + qq*y}"
    q = f"Solve the system: {eq1} and {eq2}."
    st = (f"From the first, y = {x+y} − x. Substituting: {p}x + {qq}({x+y} − x) = "
          f"{p*x+qq*y} → {p-qq}x + {qq*(x+y)} = {p*x+qq*y} → x = {x}, then y = {y}.")
    return _answer(q, f"x = {x}, y = {y}", st, rng)

def _workrate(rng):
    a, b = rng.randint(2, 8), rng.randint(2, 8)
    ans = round(a * b / (a + b), 2)
    q = f"Machine A finishes a job in {a} hours, machine B in {b} hours. "
    q += "How long with both working together?"
    st = f"Rate = 1/{a} + 1/{b} = {a+b}/({a*b}) jobs/hour → time = {a*b}/({a+b}) = {ans} h."
    return _answer(q, f"{ans} hours", st, rng)

def _prob(rng):
    r, bl = rng.randint(1, 9), rng.randint(1, 9)
    g = math.gcd(r, r + bl)
    ans = f"{r//g}/{ (r+bl)//g }"
    q = f"A bag has {r} red and {bl} blue balls. Draw one at random: probability it is red?"
    st = f"P = red/total = {r}/({r}+{bl}) = {ans}."
    return _answer(q, ans, st, rng)

# ---------- Сборка ----------

def mix_selector(rng, frac):
    r = rng.random()
    if r < frac["bash"]: return "bash"
    if r < frac["bash"] + frac["tool"]: return "tool"
    return "math"

BUILDERS = {"bash": bash_sample, "tool": tool_sample, "math": math_sample}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=int(os.environ.get("JARVIS_COUNT", "500000")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("JARVIS_SEED", "42")))
    ap.add_argument("--out", default="data/jarvis_special.json")
    args = ap.parse_args()

    frac = {"bash": 0.30, "tool": 0.30, "math": 0.40}
    rng = random.Random(args.seed)
    counter = {"bash": 0, "tool": 0, "math": 0}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        f.write("[")
        for i in range(args.count):
            kind = mix_selector(rng, frac)
            conv = BUILDERS[kind](rng)
            f.write("\n" + json.dumps(conv) + ("," if i < args.count - 1 else ""))
            counter[kind] += 1
            if i and i % 50000 == 0:
                print(f"  {i} samples...", flush=True)
        f.write("\n]")
    os.replace(tmp, args.out)

    total = args.count
    print(f"Saved {total} conversations -> {args.out}")
    for k, v in counter.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

if __name__ == "__main__":
    main()
