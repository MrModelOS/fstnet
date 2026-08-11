#!/usr/bin/env python3
"""Update repo in Colab via git pull."""
import subprocess, os

# Find repo root
current = os.getcwd()
while current != "/":
    if os.path.exists(os.path.join(current, ".git")):
        break
    current = os.path.dirname(current)

if current == "/":
    print("ERROR: not in a git repo")
else:
    os.chdir(current)
    print(f"Updating {current}...")
    result = subprocess.run(["git", "pull"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    print(f"Now at: {subprocess.getoutput('git log --oneline -1')}")
