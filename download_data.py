from datasets import load_dataset
import json

print("Loading ultrachat_200k...", flush=True)
ds1 = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
ultrachat = []
for i, d in enumerate(ds1):
    msgs = [(m["role"], m["content"]) for m in d.get("messages", []) if m["role"] in ("user","assistant")]
    if len(msgs) >= 2:
        ultrachat.append(msgs)
    if i >= 10000:
        break
print(f"  ultrachat: {len(ultrachat)}", flush=True)

print("Loading gsm8k...", flush=True)
ds2 = load_dataset("openai/gsm8k", "main", split="train")
gsm8k = [[("user", d["question"]), ("assistant", d["answer"])] for d in ds2]
print(f"  gsm8k: {len(gsm8k)}", flush=True)

print("Loading python_code...", flush=True)
ds3 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
pycode = []
for i, d in enumerate(ds3):
    instr = d.get("instruction", "").strip()
    out = d.get("output", "").strip()
    if instr and out:
        pycode.append([("user", instr), ("assistant", out)])
    if i >= 8000:
        break
print(f"  pycode: {len(pycode)}", flush=True)

all_data = ultrachat + gsm8k + pycode
with open("data/train_extra.json", "w") as f:
    json.dump(all_data, f)
print(f"Total: {len(all_data)}", flush=True)
