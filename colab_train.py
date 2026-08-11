#!/usr/bin/env python3
"""FST-Net training script for Google Colab with T4 GPU."""
import os, subprocess

def setup():
    """Install deps and clone repo."""
    subprocess.run(["pip", "install", "-q", "torch", "transformers", "datasets", 
                    "tokenizers", "tqdm", "safetensors"], check=True)
    
    if not os.path.exists("fstnet"):
        subprocess.run(["git", "clone", "https://github.com/MrModelOS/fstnet.git"], check=True)
    
    os.chdir("fstnet")
    print(f"Working dir: {os.getcwd()}")
    print(f"GPU: {os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()}")

def download_data():
    """Download training data."""
    from datasets import load_dataset
    import json
    
    print("Loading ultrachat_200k...", flush=True)
    ds1 = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    ultrachat = []
    for i, d in enumerate(ds1):
        msgs = [(m["role"], m["content"]) for m in d.get("messages", []) 
                if m["role"] in ("user","assistant")]
        if len(msgs) >= 2:
            ultrachat.append(msgs)
        if i >= 10000:
            break
    print(f"  ultrachat: {len(ultrachat)}", flush=True)
    
    print("Loading gsm8k...", flush=True)
    ds2 = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k = [[("user", d["question"]), ("assistant", d["answer"])] for d in ds2]
    print(f"  gsm8k: {len(gsm8k)}", flush=True)
    
    all_data = ultrachat + gsm8k
    with open("data/train_extra.json", "w") as f:
        json.dump(all_data, f)
    print(f"Total: {len(all_data)} samples", flush=True)

def train():
    """Run training."""
    import subprocess
    subprocess.run(["python3", "train_colab.py"], check=True)

if __name__ == "__main__":
    setup()
    download_data()
    train()
