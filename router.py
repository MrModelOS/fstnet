#!/usr/bin/env python3
"""FST-Net Router - orchestrates 152M (fast) and 800M (deep) models.

Light tasks → 152M (instant, ~100 t/s)
Heavy tasks → 800M (deep analysis, ~35-50 t/s)
"""
import os, requests, json, time

LIGHT_URL = "http://localhost:8000/v1/chat/completions"
HEAVY_URL = "http://localhost:8001/v1/chat/completions"

# Heuristics for routing
LIGHT_KEYWORDS = ["fix", "typo", "rename", "format", "lint", "comment", "docstring"]
HEAVY_KEYWORDS = ["refactor", "architecture", "design", "test", "debug", "optimize", "review"]


def route(task: str) -> str:
    """Decide which model handles the task."""
    task_lower = task.lower()
    
    # Check for heavy indicators
    for kw in HEAVY_KEYWORDS:
        if kw in task_lower:
            return "heavy"
    
    # Short tasks → light
    if len(task.split()) < 15:
        return "light"
    
    # Medium ambiguity → let light try first
    return "light"


def generate(task: str, chat_id: str = "default", max_tokens: int = 256) -> dict:
    """Smart routing with fallback."""
    model = route(task)
    url = HEAVY_URL if model == "heavy" else LIGHT_URL
    
    try:
        t0 = time.time()
        r = requests.post(url, json={
            "messages": [{"role": "user", "content": task}],
            "max_tokens": max_tokens,
            "chat_id": chat_id,
        }, timeout=30)
        elapsed = time.time() - t0
        
        result = r.json()
        result["_model"] = model
        result["_time"] = elapsed
        return result
        
    except Exception as e:
        # Fallback to other model
        url = LIGHT_URL if model == "heavy" else HEAVY_URL
        try:
            r = requests.post(url, json={
                "messages": [{"role": "user", "content": task}],
                "max_tokens": max_tokens,
                "chat_id": chat_id,
            }, timeout=30)
            result = r.json()
            result["_model"] = "fallback_" + ("light" if model == "heavy" else "heavy")
            return result
        except Exception as e2:
            return {"error": str(e2)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        result = generate(task)
        print(f"Model: {result.get('_model', '?')}")
        print(f"Time: {result.get('_time', 0):.2f}s")
        print(f"Response: {result.get('choices', [{}])[0].get('message', {}).get('content', result.get('error', 'NONE'))}")
