#!/bin/bash
# FST-Net local server optimized for Wayland/Hyprland + Vulkan
export GGML_VK_VISIBLE_DEVICES=0
export OLLAMA_NUM_PARALLEL=1

# Option 1: Ollama
# ollama serve &
# ollama run fstnet "prompt"

# Option 2: Custom server (lower overhead)
nice -n 10 ionice -c 3 python3 body/fstnet_server.py &

# Option 3: llama.cpp with Vulkan
# ./llama-server -m checkpoints/fstnet-152m-q8.gguf -ngl 99 --ctx-size 2048 -t 4 --memory-f16
