#!/bin/bash
# Update repo and sync with GitHub
cd /home/mrmodel/Проекты/MyAi/fstnet
git pull origin main 2>&1 | tail -3
echo "Updated to latest commit: $(git log --oneline -1)"
