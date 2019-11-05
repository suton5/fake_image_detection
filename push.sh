#!/bin/bash
# This script is for git pull, add, commit, push

echo "Commit message?"

read msg

# To prevent merging issues, pull first
git pull

git add .
git commit -m "$msg"
git push

