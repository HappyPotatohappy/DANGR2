#!/bin/bash

# 폴더 생성
mkdir -p dangr scripts configs experiments data

# Python 파일 생성
touch dangr/__init__.py
touch dangr/algorithm.py
touch dangr/networks.py
touch dangr/utils.py

# 스크립트 파일 생성
touch scripts/train.py
touch scripts/sweep.py

# 기타 파일 생성
touch configs/dangr_config.json
touch requirements.txt
touch .gitignore

echo "프로젝트 구조 생성 완료!"
echo ""
echo "생성된 구조:"
tree -L 2
