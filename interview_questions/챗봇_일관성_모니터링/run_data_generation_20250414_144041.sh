#!/bin/bash

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 로그 디렉토리 생성
mkdir -p logs

# 데이터 생성 스크립트 실행
echo "데이터 생성 시작..."
python interview_questions\AI_엔지니어_면접_문제_20250414_144041/generate_data_20250414_144041.py 2>&1 | tee logs/data_generation_20250414_144041.log

# 에러 확인
if [ $? -eq 0 ]; then
    echo "데이터 생성 완료"
else
    echo "데이터 생성 실패. 로그 파일을 확인하세요."
    exit 1
fi
