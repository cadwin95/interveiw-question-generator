import google.generativeai as genai
import random
import json
import os
import datetime
import argparse
import shutil
import re
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Generative AI 설정
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 사용 모델 변경 (Gemini 2.5 Pro Preview 모델 사용)
model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')

# AI 면접 주제 후보
interview_topics = [
    "비트코인 미래 시세 예측", "자가 발전하는 AI 방향성", 
]

# 직업군은 이름만 유지 - 간소화된 접근
job_positions = {
    "ai_engineer": "AI 엔지니어",
    "data_scientist": "데이터 사이언티스트",
    "backend_developer": "백엔드 개발자",
    "frontend_developer": "프론트엔드 개발자",
    "devops_engineer": "DevOps 엔지니어",
    "security_engineer": "보안 엔지니어",
    "product_manager": "제품 매니저",
    "ux_designer": "UX 디자이너"
}

# 📌 1️⃣ 기획 AI: 문제 초안 생성 (직무 파라미터 추가)
def generate_initial_question(specified_topic=None, job_role="ai_engineer"):
    job_name = job_positions[job_role]
    
    if specified_topic:
        topic = specified_topic
        prompt = f"""
        '{topic}'을 주제로 {job_name} 면접 문제를 만들어주세요.

        요구사항:
        - 30분 내 해결 가능한 실전형 과제
        - 기술적 능력과 사고 과정 평가 가능
        - 주제와 직무를 연결하는 참신한 관점

        출력 형식:
        ### 문제 제목
        [간결한 제목 (10자 이내)]

        ### 문제 설명
        [상세 설명]
        """
    else:
        topic = None
        prompt = f"""
        {job_name} 면접 문제를 만들어주세요.

        요구사항:
        - 30분 내 해결 가능한 실전형 과제
        - 기술적 능력과 사고 과정 평가 가능
        - 직무와 관련된 참신한 관점

        출력 형식:
        ### 문제 제목
        [간결한 제목 (10자 이내)]

        ### 문제 설명
        [상세 설명]
        """
    
    response = model.generate_content(prompt)
    return topic, response.text

# �� 2️⃣ 논리 검증 AI (직무 정보 추가)
def validate_question(question, topic, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제의 논리적 오류를 검토하고 개선할 부분을 제안하세요.

    **면접 문제:**  
    {question}

    **검토 기준:**  
    - 문제의 논리적 일관성이 있는가?  
    - 모호한 부분이 있는가?  
    - 해결을 위해 필요한 정보가 충분한가?
    - {topic if topic else "주제"}에서 벗어나지 않았는가?
    - {job_name} 직무에 적합한 문제인가?

    **출력 형식:**
    ### 검토 결과
    [전체적인 검토 결과를 요약하세요]

    ### 개선 사항
    - [개선 사항 1]
    - [개선 사항 2]
    - [개선 사항 3]

    ### 수정된 문제
    [검토 결과를 반영하여 수정된 문제를 작성하세요]
    """
    response = model.generate_content(prompt)
    return response.text

# 📌 3️⃣ 난이도 조정 AI: 문제 난이도 평가 및 조정
def adjust_difficulty(question, topic, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제의 난이도를 평가하고, 적절한 수준으로 조정하세요.

    **면접 문제:**  
    {question}

    **조정 기준:**  
    - 너무 쉬운 경우: 추가 도전 과제를 제안  
    - 너무 어려운 경우: 해결 범위를 좁히거나, 필수 요구사항을 줄이기  
    - 30분 내 해결 가능하도록 문제 조정
    - {topic if topic else "주제"}에서 벗어나지 않도록 조정

    **출력 형식:**
    ### 난이도 평가
    [현재 문제의 난이도를 평가하고 조정이 필요한 부분을 설명하세요]

    ### 조정 사항
    - [조정 사항 1]
    - [조정 사항 2]
    - [조정 사항 3]

    ### 조정된 문제
    [난이도 조정을 반영한 수정된 문제를 작성하세요]
    """
    response = model.generate_content(prompt)
    return response.text

# 📌 4️⃣ 창의성 조정 AI: 창의적 문제 변형
def enhance_creativity(question, topic, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제를 더 창의적이고 흥미로운 방식으로 변형하세요.

    **면접 문제:**  
    {question}

    **변형 기준:**  
    - AI 엔지니어가 새로운 접근 방식을 고민하도록 유도하세요.
    - 기존 문제와 차별화된 요소를 추가하되, 해결에 필요한 시간이 1~2시간을 넘지 않도록 유지하세요.
    - 추가 요소가 문제의 핵심을 흐리지 않도록 하세요.
    - {topic if topic else "주제"}에서 벗어나지 않도록 조정

    **출력 형식:**
    ### 창의성 강화 방안
    [문제를 더 창의적으로 만들기 위한 방안을 설명하세요]

    ### 추가된 요소
    - [추가된 요소 1]
    - [추가된 요소 2]
    - [추가된 요소 3]

    ### 변형된 문제
    [창의성을 강화한 수정된 문제를 작성하세요]
    """
    response = model.generate_content(prompt)
    return response.text

# 📌 5️⃣ 난해함 조정 AI: 문제 복잡성 증가
def enhance_complexity(question, topic, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제를 더욱 난해하게 만들어 주세요.

    **면접 문제:**  
    {question}

    **변형 기준:**  
    - 논리적 장애물을 추가하되, 30분 내에 해결 가능한 수준으로 유지하세요.
    - 불가능해보이는 문제면 더 좋습니다.
    - {topic if topic else "주제"}에서 벗어나지 않도록 조정

    **출력 형식:**
    ### 복잡성 강화 방안
    [문제를 더 복잡하게 만들기 위한 방안을 설명하세요]

    ### 추가된 복잡성
    - [추가된 복잡성 1]
    - [추가된 복잡성 2]
    - [추가된 복잡성 3]

    ### 변형된 문제
    [복잡성을 강화한 수정된 문제를 작성하세요]
    """
    response = model.generate_content(prompt)
    return response.text

def simplify_question(question, topic, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제를 더 단순하고 직관적이게 만들어 주세요.

    **면접 문제:**  
    {question}

    **단순화 기준:**
    - 질문의 문구를 단순화하되, 문제의 핵심을 흐리지 않도록 하세요.
    - 불필요한 복잡성을 제거하세요.
    - 데이터는 잘 알려진 데이터를 사용하거나 정확한 데이터 생성이 가능한 쪽으로 사용하세요.
    - {topic if topic else "주제"}에서 벗어나지 않도록 조정

    **출력 형식:**
    ### 단순화 방안
    [문제를 단순화하기 위한 방안을 설명하세요]

    ### 제거된 복잡성
    - [제거된 복잡성 1]
    - [제거된 복잡성 2]
    - [제거된 복잡성 3]

    ### 단순화된 문제
    [단순화된 수정된 문제를 작성하세요]
    """
    response = model.generate_content(prompt)
    return response.text

# 📌 6️⃣ 최종 검토 AI: 최적 문제 선정 및 평가 기준 설정
def finalize_question(question, topic, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제를 최종적으로 정리해주세요.
    **중요: 출력 형식을 정확히 따라야 합니다. 형식이 다르면 오류가 발생합니다.**

    **면접 문제:**  
    {question}

    **최종 검토 기준:**  
    - 창의적이고 차별화된 문제인가?  
    - AI 엔지니어의 기술력을 정확히 평가할 수 있는가?  
    - 문제에 논리적 오류가 있는가?
    - 관련 환경 및 데이터를 준비하는데 큰 어려움이 없는가?
    - {topic if topic else "주제"}에서 벗어나지 않도록 조정

    **출력 형식 (이 형식을 정확히 따라야 합니다):**
    ### 문제 제목
    [간결한 제목 (10자 이내)]

    ### 문제 설명
    **상황:**
    [문제 상황 설명]

    **요구사항:**
    1. [요구사항 1]
    2. [요구사항 2]
    3. [요구사항 3]

    ### 평가 포인트
    - [평가 포인트 1]
    - [평가 포인트 2]
    - [평가 포인트 3]

    ### 제약 조건
    - [제약 조건 1]
    - [제약 조건 2]
    - [제약 조건 3]
    """
    response = model.generate_content(prompt)
    return response.text

def generate_data_script(question, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제에 대한 데이터 생성 코드를 만들어주세요.
    **중요: 
    1. 코드 블록 표시(```)를 절대 포함하지 마세요.
    2. 순수 Python 코드만 생성하세요.
    3. 출력 형식을 정확히 따라야 합니다.**

    **면접 문제:**
    {question}

    **코드 생성 기준:**
    - 문제의 핵심 요구사항을 파악하여 필요한 데이터를 생성합니다.
    - 데이터는 문제 해결에 필요한 최소한의 정보만 포함합니다.
    - 코드는 간단하고 이해하기 쉬워야 합니다.
    - 필요한 패키지는 최소한으로 사용합니다.
    - 에러 처리를 포함합니다.
    - 데이터 검증 로직을 포함합니다.
    - 로깅 기능을 포함합니다.

    **출력 형식 (이 형식을 정확히 따라야 합니다):**
    # 필요한 패키지
    import pandas as pd
    import numpy as np
    import random
    import logging
    import os
    from datetime import datetime

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_generation.log'),
            logging.StreamHandler()
        ]
    )

    class DataGenerator:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            
        def generate_data(self):
            try:
                self.logger.info("데이터 생성 시작")
                # 데이터 생성 로직
                pass
            except Exception as exc:
                self.logger.error(f"데이터 생성 중 오류 발생: {{str(exc)}}")
                raise
            
        def validate_data(self, data):
            try:
                self.logger.info("데이터 검증 시작")
                # 데이터 검증 로직
                pass
            except Exception as exc:
                self.logger.error(f"데이터 검증 중 오류 발생: {{str(exc)}}")
                raise
            
        def save_data(self, data):
            try:
                self.logger.info("데이터 저장 시작")
                # 데이터 저장 로직
                pass
            except Exception as exc:
                self.logger.error(f"데이터 저장 중 오류 발생: {{str(exc)}}")
                raise

    def main():
        try:
            generator = DataGenerator()
            data = generator.generate_data()
            generator.validate_data(data)
            generator.save_data(data)
            logging.info("데이터 생성 및 저장 완료")
        except Exception as exc:
            logging.error(f"프로그램 실행 중 오류 발생: {{str(exc)}}")
            raise

    if __name__ == "__main__":
        main()
    """
    response = model.generate_content(prompt)
    # 코드 블록 표시 제거
    code = response.text.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.endswith("```"):
        code = code[:-3]
    # 추가: 코드 블록 표시가 중간에 있는 경우도 제거
    code = code.replace("```python", "").replace("```", "")
    return code.strip()

def create_auto_run_script(question_dir, timestamp):
    script_content = f"""#!/bin/bash

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 로그 디렉토리 생성
mkdir -p logs

# 데이터 생성 스크립트 실행
echo "데이터 생성 시작..."
python {question_dir}/generate_data_{timestamp}.py 2>&1 | tee logs/data_generation_{timestamp}.log

# 에러 확인
if [ $? -eq 0 ]; then
    echo "데이터 생성 완료"
else
    echo "데이터 생성 실패. 로그 파일을 확인하세요."
    exit 1
fi
"""
    
    script_path = os.path.join(question_dir, f"run_data_generation_{timestamp}.sh")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 실행 권한 부여
    os.chmod(script_path, 0o755)
    return script_path

def generate_interviewer_guide(question, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제를 바탕으로 면접관이 숙지해야 할 핵심 사항을 정리해주세요.

    **면접 문제:**  
    {question}

    **요구사항:**  
    - 문제 해결에 필요한 핵심 기술 및 개념  
    - 후보자의 접근 방식 평가 시 주의사항
    - 면접관의 질문 포인트

    **출력 형식:**
    ### 핵심 기술 및 개념
    - [핵심 기술 1]
    - [핵심 기술 2]
    - [핵심 기술 3]

    ### 주의사항
    - [주의사항 1]
    - [주의사항 2]
    - [주의사항 3]

    ### 질문 가이드
    - [질문 1]
    - [질문 2]
    - [질문 3]
    """
    response = model.generate_content(prompt)
    return response.text

# 🎯 AI 면접 문제 생성 및 조정 실행 (직무 파라미터 추가)
def ai_interview_question_generator(test_type="creative", save_results=True, topic=None, job_role="ai_engineer"):
    # 결과를 저장할 딕셔너리 생성
    results = {}
    job_name = job_positions[job_role]
    
    print("\n" + "="*80)
    print(f"# {job_name} 면접 문제 생성 프로세스")
    print("="*80 + "\n")
    
    # 1. 초기 문제 생성
    print("## 1. 문제 기획")
    print("-"*40)
    selected_topic, initial_question = generate_initial_question(specified_topic=topic, job_role=job_role)
    if selected_topic:
        print(f"**주제:** {selected_topic}\n")
    print(initial_question)
    results["topic"] = selected_topic
    results["job_role"] = job_role
    results["job_name"] = job_name
    results["initial_question"] = initial_question

    # 2. 문제 검증 및 개선
    print("\n## 2. 문제 검증 및 개선")
    print("-"*40)
    validated_question = validate_question(initial_question, selected_topic or "", job_name)
    print(validated_question)
    results["validated_question"] = validated_question

    # 3. 난이도 조정
    print("\n## 3. 난이도 조정")
    print("-"*40)
    adjusted_question = adjust_difficulty(validated_question, selected_topic or "", job_name)
    print(adjusted_question)
    results["adjusted_question"] = adjusted_question

    # 4. 최종 문제 확정
    print("\n## 4. 최종 문제 확정")
    print("-"*40)
    final_question = finalize_question(adjusted_question, selected_topic or "", job_name)
    print(final_question)
    results["final_question"] = final_question

    # 5. 데이터 생성 코드
    print("\n## 5. 데이터 생성 코드")
    print("-"*40)
    data_generation = generate_data_script(final_question, job_name)
    print("```python")
    print(data_generation)
    print("```")
    results["data_generation"] = data_generation

    # 6. 면접관 가이드
    print("\n## 6. 면접관 가이드")
    print("-"*40)
    interviewer_guide = generate_interviewer_guide(final_question, job_name)
    print(interviewer_guide)
    results["interviewer_guide"] = interviewer_guide
    
    # 타임스탬프 추가
    results["timestamp"] = datetime.datetime.now().isoformat()
    results["test_type"] = test_type
    
    # 결과 저장
    if save_results:
        save_results_to_files(results)
    
    return final_question, results

def save_results_to_files(results):
    # 결과 저장 디렉토리 생성
    output_dir = "interview_questions"
    os.makedirs(output_dir, exist_ok=True)
    
    # 타임스탬프 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 면접 문제 제목 추출
    final_question = results["final_question"]
    lines = final_question.split("\n")
    question_title = ""
    for i, line in enumerate(lines):
        if line.strip() == "### 문제 제목":
            if i + 1 < len(lines):
                question_title = lines[i + 1].strip()
                break
                    
    if not question_title:
        question_title = f"{results['job_name']}_면접_문제_{timestamp}"
    
    # 파일 시스템에 안전한 이름으로 변환
    question_title = question_title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    # 문제별 폴더 생성 (문제 제목 사용)
    question_dir = os.path.join(output_dir, question_title)
    os.makedirs(question_dir, exist_ok=True)
    
    # 데이터 생성 코드를 Python 파일로 저장
    if "data_generation" in results and results["data_generation"]:
        py_path = os.path.join(question_dir, f"generate_data.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(results["data_generation"])
        
        # 자동 실행 스크립트 생성
        run_script_path = create_auto_run_script(question_dir, timestamp)
    
    # 최종 문제 마크다운 파일 생성
    question_md_path = os.path.join(question_dir, "question.md")
    with open(question_md_path, "w", encoding="utf-8") as f:
        # [title] 토큰을 HTML 주석으로 감싸서 저장
        f.write("<!-- [title]" + question_title + " -->\n\n")
        f.write(results["final_question"] + "\n\n")
        
        # 데이터 생성 코드 참조
        if "data_generation" in results and results["data_generation"]:
            f.write("## 데이터 생성\n\n")
            f.write("데이터 생성 코드는 `generate_data.py` 파일에 저장되어 있습니다.\n\n")
            f.write("### 실행 방법\n")
            f.write("다음 방법 중 하나를 선택하여 실행할 수 있습니다:\n\n")
            f.write("1. 자동 실행 스크립트 사용:\n")
            f.write("```bash\n")
            f.write(f"./run_data_generation_{timestamp}.sh\n")
            f.write("```\n\n")
            f.write("2. Python 스크립트 직접 실행:\n")
            f.write("```bash\n")
            f.write("python generate_data.py\n")
            f.write("```\n\n")
            f.write("### 로그 확인\n")
            f.write("실행 로그는 `logs/data_generation.log` 파일에서 확인할 수 있습니다.\n\n")
    
    # 면접관 가이드 마크다운 파일 생성
    if "interviewer_guide" in results and results["interviewer_guide"]:
        guide_md_path = os.path.join(question_dir, "interviewer_guide.md")
        with open(guide_md_path, "w", encoding="utf-8") as f:
            f.write(f"# {question_title} - 면접관 가이드\n\n")
            f.write(results["interviewer_guide"] + "\n\n")
    
    # 압축 파일 생성 (같은 폴더 안에)
    zip_filename = f"{question_title}.zip"
    zip_path = os.path.join(question_dir, zip_filename)
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', question_dir)
    
    print("\n" + "="*80)
    print("## 결과 저장 완료")
    print(f"- 저장 위치: {question_dir}")
    print(f"- 문제 파일: {question_md_path}")
    if "data_generation" in results and results["data_generation"]:
        print(f"- 데이터 생성 코드: {py_path}")
        print(f"- 실행 스크립트: {run_script_path}")
    if "interviewer_guide" in results and results["interviewer_guide"]:
        print(f"- 면접관 가이드: {guide_md_path}")
    print(f"- 압축 파일: {zip_path}")
    print("="*80 + "\n")

# 사용 가능한 직업군 목록 출력 함수
def list_available_jobs():
    print("사용 가능한 직업군 목록:")
    for i, (job_key, job_name) in enumerate(job_positions.items(), 1):
        print(f"{i}. {job_name} ({job_key})")
def list_available_topics():
    print("사용 가능한 주제 목록:")
    for i, topic in enumerate(interview_topics, 1):
        print(f"{i}. {topic}")

# 명령줄 인수 처리에 직업군 옵션 추가
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI 면접 문제 생성기')
    
    # 기존 옵션
    parser.add_argument('--test-type', choices=['creative', 'complex', 'default'], 
                       default='creative', help='문제 생성 모드 (creative/complex/default)')
    parser.add_argument('--no-save', action='store_true', help='결과를 파일로 저장하지 않음')
    
    # 주제 관련 옵션
    topic_group = parser.add_mutually_exclusive_group()
    topic_group.add_argument('--topic', type=str, help='특정 주제 지정 (예: 머신러닝)')
    topic_group.add_argument('--topic-id', type=int, help='주제 ID로 선택 (목록 보기: --list-topics)')
    topic_group.add_argument('--random-topic', action='store_true', help='랜덤 주제 선택 (기본값)')
    topic_group.add_argument('--list-topics', action='store_true', help='사용 가능한 주제 목록 표시')
    
    # 직업군 관련 옵션 추가
    job_group = parser.add_mutually_exclusive_group()
    job_group.add_argument('--job', type=str, default="ai_engineer", 
                          help='직업군 지정 (기본값: ai_engineer)')
    job_group.add_argument('--job-id', type=int, 
                          help='직업군 ID로 선택 (목록 보기: --list-jobs)')
    job_group.add_argument('--list-jobs', action='store_true', 
                          help='사용 가능한 직업군 목록 표시')
    
    args = parser.parse_args()
    
    # 직업군 목록 표시
    if args.list_jobs:
        list_available_jobs()
        exit(0)
    
    # 주제 목록 표시
    if args.list_topics:
        list_available_topics()
        exit(0)
    
    # 주제 선택 로직 (기존 코드와 같음)
    selected_topic = None
    if args.topic:
        selected_topic = args.topic
        if selected_topic not in interview_topics:
            print(f"경고: '{selected_topic}'은 기본 주제 목록에 없습니다. 새로운 주제로 진행합니다.")
    elif args.topic_id:
        if 1 <= args.topic_id <= len(interview_topics):
            selected_topic = interview_topics[args.topic_id - 1]
        else:
            print(f"오류: 유효하지 않은 주제 ID입니다. 1부터 {len(interview_topics)}까지의 숫자를 입력하세요.")
            exit(1)
    
    # 직업군 선택 로직
    selected_job = "ai_engineer"  # 기본값
    if args.job:
        if args.job in job_positions:
            selected_job = args.job
        else:
            print(f"경고: '{args.job}'은 지원하지 않는 직업군입니다. 기본값(AI 엔지니어)으로 진행합니다.")
    elif args.job_id:
        job_keys = list(job_positions.keys())
        if 1 <= args.job_id <= len(job_keys):
            selected_job = job_keys[args.job_id - 1]
        else:
            print(f"오류: 유효하지 않은 직업군 ID입니다. 1부터 {len(job_keys)}까지의 숫자를 입력하세요.")
            exit(1)
    
    # 면접 문제 생성 실행
    finalized_question, results = ai_interview_question_generator(
        test_type=args.test_type, 
        save_results=not args.no_save,
        topic=selected_topic,
        job_role=selected_job
    )
    
    # 최종 문제 출력
    print(f"\n⚖️ **최종 {job_positions[selected_job]} 면접 문제:**\n", finalized_question)

def read_question_file(question_dir):
    """문제 파일을 읽고 내용을 반환합니다."""
    try:
        question_file = os.path.join(question_dir, "question.md")
        if not os.path.exists(question_file):
            raise FileNotFoundError(f"문제 파일이 없습니다: {question_file}")
            
        with open(question_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # 문제 제목 찾기
        question_title = None
        for i, line in enumerate(lines):
            if line.strip() == "### 문제 제목":
                if i + 1 < len(lines):
                    question_title = lines[i + 1].strip()
                    break
                    
        if not question_title:
            raise ValueError("문제 제목을 찾을 수 없습니다.")
            
        # 문제 설명 찾기
        question_content = ""
        start_reading = False
        for line in lines:
            if line.strip() == "### 문제 설명":
                start_reading = True
                continue
            if start_reading and line.strip().startswith("###"):
                break
            if start_reading:
                question_content += line
                
        if not question_content.strip():
            raise ValueError("문제 설명을 찾을 수 없습니다.")
            
        return question_title, question_content.strip()
        
    except Exception as exc:
        logging.error(f"문제 파일 읽기 중 오류 발생: {str(exc)}")
        raise
