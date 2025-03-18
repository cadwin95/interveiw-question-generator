import google.generativeai as genai
import random
import json
import os
import datetime
import argparse

# Google Generative AI 설정
genai.configure(api_key="your-key")  # 여기에 실제 API 키를 입력하세요

# 사용 모델 변경 (Gemini 모델 사용)
model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

# AI 면접 주제 후보
interview_topics = [
    "비트코인 미래 시세 예측", "자가 발전하는 AI 방향성", 
    "Ai는 일정한 결과물을 낼수가 없는데 어떻게 품질을 일정하게 유지할수 있을까",
    "긴 컨텍스트 Output을 요약해버리는 GPT를 어떻게 방지할까"
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
    else:
        topic = random.choice(interview_topics)
        
    prompt = f"""
    '{topic}'을 주제로 {job_name} 면접 문제를 창의적으로 만들어 주세요.

    **문제 요구사항:**  
    - 문제는 1~2시간 내에 해결할 수 있는 실전형 과제여야 합니다.
    - 지원자의 기술적 능력과 사고 과정을 모두 평가할 수 있어야 합니다.
    - 주제와 직무를 연결하는 참신한 관점의 문제를 찾아보세요.
    - 정형화된 접근을 넘어 창의적 사고를 평가할 수 있어야 합니다.
    """
    
    response = model.generate_content(prompt)
    return topic, response.text

# 📌 2️⃣ 논리 검증 AI (직무 정보 추가)
def validate_question(question, topic, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제의 논리적 오류를 검토하고 개선할 부분을 제안하세요.

    **면접 문제:**  
    {question}

    **검토 기준:**  
    - 문제의 논리적 일관성이 있는가?  
    - 모호한 부분이 있는가?  
    - 해결을 위해 필요한 정보가 충분한가?
    - {topic} 주제에서 벗어나지 않았는가?
    - {job_name} 직무에 적합한 문제인가?
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
    - 1~2시간 내 해결 가능하도록 문제 조정
    - {topic} 주제에서 벗어나지 않도록 조정
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
    - {topic} 주제에서 벗어나지 않도록 조정
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
    - 논리적 장애물을 추가하되, 1~2시간 내에 해결 가능한 수준으로 유지하세요.
    - 불가능해보이는 문제면 더 좋습니다.
    - {topic} 주제에서 벗어나지 않도록 조정
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
    - {topic} 주제에서 벗어나지 않도록 조정
    """
    response = model.generate_content(prompt)
    return response.text

# 📌 6️⃣ 최종 검토 AI: 최적 문제 선정 및 평가 기준 설정
def finalize_question(question, topic, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제의 창의성, 평가 기준, 실용성을 바탕으로 최종 문제를 만들어 주세요. 

    **면접 문제:**  
    {question}

    **최종 검토 기준:**  
    - 창의적이고 차별화된 문제인가?  
    - AI 엔지니어의 기술력을 정확히 평가할 수 있는가?  
    - 문제에 논리적 오류가 있는가?
    - 평가 기준(채점 방식)이 명확한가?
    - 관련 환경 및 데이터를 준비하는데 큰 어려움이 없는가?
    - {topic} 주제에서 벗어나지 않도록 조정
    """
    response = model.generate_content(prompt)
    return response.text

def data_generate(question, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제에 대한 환경 구축 및 데이터셋을 생성하는 코드를 생성해서 주세요.

    **면접 문제:**
    {question}

    **코드 생성 기준:**
    - 데이터셋은 문제를 해결하는데 필요한 데이터셋이어야 합니다.
    - 환경은 문제를 해결하는데 필요한 환경이어야 합니다.
    """
    response = model.generate_content(prompt)
    return response.text

def generate_interviewer_guide(question, job_name):
    prompt = f"""
    다음 {job_name} 면접 문제를 바탕으로 면접관이 숙지해야 할 주요 지식, 개념 및 평가 포인트를 정리해 주세요.

    **면접 문제:**  
    {question}

    **요구사항:**  
    - 문제 해결에 필요한 핵심 기술 및 개념  
    - 후보자의 접근 방식 평가 시 주의해야 할 점  
    - 면접관이 확인해야 할 추가적인 고려 사항
    - 면접관이 평가할 포인트
    - 면접관의 질문 포인트
    """
    response = model.generate_content(prompt)
    return response.text

# 📌 AI 면접 문제 생성 및 조정 실행 (직무 파라미터 추가)
def ai_interview_question_generator(test_type="creative", save_results=True, topic=None, job_role="ai_engineer"):
    # 결과를 저장할 딕셔너리 생성
    results = {}
    job_name = job_positions[job_role]
    
    print(f"🎯 **1. {job_name} 면접 문제 기획:**")
    selected_topic, initial_question = generate_initial_question(specified_topic=topic, job_role=job_role)
    print(f"📌 주제: {selected_topic}\n")
    print(initial_question)
    results["topic"] = selected_topic
    results["job_role"] = job_role
    results["job_name"] = job_name
    results["initial_question"] = initial_question

    print(f"\n🧠 **2. 논리 검증:**")
    validated_question = validate_question(initial_question, selected_topic, job_name)
    print(validated_question)
    results["validated_question"] = validated_question

    print(f"\n📊 **3. 난이도 조정:**")
    adjusted_question = adjust_difficulty(validated_question, selected_topic, job_name)
    print(adjusted_question)
    results["adjusted_question"] = adjusted_question

    print(f"\n🎨 **4. 창의성 강화:**")
    creative_question = enhance_creativity(adjusted_question, selected_topic, job_name)
    print(creative_question)
    results["creative_question"] = creative_question

    print(f"\n🏆 **5. 난해한 문제 강화:**")
    enhanced_question = enhance_complexity(creative_question, selected_topic, job_name)
    print(enhanced_question)
    results["enhanced_question"] = enhanced_question

    print(f"\n🔍 **6. 문항의 단순화:**")
    simplified_question = simplify_question(enhanced_question, selected_topic, job_name)
    print(simplified_question)
    results["simplified_question"] = simplified_question

    print(f"\n🔍 **7. 최종 검토 및 확정:**")
    final_question = finalize_question(enhanced_question, selected_topic, job_name)
    print(final_question)
    results["final_question"] = final_question

    print(f"\n🔍 **8. 데이터 검색:**")
    data_generation = data_generate(final_question, job_name)
    print(data_generation)
    results["data_generation"] = data_generation

    print(f"\n🔍 **9. 면접관 가이드라인:**")
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

# 파일 저장 함수 수정 - 직무 정보 포함
def save_results_to_files(results):
    # 결과 저장 디렉토리 생성
    output_dir = "interview_questions"
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성 (주제, 직무, 타임스탬프 사용)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    topic = results["topic"].replace(" ", "_")
    job = results["job_role"]
    base_filename = f"{job}_{topic}_{timestamp}"
    
    # JSON 파일로 저장
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 마크다운 보고서 생성
    md_path = os.path.join(output_dir, f"{base_filename}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {results['job_name']} 면접 문제 생성 보고서: {results['topic']}\n\n")
        f.write(f"생성 일시: {results['timestamp']}\n")
        f.write(f"모드: {results['test_type']}\n\n")
        
        f.write("## 1. 초기 문제 기획\n\n")
        f.write(results["initial_question"] + "\n\n")
        
        f.write("## 2. 논리 검증\n\n")
        f.write(results["validated_question"] + "\n\n")
        
        f.write("## 3. 난이도 조정\n\n")
        f.write(results["adjusted_question"] + "\n\n")
        
        f.write("## 4. 창의성 강화\n\n")
        f.write(results["creative_question"] + "\n\n")
        
        f.write("## 5. 난해한 문제 강화\n\n")
        f.write(results["enhanced_question"] + "\n\n")
        
        f.write("## 6. 문항 단순화\n\n")
        f.write(results["simplified_question"] + "\n\n")
        
        f.write("## 7. 최종 문제\n\n")
        f.write(results["final_question"] + "\n\n")
        
        f.write("## 8. 데이터 생성 코드\n\n")
        f.write("```python\n" + results["data_generation"] + "\n```\n\n")
        
        f.write("## 9. 면접관 가이드\n\n")
        f.write(results["interviewer_guide"] + "\n\n")
    
    print(f"\n💾 결과가 저장되었습니다:")
    print(f"- JSON: {json_path}")
    print(f"- 마크다운: {md_path}")

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
