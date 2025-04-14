# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime, timedelta

# 로깅 설정
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f"data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class DataGenerator:
    def __init__(self, num_logs=200, output_dir="data"):
        self.logger = logging.getLogger(__name__)
        self.num_logs = num_logs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 시뮬레이션 데이터 풀
        self.common_questions = [
            "How do I reset my password?",
            "What are the system requirements for Product X?",
            "I'm getting a 503 error when trying to connect.",
            "Where can I find the API documentation for user management?",
            "How to configure the notification settings?",
            "Can you explain the pricing model?",
            "How do I integrate your service with Salesforce?",
            "What's the process for upgrading my subscription plan?",
            "Troubleshooting guide for connection issues?",
            "How to add a new user to my team account?",
            "Password reset procedure?", # 유사 질문
            "System specs needed for Product X?", # 유사 질문
            "API docs for creating users?", # 유사 질문
            "Help with 503 connection error.", # 유사 질문
        ]
        self.response_styles = ["detailed", "simple", "casual", "technical"]
        self.mock_doc_ids = [f"doc_{i:03d}" for i in range(1, 21)] + [None] # RAG 문서 ID (None은 검색 실패 의미)

    def _generate_response(self, question, style, doc_id, introduce_error=False):
        """가짜 챗봇 응답 생성 (문제점 시뮬레이션 포함)"""
        base_response = f"Regarding '{question[:30]}...',"

        # 1. 일관성 부족 시뮬레이션 (스타일 변화)
        if style == "detailed":
            response = f"{base_response} Here is a detailed step-by-step guide: Step 1..., Step 2..., Step 3..."
        elif style == "simple":
            response = f"{base_response} You can do this by following these simple steps: [Simplified Instructions]."
        elif style == "casual":
            response = f"{base_response} Hey there! Just head over to [Settings > ...] and you should be good to go!"
        else: # technical
            response = f"{base_response} The required process involves invoking the `resetPassword` method via the User API endpoint. Ensure proper authentication headers are set."

        # RAG 정보 추가 (시뮬레이션)
        if doc_id:
            response += f" (Ref: {doc_id})"
        else:
            response += " (Could not find specific documentation)"

        # 2. 품질 저하 (환각) 시뮬레이션
        if introduce_error:
            error_type = random.choice(["hallucination", "off_topic"])
            if error_type == "hallucination":
                # 문서 내용과 다르거나 없는 내용 생성
                if doc_id:
                     response = f"{base_response} According to {doc_id}, you should perform [Made-up/Incorrect Action]. (Ref: {doc_id})"
                else:
                     response = f"{base_response} I believe the best way is to [Completely Fabricated Method]. (Could not find specific documentation)"
                is_accurate = False
            else: # off_topic
                response = f"{base_response} By the way, have you seen the weather today? It's quite nice. (Ref: {doc_id if doc_id else 'N/A'})"
                is_accurate = False
        else:
            # RAG 검색 실패 시에도 환각 가능성 약간 높임
            if not doc_id and random.random() < 0.15: # 15% 확률로 RAG 실패 시 부정확
                 response = f"{base_response} Generally, you might try [Plausible but potentially incorrect suggestion]. (Could not find specific documentation)"
                 is_accurate = False
            else:
                # 문서가 있거나, RAG 실패했지만 운 좋게 정확한 일반 답변 생성
                is_accurate = True if not introduce_error else False # 명시적 에러 주입 아니면 True

        return response, is_accurate

    def generate_data(self):
        """대화 로그 및 피드백 데이터 생성"""
        try:
            self.logger.info(f"{self.num_logs}개의 대화 로그 생성을 시작합니다.")
            data = []
            current_time = datetime.now()
            session_id_counter = 1
            user_id_pool = [f"user_{i:04d}" for i in range(1, 51)]

            for i in range(self.num_logs):
                session_id = f"session_{session_id_counter:04d}"
                user_id = random.choice(user_id_pool)
                timestamp = current_time - timedelta(minutes=random.randint(1, self.num_logs * 2)) # 시간 역순으로 분포
                question = random.choice(self.common_questions)

                # 문제점 시뮬레이션 요소 결정
                # - 특정 질문에 대해 스타일 다르게 하기 (질문 기반으로 스타일 랜덤 선택)
                # - 가끔 RAG 실패 시뮬레이션 (doc_id=None)
                # - 가끔 의도적으로 에러 주입 (hallucination / off-topic)
                chosen_style = random.choice(self.response_styles)
                chosen_doc_id = random.choice(self.mock_doc_ids)
                force_error = random.random() < 0.15 # 15% 확률로 의도적 에러 주입

                # 유사 질문에 대해 다른 스타일/정확도 부여하여 일관성 문제 시뮬레이션
                if "password reset" in question.lower() or "reset procedure" in question.lower():
                    # 그냥 랜덤 스타일/정확도 유지 (이미 랜덤성 있음)
                    pass
                elif "system requirements" in question.lower() or "system specs" in question.lower():
                    pass # 마찬가지로 랜덤성 유지

                chatbot_response, is_accurate = self._generate_response(
                    question, chosen_style, chosen_doc_id, force_error
                )

                # 사용자 피드백 시뮬레이션 (정확도, 스타일에 영향 받음)
                feedback = 'good'
                feedback_reason_prob = random.random()
                if not is_accurate:
                     # 부정확하면 높은 확률로 'bad'
                     feedback = 'bad' if feedback_reason_prob < 0.85 else 'good' # 15%는 부정확해도 좋다고 잘못 피드백
                elif chosen_style == "casual" and feedback_reason_prob < 0.3:
                     # 캐주얼한 스타일에 30% 확률로 'bad'
                     feedback = 'bad'
                elif chosen_style == "simple" and len(question) > 50 and feedback_reason_prob < 0.2:
                     # 긴 질문에 너무 간단하면 20% 확률로 'bad'
                     feedback = 'bad'
                elif feedback_reason_prob < 0.1: # 10% 확률로 이유없이 'bad'
                    feedback = 'bad'


                data.append({
                    "log_id": f"log_{i:05d}",
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "user_query": question,
                    "retrieved_doc_id": chosen_doc_id,
                    "chatbot_response": chatbot_response,
                    "response_style_simulated": chosen_style, # 시뮬레이션용 내부 필드
                    "is_accurate_simulated": is_accurate,     # 시뮬레이션용 내부 필드
                    "user_feedback": feedback, # 'good' or 'bad'
                    "llm_model_used": "gpt-4-simulated" # 사용된 모델 (시뮬레이션)
                })

                # 세션 ID 가끔 변경
                if random.random() < 0.3:
                    session_id_counter += 1

            df = pd.DataFrame(data)
            self.logger.info(f"{len(df)}개의 대화 로그 생성 완료.")
            return df

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}", exc_info=True)
            raise

    def validate_data(self, df):
        """생성된 데이터 검증"""
        try:
            self.logger.info("데이터 검증 시작")
            if df is None or df.empty:
                raise ValueError("데이터프레임이 비어있습니다.")

            required_columns = [
                "log_id", "session_id", "timestamp", "user_id", "user_query",
                "retrieved_doc_id", "chatbot_response", "user_feedback", "llm_model_used"
                # 시뮬레이션용 컬럼은 검증에서 제외 가능: "response_style_simulated", "is_accurate_simulated"
            ]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"필수 컬럼 누락: {missing_cols}")

            # 데이터 타입 검증 (간단 예시)
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                 self.logger.warning("Timestamp 컬럼이 datetime 타입이 아닙니다. 변환 시도.")
                 try:
                     df['timestamp'] = pd.to_datetime(df['timestamp'])
                 except Exception as e:
                     raise TypeError(f"Timestamp 컬럼을 datetime으로 변환할 수 없습니다: {e}")


            if not pd.api.types.is_string_dtype(df['user_feedback']):
                raise TypeError("user_feedback 컬럼이 문자열 타입이 아닙니다.")

            # 값 범위 검증
            allowed_feedback = ['good', 'bad']
            invalid_feedback = df[~df['user_feedback'].isin(allowed_feedback)]
            if not invalid_feedback.empty:
                raise ValueError(f"user_feedback 컬럼에 허용되지 않은 값 포함: {invalid_feedback['user_feedback'].unique()}")

            # Null 값 검증 (retrieved_doc_id는 Null 허용)
            critical_cols_for_null = ["log_id", "session_id", "timestamp", "user_id", "user_query", "chatbot_response", "user_feedback", "llm_model_used"]
            null_check = df[critical_cols_for_null].isnull().sum()
            null_cols = null_check[null_check > 0]
            if not null_cols.empty:
                raise ValueError(f"Null 값이 허용되지 않는 컬럼에 Null 포함: \n{null_cols}")

            self.logger.info("데이터 검증 완료 (성공)")
            return True

        except Exception as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}", exc_info=True)
            raise

    def save_data(self, df):
        """데이터프레임을 CSV 파일로 저장"""
        try:
            output_path = os.path.join(self.output_dir, f"chatbot_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self.logger.info(f"데이터를 '{output_path}'에 저장 시작")
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"데이터 저장 완료: {output_path}")
            return output_path

        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}", exc_info=True)
            raise

def main():
    try:
        generator = DataGenerator(num_logs=300, output_dir="generated_interview_data") # 로그 수 및 출력 디렉토리 지정
        generated_data = generator.generate_data()
        if generator.validate_data(generated_data):
            generator.save_data(generated_data)
            logging.info("데이터 생성, 검증 및 저장 완료")
        else:
             logging.error("데이터 검증 실패로 저장을 중단합니다.")

    except Exception as exc:
        logging.error(f"메인 프로그램 실행 중 오류 발생: {str(exc)}", exc_info=True)
        # raise # 필요시 주석 해제하여 프로그램 중단

if __name__ == "__main__":
    main()