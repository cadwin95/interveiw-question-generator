# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_generation.log', mode='w'), # 'w' 모드로 변경하여 실행 시마다 로그 파일 덮어쓰기
        logging.StreamHandler()
    ]
)

class DataGenerator:
    def __init__(self, num_records=10000, output_dir="chatbot_data"):
        """
        데이터 생성기 초기화

        Args:
            num_records (int): 생성할 총 레코드(상호작용) 수.
            output_dir (str): 생성된 데이터를 저장할 디렉토리 이름.
        """
        self.logger = logging.getLogger(__name__)
        self.num_records = num_records
        self.output_dir = output_dir
        # 비용 모델 파라미터 (예: GPT-4 수준 가정)
        self.cost_per_1k_input_tokens = 0.03 # $ per 1k input tokens
        self.cost_per_1k_output_tokens = 0.06 # $ per 1k output tokens
        self.high_cost_model_name = 'high-perf-llm' # 사용 중인 고성능 모델 이름

    def _generate_single_record(self, record_id, session_tracker):
        """단일 상호작용 레코드를 생성합니다."""
        try:
            # 타임스탬프 생성 (최근 한 달간 데이터)
            timestamp = datetime.now() - timedelta(days=random.uniform(0, 30), hours=random.uniform(0, 24))

            # 사용자 ID 및 세션 ID 생성 (간단한 시뮬레이션)
            user_id = f"user_{random.randint(1, 10000)}" # 실제 10만 MAU 중 일부 시뮬레이션
            # 세션 관리: 사용자당 평균 2개 세션, 세션당 평균 3개 턴 가정
            if user_id not in session_tracker or session_tracker[user_id]['turns'] >= 3:
                session_id = f"session_{record_id}" # 새 세션 시작
                session_tracker[user_id] = {'session_id': session_id, 'turns': 1}
            else:
                session_id = session_tracker[user_id]['session_id']
                session_tracker[user_id]['turns'] += 1

            # 질문 복잡도 및 유형 분포 (문제 설명 기반)
            # 복잡도: Simple (60%), Medium (30%), Complex (10%)
            # 유형: Product Info (40%), FAQ (40%), General (15%), Unsupported (5%)
            complexity_roll = random.random()
            type_roll = random.random()

            if complexity_roll < 0.6:
                query_complexity = "Simple"
                input_tokens = random.randint(10, 40)
                output_tokens = random.randint(30, 80)
            elif complexity_roll < 0.9:
                query_complexity = "Medium"
                input_tokens = random.randint(40, 80)
                output_tokens = random.randint(80, 150)
            else:
                query_complexity = "Complex"
                input_tokens = random.randint(80, 150)
                output_tokens = random.randint(150, 300)

            if type_roll < 0.4:
                query_type = "Product Info"
            elif type_roll < 0.8:
                query_type = "FAQ"
            elif type_roll < 0.95:
                query_type = "General"
                # 일반 질문은 약간 더 길 수 있음
                input_tokens = int(input_tokens * 1.1)
                output_tokens = int(output_tokens * 1.1)
            else:
                query_type = "Unsupported"
                output_tokens = random.randint(20, 50) # 미지원 시 짧은 응답

            # 질문 및 응답 텍스트 (단순 플레이스홀더)
            query = f"Sample {query_complexity} {query_type} query text. ID: {record_id}"
            response = f"Generated response for {query_complexity} query. Type: {query_type}. Tokens: {output_tokens}"

            # API 호출 비용 계산
            api_call_cost = (input_tokens / 1000 * self.cost_per_1k_input_tokens) + \
                            (output_tokens / 1000 * self.cost_per_1k_output_tokens)
            # 소수점 5자리까지 반올림
            api_call_cost = round(api_call_cost, 5)

            # 사용자 만족도 및 해결 여부 (시뮬레이션)
            # 단순/FAQ 질문은 해결률 및 만족도 높음, 복잡/미지원은 낮음
            if query_complexity == "Simple" or query_type == "FAQ":
                user_satisfaction = random.randint(4, 5)
                resolved = random.choices([True, False], weights=[0.9, 0.1], k=1)[0]
            elif query_type == "Unsupported":
                user_satisfaction = random.randint(1, 3)
                resolved = False
            else: # Medium, Complex, General
                user_satisfaction = random.randint(2, 4)
                resolved = random.choices([True, False], weights=[0.7, 0.3], k=1)[0]


            return {
                "timestamp": timestamp,
                "user_id": user_id,
                "session_id": session_id,
                "query": query,
                "query_length_tokens": input_tokens,
                "query_complexity": query_complexity,
                "query_type": query_type,
                "llm_model_used": self.high_cost_model_name,
                "response": response,
                "response_length_tokens": output_tokens,
                "api_call_cost": api_call_cost,
                "user_satisfaction": user_satisfaction, # 예시 KPI
                "resolved": resolved # 예시 KPI
            }
        except Exception as exc:
            self.logger.error(f"Error generating single record: {str(exc)}")
            # 오류 발생 시 빈 딕셔너리나 None을 반환하여 상위에서 처리하도록 할 수 있음
            return None

    def generate_data(self):
        """
        문제 상황을 시뮬레이션하기 위한 상호작용 로그 데이터를 생성합니다.
        """
        try:
            self.logger.info(f"데이터 생성 시작: {self.num_records}개의 레코드 생성 예정")
            data = []
            session_tracker = {} # 간단한 세션 추적용 딕셔너리

            for i in range(self.num_records):
                record = self._generate_single_record(i, session_tracker)
                if record: # 오류 없이 레코드가 생성된 경우에만 추가
                    data.append(record)
                if (i + 1) % (self.num_records // 10) == 0: # 진행 상황 로깅 (10% 마다)
                    self.logger.info(f"{(i + 1) / self.num_records * 100:.0f}% 생성 완료...")

            df = pd.DataFrame(data)
            self.logger.info(f"데이터 생성 완료: 총 {len(df)}개의 레코드 생성됨")
            return df
        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}")
            raise # 오류를 다시 발생시켜 main 함수에서 처리하도록 함

    def validate_data(self, data):
        """
        생성된 데이터의 유효성을 검사합니다.

        Args:
            data (pd.DataFrame): 검증할 데이터프레임.
        """
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("입력 데이터가 Pandas DataFrame이 아닙니다.")

            if data.empty:
                raise ValueError("생성된 데이터프레임이 비어 있습니다.")

            expected_columns = [
                "timestamp", "user_id", "session_id", "query", "query_length_tokens",
                "query_complexity", "query_type", "llm_model_used", "response",
                "response_length_tokens", "api_call_cost", "user_satisfaction", "resolved"
            ]
            missing_columns = [col for col in expected_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼 누락: {', '.join(missing_columns)}")

            # Null 값 검사 (비용, 토큰 수 등 중요 컬럼)
            critical_cols_for_null = ["api_call_cost", "query_length_tokens", "response_length_tokens", "user_id", "session_id"]
            if data[critical_cols_for_null].isnull().any().any():
                 self.logger.warning(f"다음 컬럼에 Null 값이 포함되어 있습니다: {data[critical_cols_for_null].isnull().sum()}")
                 # 상황에 따라 오류로 처리할 수도 있음
                 # raise ValueError(f"다음 중요 컬럼에 Null 값이 존재합니다: {data[critical_cols_for_null].isnull().sum()}")

            # 데이터 타입 및 값 범위 검사 (예시)
            if not pd.api.types.is_numeric_dtype(data['api_call_cost']):
                raise TypeError("api_call_cost 컬럼이 숫자 타입이 아닙니다.")
            if data['api_call_cost'].min() < 0:
                self.logger.warning("api_call_cost에 0 미만 값이 존재합니다.")
            if not pd.api.types.is_integer_dtype(data['user_satisfaction']):
                 # 정수가 아니면 경고만 로깅 (실수형일 수도 있으므로)
                 self.logger.warning("user_satisfaction 컬럼 타입이 정수가 아닙니다.")
            if data['user_satisfaction'].min() < 1 or data['user_satisfaction'].max() > 5:
                 self.logger.warning("user_satisfaction 값이 예상 범위(1-5)를 벗어납니다.")

            self.logger.info("데이터 검증 완료: 기본적인 검증 통과")

        except (TypeError, ValueError) as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}")
            raise
        except Exception as exc:
            self.logger.error(f"데이터 검증 중 예상치 못한 오류 발생: {str(exc)}")
            raise

    def save_data(self, data):
        """
        생성된 데이터를 CSV 파일로 저장합니다.

        Args:
            data (pd.DataFrame): 저장할 데이터프레임.
        """
        try:
            self.logger.info("데이터 저장 시작")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.logger.info(f"'{self.output_dir}' 디렉토리 생성 완료")

            # 현재 시간을 포함한 파일명 생성
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.output_dir, f"chatbot_interactions_{timestamp_str}.csv")

            data.to_csv(file_path, index=False, encoding='utf-8-sig') # UTF-8 BOM 인코딩 사용
            self.logger.info(f"데이터 저장 완료: {file_path}")

        except IOError as exc:
            self.logger.error(f"파일 저장 중 I/O 오류 발생: {str(exc)}")
            raise
        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}")
            raise

def main():
    try:
        # 생성할 레코드 수 및 출력 디렉토리 설정 가능
        generator = DataGenerator(num_records=10000, output_dir="chatbot_cost_data")
        generated_data = generator.generate_data()

        # 생성된 데이터가 비어있지 않은 경우에만 검증 및 저장 수행
        if generated_data is not None and not generated_data.empty:
            generator.validate_data(generated_data)
            generator.save_data(generated_data)
            logging.info("데이터 생성, 검증 및 저장 작업 성공적으로 완료")
        else:
            logging.warning("생성된 데이터가 비어있어 검증 및 저장을 건너<0xEB><0x9A><0x85>니다.")

    except Exception as exc:
        # main 함수 레벨에서 최종 오류 로깅
        logging.error(f"프로그램 실행 중 심각한 오류 발생: {str(exc)}")
        # 필요한 경우 여기서 프로그램을 종료하거나 추가적인 복구 로직 수행
        # raise # 필요시 에러를 다시 발생시켜 외부에서 처리하도록 할 수 있음

if __name__ == "__main__":
    main()