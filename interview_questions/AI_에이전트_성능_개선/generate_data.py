# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime, timedelta
import uuid

# 로깅 설정
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f'data_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class DataGenerator:
    """
    AI 에이전트 성능 관련 데이터를 생성하는 클래스.
    사용자 피드백 로그, 상호작용 메트릭을 시뮬레이션합니다.
    """
    def __init__(self, num_records=1000, start_date_str='2024-01-01', days_range=90):
        """
        데이터 생성기 초기화

        Args:
            num_records (int): 생성할 총 레코드(상호작용) 수.
            start_date_str (str): 데이터 생성 시작 날짜 (YYYY-MM-DD).
            days_range (int): 시작 날짜로부터 며칠간의 데이터를 생성할지 범위.
        """
        self.logger = logging.getLogger(__name__)
        self.num_records = num_records
        try:
            self.start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError as e:
            self.logger.error(f"잘못된 날짜 형식입니다: {start_date_str}. 'YYYY-MM-DD' 형식을 사용하세요. 기본값 2024-01-01로 설정합니다.")
            self.start_date = datetime(2024, 1, 1)
        self.end_date = self.start_date + timedelta(days=days_range)
        self.output_dir = 'generated_data'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"출력 디렉토리 생성: {self.output_dir}")

    def _generate_timestamps(self):
        """지정된 기간 내에서 무작위 타임스탬프 목록 생성"""
        timestamps = []
        for _ in range(self.num_records):
            random_seconds = random.uniform(0, (self.end_date - self.start_date).total_seconds())
            timestamps.append(self.start_date + timedelta(seconds=random_seconds))
        return sorted(timestamps)

    def _generate_complaint_type(self):
        """문제 설명에 제시된 비율에 따라 불만 유형 생성"""
        # 불만 비율: 부정확 40%, 느림 30%, 맥락 부족 20%, 기타 10%
        # 실제로는 모든 상호작용에 불만이 있는 것은 아니므로 '불만 없음' 비율 추가
        no_complaint_rate = 0.6 # 예: 60%는 불만 없음
        inaccurate_rate = (1 - no_complaint_rate) * 0.40
        slow_rate = (1 - no_complaint_rate) * 0.30
        no_context_rate = (1 - no_complaint_rate) * 0.20
        other_rate = (1 - no_complaint_rate) * 0.10

        # 합계가 1이 되도록 조정 (부동 소수점 오차 방지)
        total_rate = no_complaint_rate + inaccurate_rate + slow_rate + no_context_rate + other_rate
        probabilities = [
            no_complaint_rate / total_rate,
            inaccurate_rate / total_rate,
            slow_rate / total_rate,
            no_context_rate / total_rate,
            other_rate / total_rate
        ]

        complaint_types = ['None', 'Inaccurate', 'Slow', 'No_Context', 'Other']
        return np.random.choice(complaint_types, p=probabilities)

    def _generate_response_time(self, complaint_type, rag_used):
        """불만 유형 및 RAG 사용 여부에 따라 응답 시간(ms) 시뮬레이션"""
        base_time = np.random.normal(loc=1500, scale=500) # 기본 응답 시간 (평균 1.5초)

        if rag_used:
            base_time += np.random.normal(loc=1000, scale=300) # RAG 사용 시 추가 시간

        if complaint_type == 'Slow':
            # '느림' 불만 시 훨씬 긴 시간 추가
            base_time += np.random.normal(loc=5000, scale=1500)
        elif complaint_type == 'Inaccurate' and rag_used:
             # 부정확하고 RAG 사용 시 약간 더 긴 시간 (잘못된 검색 등)
            base_time += np.random.normal(loc=500, scale=200)

        # 최소 응답 시간 보장 및 음수 방지
        return max(500, int(base_time))

    def generate_data(self):
        """
        시뮬레이션된 AI 에이전트 상호작용 데이터 생성.
        타임스탬프, 세션 ID, 불만 유형, 응답 시간 등을 포함.
        """
        try:
            self.logger.info(f"{self.num_records}개 레코드 데이터 생성 시작...")
            data = []
            timestamps = self._generate_timestamps()
            session_context_tracker = {} # 간단한 세션 내 맥락 추적 시뮬레이션

            for i in range(self.num_records):
                timestamp = timestamps[i]
                # 세션 ID: 간단하게 10% 확률로 새 세션 시작 또는 이전 세션 이어가기
                if random.random() < 0.1 or i == 0:
                    session_id = str(uuid.uuid4())
                    session_turn = 1
                    session_context_tracker[session_id] = {'last_interaction_time': timestamp}
                else:
                    # 가장 최근 세션 ID 재사용 (실제로는 더 복잡한 로직 필요)
                    session_id = list(session_context_tracker.keys())[-1]
                    # 시간 간격 확인 (예: 30분 이상 지나면 새 세션으로 간주 - 여기서는 단순화)
                    if (timestamp - session_context_tracker[session_id]['last_interaction_time']).total_seconds() > 1800:
                         session_id = str(uuid.uuid4())
                         session_turn = 1
                         session_context_tracker[session_id] = {'last_interaction_time': timestamp}
                    else:
                        session_turn += 1
                        session_context_tracker[session_id]['last_interaction_time'] = timestamp


                complaint_type = self._generate_complaint_type()

                # 맥락 부족 불만 시뮬레이션: 세션 턴이 1보다 큰데 맥락 부족 불만이 발생할 확률 증가
                if session_turn > 1 and complaint_type == 'None' and random.random() < 0.1: # 10% 확률로 맥락 부족 발생 가능성
                     # 실제 데이터에서는 맥락 부족 판단 로직이 더 복잡할 것임
                     pass # 여기서는 생성된 complaint_type을 그대로 사용

                # RAG 사용 여부 시뮬레이션 (예: 50% 확률로 RAG 사용)
                rag_used = random.choice([True, False])

                # RAG 성공 여부 시뮬레이션 (RAG 사용 시)
                rag_hit = False
                if rag_used:
                    # 부정확 불만이면 RAG 실패(hit=False) 확률 높임
                    if complaint_type == 'Inaccurate':
                        rag_hit = random.random() < 0.2 # 20%만 성공
                    else:
                        rag_hit = random.random() < 0.8 # 80% 성공

                response_time_ms = self._generate_response_time(complaint_type, rag_used)

                # 간단한 사용자 질문/에이전트 응답 예시 (실제 데이터는 훨씬 다양함)
                user_query = f"문의사항_{random.randint(1, 100)}"
                agent_response = f"답변_{random.randint(1, 50)}"
                if complaint_type == 'Inaccurate' and rag_used and not rag_hit:
                    agent_response = "죄송합니다. 관련 정보를 찾지 못했습니다."
                elif complaint_type == 'Inaccurate':
                     agent_response = f"잘못된_답변_{random.randint(1, 20)}"


                data.append({
                    'timestamp': timestamp,
                    'session_id': session_id,
                    'session_turn': session_turn,
                    'user_query': user_query,
                    'agent_response': agent_response,
                    'response_time_ms': response_time_ms,
                    'complaint_type': complaint_type,
                    'rag_used': rag_used,
                    'rag_hit': rag_hit if rag_used else None # RAG 사용 안 했으면 N/A
                })

            df = pd.DataFrame(data)
            self.logger.info(f"데이터 생성 완료. 총 {len(df)}개 레코드 생성됨.")
            return df

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}", exc_info=True)
            raise

    def validate_data(self, df):
        """
        생성된 데이터프레임의 유효성 검사.

        Args:
            df (pd.DataFrame): 검사할 데이터프레임.

        Raises:
            ValueError: 데이터 유효성 검사 실패 시.
        """
        try:
            self.logger.info("데이터 검증 시작...")
            if not isinstance(df, pd.DataFrame):
                raise ValueError("입력 데이터가 Pandas DataFrame이 아닙니다.")

            if df.empty:
                raise ValueError("데이터프레임이 비어 있습니다.")

            required_columns = ['timestamp', 'session_id', 'session_turn', 'user_query',
                                'agent_response', 'response_time_ms', 'complaint_type',
                                'rag_used', 'rag_hit']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"필수 컬럼 누락: {', '.join(missing_cols)}")

            # 데이터 타입 검사
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                raise ValueError("'timestamp' 컬럼이 datetime 타입이 아닙니다.")
            if not pd.api.types.is_integer_dtype(df['session_turn']):
                raise ValueError("'session_turn' 컬럼이 integer 타입이 아닙니다.")
            if not pd.api.types.is_integer_dtype(df['response_time_ms']):
                raise ValueError("'response_time_ms' 컬럼이 integer 타입이 아닙니다.")
            if not pd.api.types.is_bool_dtype(df['rag_used']):
                 # Boolean 컬럼에 None이 포함될 수 있으므로 object 타입일 수 있음
                 if not df['rag_used'].apply(lambda x: isinstance(x, bool)).all():
                     raise ValueError("'rag_used' 컬럼이 boolean 타입이 아닙니다.")
            # rag_hit는 rag_used가 False일 때 None일 수 있으므로 object 타입 검사
            if not df['rag_hit'].apply(lambda x: isinstance(x, bool) or x is None).all():
                raise ValueError("'rag_hit' 컬럼이 boolean 또는 None 타입이 아닙니다.")


            # Null 값 검사 (rag_hit 제외)
            # rag_hit는 rag_used=False일 때 None이 될 수 있으므로 제외
            cols_to_check_null = ['timestamp', 'session_id', 'session_turn', 'user_query',
                                   'agent_response', 'response_time_ms', 'complaint_type', 'rag_used']
            if df[cols_to_check_null].isnull().values.any():
                 null_counts = df[cols_to_check_null].isnull().sum()
                 raise ValueError(f"Null 값이 허용되지 않는 컬럼에 Null 값이 존재합니다: \n{null_counts[null_counts > 0]}")

            # 값 범위/카테고리 검사
            if (df['response_time_ms'] <= 0).any():
                raise ValueError("'response_time_ms'는 양수여야 합니다.")

            expected_complaints = {'None', 'Inaccurate', 'Slow', 'No_Context', 'Other'}
            actual_complaints = set(df['complaint_type'].unique())
            if not actual_complaints.issubset(expected_complaints):
                raise ValueError(f"예상치 못한 'complaint_type' 값 발견: {actual_complaints - expected_complaints}")

            self.logger.info("데이터 검증 성공.")
            return True

        except ValueError as ve:
            self.logger.error(f"데이터 검증 실패: {str(ve)}")
            raise
        except Exception as exc:
            self.logger.error(f"데이터 검증 중 예기치 않은 오류 발생: {str(exc)}", exc_info=True)
            raise

    def save_data(self, df, filename="ai_agent_interactions.csv"):
        """
        데이터프레임을 CSV 파일로 저장.

        Args:
            df (pd.DataFrame): 저장할 데이터프레임.
            filename (str): 저장할 파일 이름.
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            self.logger.info(f"데이터 저장 시작: {filepath}")
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            self.logger.info(f"데이터 성공적으로 저장 완료: {filepath}")

        except IOError as ioe:
            self.logger.error(f"파일 쓰기 오류 발생: {filepath}. 권한 또는 디스크 공간을 확인하세요. {str(ioe)}", exc_info=True)
            raise
        except Exception as exc:
            self.logger.error(f"데이터 저장 중 예기치 않은 오류 발생: {str(exc)}", exc_info=True)
            raise

def main():
    """
    데이터 생성, 검증, 저장 프로세스 실행.
    """
    logging.info("데이터 생성 프로세스 시작")
    try:
        # 데이터 생성기 인스턴스화 (예: 2000개 레코드 생성)
        generator = DataGenerator(num_records=2000, start_date_str='2024-03-01', days_range=60)

        # 데이터 생성
        generated_data = generator.generate_data()

        # 데이터 검증
        generator.validate_data(generated_data)

        # 데이터 저장
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_agent_interactions_{timestamp_str}.csv"
        generator.save_data(generated_data, filename=filename)

        logging.info("데이터 생성, 검증 및 저장 완료")

    except ValueError as ve:
         # 유효성 검사 오류는 이미 로깅되었으므로 추가 로깅 없이 종료
         logging.error(f"데이터 유효성 검사 실패로 프로세스 중단: {str(ve)}")
    except Exception as exc:
        logging.critical(f"프로그램 실행 중 심각한 오류 발생: {str(exc)}", exc_info=True)
        # 필요시 여기서 추가적인 오류 처리/알림 로직 추가 가능
    finally:
        logging.info("데이터 생성 프로세스 종료")

if __name__ == "__main__":
    main()