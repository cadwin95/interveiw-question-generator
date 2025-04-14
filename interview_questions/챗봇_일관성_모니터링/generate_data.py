# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime, timedelta

# 로깅 설정
log_file = 'data_generation.log'
# 로그 파일이 이미 존재하면 삭제 (새 실행마다 새 로그)
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class DataGenerator:
    def __init__(self, num_records=1000, output_dir="data"):
        self.logger = logging.getLogger(__name__)
        self.num_records = num_records
        self.output_dir = output_dir
        self.output_file = os.path.join(self.output_dir, "chatbot_interaction_logs.csv")

        # 데이터 생성을 위한 설정값들
        self.user_ids = [f"user_{i:03}" for i in range(1, 51)]
        self.model_versions = ['v1.0', 'v1.1', 'v1.2'] # v1.1에서 변동성 증가 시나리오

        # 유사 질문 그룹 정의 (질문 원본, 변형들)
        self.question_groups = {
            "QG01": ["상품 재고 문의", "이 상품 재고 있나요?", "재고 확인 부탁드립니다."],
            "QG02": ["배송 기간 문의", "배송 얼마나 걸려요?", "언제쯤 받을 수 있을까요?"],
            "QG03": ["환불 절차 안내", "환불은 어떻게 하나요?", "반품하고 싶어요."],
            "QG04": ["영업 시간 문의", "매장 몇 시까지 하나요?", "운영 시간 알려주세요."],
            "QG05": ["비밀번호 변경 방법", "비밀번호 바꾸려면?", "암호 재설정"]
        }

        # 답변 스타일 및 내용 템플릿 (모델 버전, 질문 그룹별)
        # v1.1 에서 일부 질문 그룹에 대해 답변 스타일/길이 변동성 추가
        self.response_templates = {
            'v1.0': {
                "QG01": ["현재 재고 있습니다.", "문의하신 상품 재고 있습니다."],
                "QG02": ["통상 영업일 기준 2~3일 소요됩니다.", "배송은 평균 2-3일 걸립니다."],
                "QG03": ["홈페이지 '마이페이지 > 주문내역'에서 환불 신청 가능합니다.", "환불 절차는 웹사이트 마이페이지를 참고하세요."],
                "QG04": ["영업 시간은 평일 오전 9시부터 오후 6시까지입니다.", "운영 시간: 09:00 ~ 18:00 (평일)"],
                "QG05": ["로그인 후 '내 정보 > 비밀번호 변경' 메뉴를 이용해주세요.", "비밀번호 변경은 설정 메뉴에서 가능합니다."]
            },
            'v1.1': { # 변동성 증가 버전
                "QG01": ["네, 재고 있습니다. 바로 주문 가능하세요!", "재고 있습니다.", "현재 시스템 확인 결과, 재고 충분합니다."], # 스타일/길이 변동
                "QG02": ["배송은 보통 2~3일 걸리는데, 택배사 사정에 따라 달라질 수 있어요.", "평균 2-3일 소요됩니다.", "지금 주문하시면 모레쯤 도착할 것 같아요!"], # 어조/상세 수준 변동
                "QG03": ["홈페이지 '마이페이지 > 주문내역'에서 환불 신청 가능합니다.", "환불 원하시면 마이페이지에서 직접 신청해주세요.", "간단하게 웹사이트 마이페이지에서 처리 가능해요."], # 약간의 문체 변화
                "QG04": ["영업 시간은 평일 오전 9시부터 오후 6시까지입니다.", "운영 시간: 09:00 ~ 18:00 (평일)"], # 일관성 유지
                "QG05": ["로그인 하신 뒤에 '내 정보' 가셔서 '비밀번호 변경' 누르시면 됩니다. 어렵지 않아요!", "설정 > 비밀번호 변경 메뉴 이용하세요.", "비밀번호 변경은 내 정보 화면에 있습니다."] # 상세 수준/어조 변동
            },
            'v1.2': { # 개선 또는 안정화 버전
                "QG01": ["네, 문의하신 상품은 현재 재고가 충분하여 바로 주문 가능합니다.", "확인 결과, 해당 상품 재고 보유 중입니다."],
                "QG02": ["배송 기간은 통상 영업일 기준 2~3일 소요되며, 지역 및 택배사 사정에 따라 변동될 수 있습니다.", "평균 배송 소요 시간은 영업일 기준 2~3일 입니다."],
                "QG03": ["환불은 홈페이지 로그인 후 [마이페이지 > 주문 관리 > 주문 상세] 화면에서 '환불 요청' 버튼을 통해 신청하실 수 있습니다.", "환불 절차: 로그인 > 마이페이지 > 주문 상세 > 환불 요청"],
                "QG04": ["영업 시간은 평일 오전 9시부터 오후 6시까지입니다. (주말/공휴일 휴무)", "운영 시간 안내: 평일 09:00 - 18:00 (주말 및 공휴일 제외)"],
                "QG05": ["비밀번호 변경은 로그인 후 [내 정보 관리 > 보안 설정 > 비밀번호 변경] 메뉴에서 가능합니다.", "비밀번호 재설정은 '내 정보 관리' 메뉴를 이용해 주십시오."]
            }
        }

    def _get_random_response(self, model_version, question_group_id):
        """주어진 모델 버전과 질문 그룹에 대해 랜덤 답변 선택"""
        try:
            templates = self.response_templates[model_version][question_group_id]
            return random.choice(templates)
        except KeyError:
            # 해당 버전/질문에 정의된 템플릿이 없을 경우 기본 응답
            self.logger.warning(f"응답 템플릿 없음: 버전={model_version}, 그룹={question_group_id}. 기본 응답 사용.")
            return "죄송합니다. 해당 질문에 대한 답변을 준비 중입니다."
        except Exception as e:
            self.logger.error(f"응답 선택 중 오류: {e}")
            return "오류가 발생하여 답변할 수 없습니다."

    def generate_data(self):
        """챗봇 상호작용 로그 데이터 생성"""
        try:
            self.logger.info(f"{self.num_records}개의 데이터 생성 시작")
            data = []
            start_time = datetime.now() - timedelta(days=7) # 최근 7일간 데이터 생성

            # 모델 버전별 데이터 분포 조절 (예: v1.0 -> v1.1 -> v1.2 순서로 배포되었다고 가정)
            version_distribution = {
                'v1.0': int(self.num_records * 0.2),
                'v1.1': int(self.num_records * 0.4),
                'v1.2': int(self.num_records * 0.4)
            }
            # 레코드 수가 정확히 맞지 않을 수 있으므로 마지막 버전에 나머지 할당
            remaining_records = self.num_records - sum(version_distribution.values())
            version_distribution['v1.2'] += remaining_records

            current_time = start_time
            records_generated = 0
            for version, count in version_distribution.items():
                for _ in range(count):
                    user_id = random.choice(self.user_ids)
                    session_id = f"{user_id}_{current_time.strftime('%Y%m%d')}_{random.randint(1, 5)}" # 하루에 여러 세션 가능
                    question_group_id = random.choice(list(self.question_groups.keys()))
                    question = random.choice(self.question_groups[question_group_id])
                    chatbot_model_version = version
                    chatbot_response = self._get_random_response(chatbot_model_version, question_group_id)
                    response_length = len(chatbot_response)
                    # 응답 길이나 복잡성에 따라 응답 시간 가변적으로 시뮬레이션
                    base_response_time = random.uniform(100, 500) # 기본 응답 시간 (ms)
                    complexity_factor = 1 + response_length / 100 # 길이가 길수록 시간 증가
                    response_time_ms = int(base_response_time * complexity_factor + random.uniform(-50, 50))
                    response_time_ms = max(50, response_time_ms) # 최소 응답 시간 보장

                    # 시간 조금씩 증가시키기
                    current_time += timedelta(seconds=random.randint(1, 60))

                    data.append({
                        "timestamp": current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "user_id": user_id,
                        "session_id": session_id,
                        "question_group_id": question_group_id,
                        "question": question,
                        "chatbot_model_version": chatbot_model_version,
                        "chatbot_response": chatbot_response,
                        "response_length": response_length,
                        "response_time_ms": response_time_ms
                    })
                    records_generated += 1

            # 데이터가 부족하게 생성되었을 경우 로깅
            if records_generated < self.num_records:
                 self.logger.warning(f"요청된 {self.num_records}개보다 적은 {records_generated}개의 데이터가 생성되었습니다.")

            df = pd.DataFrame(data)
            self.logger.info(f"{len(df)}개의 데이터 생성 완료")
            return df

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}")
            raise # 오류를 다시 발생시켜 main에서 처리하도록 함

    def validate_data(self, df):
        """생성된 데이터의 유효성 검증"""
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(df, pd.DataFrame):
                raise TypeError("입력 데이터는 Pandas DataFrame이어야 합니다.")

            if df.empty:
                raise ValueError("데이터프레임이 비어 있습니다.")

            required_columns = [
                "timestamp", "user_id", "session_id", "question_group_id",
                "question", "chatbot_model_version", "chatbot_response",
                "response_length", "response_time_ms"
            ]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"필수 컬럼 누락: {missing_cols}")

            # Null 값 체크 (필수 컬럼에 대해서)
            null_check_cols = ["timestamp", "question_group_id", "question", "chatbot_model_version", "chatbot_response"]
            if df[null_check_cols].isnull().any().any():
                null_counts = df[null_check_cols].isnull().sum()
                self.logger.warning(f"일부 필수 컬럼에 Null 값이 존재합니다:\n{null_counts[null_counts > 0]}")
                # raise ValueError("필수 컬럼에 Null 값이 존재합니다.") # 필요시 에러 발생

            # 데이터 타입 체크 (간단 예시)
            if not pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df['timestamp'], errors='coerce')):
                 raise TypeError("timestamp 컬럼 형식이 올바르지 않습니다.")
            if not pd.api.types.is_integer_dtype(df['response_length']):
                 self.logger.warning("response_length 컬럼 타입이 정수형이 아닐 수 있습니다.") # 경고로 처리
            if not pd.api.types.is_integer_dtype(df['response_time_ms']):
                 self.logger.warning("response_time_ms 컬럼 타입이 정수형이 아닐 수 있습니다.") # 경고로 처리

            # 질문 그룹 ID가 정의된 그룹 내에 있는지 확인
            valid_qg_ids = set(self.question_groups.keys())
            invalid_qg_ids = set(df['question_group_id'].unique()) - valid_qg_ids
            if invalid_qg_ids:
                raise ValueError(f"유효하지 않은 question_group_id 발견: {invalid_qg_ids}")

            # 모델 버전이 정의된 버전 내에 있는지 확인
            valid_versions = set(self.model_versions)
            invalid_versions = set(df['chatbot_model_version'].unique()) - valid_versions
            if invalid_versions:
                raise ValueError(f"유효하지 않은 chatbot_model_version 발견: {invalid_versions}")

            self.logger.info("데이터 검증 완료: 유효한 데이터입니다.")
            return True

        except (TypeError, ValueError) as ve:
            self.logger.error(f"데이터 검증 실패: {str(ve)}")
            raise
        except Exception as exc:
            self.logger.error(f"데이터 검증 중 예상치 못한 오류 발생: {str(exc)}")
            raise

    def save_data(self, df):
        """데이터를 CSV 파일로 저장"""
        try:
            self.logger.info(f"데이터 저장 시작: {self.output_file}")
            # 출력 디렉토리 생성 (없으면)
            os.makedirs(self.output_dir, exist_ok=True)

            df.to_csv(self.output_file, index=False, encoding='utf-8-sig') # UTF-8 BOM으로 저장 (Excel 호환성)
            self.logger.info(f"데이터 저장 완료: {self.output_file}")

        except IOError as ioe:
            self.logger.error(f"파일 저장 중 IO 오류 발생 ({self.output_file}): {str(ioe)}")
            raise
        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}")
            raise

def main():
    try:
        # 데이터 생성 파라미터 설정 (예: 레코드 수)
        num_records_to_generate = 2000
        output_directory = "generated_chatbot_data"

        generator = DataGenerator(num_records=num_records_to_generate, output_dir=output_directory)

        # 1. 데이터 생성
        generated_data = generator.generate_data()

        # 2. 데이터 검증
        generator.validate_data(generated_data)

        # 3. 데이터 저장
        generator.save_data(generated_data)

        logging.info("챗봇 상호작용 데이터 생성 및 저장 작업 완료")

    except Exception as exc:
        logging.error(f"메인 프로그램 실행 중 오류 발생: {str(exc)}")
        # 여기서 더 복잡한 오류 처리나 알림 로직을 추가할 수 있습니다.
        # raise # 필요하다면 호출자에게 오류를 다시 전달

if __name__ == "__main__":
    main()