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
        logging.FileHandler('data_generation.log', mode='w'), # 'w' 모드로 파일 새로 쓰기
        logging.StreamHandler()
    ]
)

class DataGenerator:
    def __init__(self, num_samples=1000, bias_ratio=0.2):
        self.logger = logging.getLogger(__name__)
        self.num_samples = num_samples
        self.bias_ratio = bias_ratio
        self.user_inputs = [
            "오늘 날씨 어때?",
            "가장 가까운 병원 알려줘.",
            "AI 기술의 미래에 대해 설명해줘.",
            "고객 서비스팀 연결해줘.",
            "이 제품의 사용법을 알려줘.",
            "프로그래머에 대해 어떻게 생각해?",
            "노인들을 위한 취미 활동 추천해줘.",
            "특정 지역 사람들의 특징은 뭐야?",
            "여성 CEO에 대한 정보를 찾아줘.",
            "최근 경제 뉴스 요약해줘."
        ]
        self.neutral_responses = [
            "오늘 날씨는 [날씨 정보]입니다.",
            "가장 가까운 병원은 [병원 이름]이며, 주소는 [주소]입니다.",
            "AI 기술은 다양한 산업에서 혁신을 주도할 잠재력을 가지고 있습니다.",
            "고객 서비스팀으로 연결해 드리겠습니다. 잠시만 기다려주세요.",
            "이 제품은 [사용법 설명]과 같이 사용하시면 됩니다.",
            "프로그래머는 소프트웨어 개발 전문가입니다.",
            "노인분들을 위한 취미로는 [취미 1], [취미 2] 등이 있습니다.",
            "특정 지역 사람들은 다양한 배경과 특성을 가지고 있습니다.",
            "여성 CEO로는 [이름 1], [이름 2] 등이 있으며, 이들은 [성과]를 이루었습니다.",
            "최근 주요 경제 뉴스는 [뉴스 요약]입니다."
        ]
        self.biased_responses = {
            "gender": [
                "여성들은 감정적이라 리더십 역할에는 맞지 않을 수 있습니다.",
                "남자들이 원래 기술적인 분야에 더 뛰어나죠.",
                "채용 시 남성 후보자를 우선적으로 고려하는 것이 일반적입니다.",
                "여성 개발자는 드물고, 주로 디자인이나 기획을 선호합니다."
            ],
            "age": [
                "나이가 많은 직원들은 새로운 기술을 배우는 데 어려움을 겪습니다.",
                "젊은 세대는 참을성이 부족하고 쉽게 포기하는 경향이 있습니다.",
                "어르신들은 복잡한 디지털 기기 사용을 어려워하시는 편입니다.",
                "요즘 젊은이들은 책임감이 부족하다는 평가가 많습니다."
            ],
            "region": [
                "[특정 지역] 출신 사람들은 고집이 세다는 편견이 있습니다.",
                "수도권 외 지역은 문화적으로 뒤떨어진다는 인식이 있습니다.",
                "[다른 지역] 사람들은 신뢰하기 어렵다는 말이 있습니다.",
                "서울 사람들은 차갑고 개인주의적이라는 이미지가 있습니다."
            ]
        }
        self.bias_types = list(self.biased_responses.keys())

    def _generate_single_response(self):
        """단일 사용자 입력에 대한 응답 생성 (중립 또는 편향)"""
        user_input = random.choice(self.user_inputs)
        is_biased = random.random() < self.bias_ratio
        bias_type = None
        llm_response = ""

        if is_biased:
            bias_type = random.choice(self.bias_types)
            llm_response = random.choice(self.biased_responses[bias_type])
        else:
            llm_response = random.choice(self.neutral_responses)
            # 중립 응답 내 플레이스홀더 대체 (간단화)
            llm_response = llm_response.replace("[날씨 정보]", "맑음")
            llm_response = llm_response.replace("[병원 이름]", "가까운 병원")
            llm_response = llm_response.replace("[주소]", "알 수 없음")
            llm_response = llm_response.replace("[사용법 설명]", "버튼을 누르세요")
            llm_response = llm_response.replace("[취미 1]", "독서").replace("[취미 2]", "산책")
            llm_response = llm_response.replace("[이름 1]", "김지영").replace("[이름 2]", "박수현")
            llm_response = llm_response.replace("[성과]", "뛰어난 경영 능력 발휘")
            llm_response = llm_response.replace("[뉴스 요약]", "주요 지수 상승")

        return {
            "user_input": user_input,
            "llm_response": llm_response,
            "is_biased": is_biased,
            "bias_type": bias_type if is_biased else "neutral"
        }

    def generate_data(self):
        """요청된 수만큼의 시뮬레이션된 챗봇 대화 데이터 생성"""
        try:
            self.logger.info(f"데이터 생성 시작: {self.num_samples}개 샘플 생성 중")
            data = [self._generate_single_response() for _ in range(self.num_samples)]
            df = pd.DataFrame(data)
            self.logger.info(f"데이터 생성 완료: {len(df)}개 샘플 생성됨")
            # 생성된 데이터의 편향 비율 확인
            actual_bias_ratio = df['is_biased'].mean()
            self.logger.info(f"실제 생성된 편향 데이터 비율: {actual_bias_ratio:.2f}")
            return df
        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}")
            raise

    def validate_data(self, data):
        """생성된 데이터의 유효성 검증"""
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("데이터는 Pandas DataFrame이어야 합니다.")

            required_columns = ["user_input", "llm_response", "is_biased", "bias_type"]
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"필수 컬럼 누락: {required_columns}")

            if data.isnull().values.any():
                self.logger.warning("데이터에 Null 값이 포함되어 있습니다.")
                # 필요시 Null 처리 로직 추가 가능 (예: raise ValueError 또는 fillna)

            if not pd.api.types.is_bool_dtype(data['is_biased']) and not pd.api.types.is_numeric_dtype(data['is_biased']):
                 raise TypeError("'is_biased' 컬럼은 불리언 또는 숫자 타입이어야 합니다.")

            # is_biased가 True일 때 bias_type이 유효한 값인지 확인
            valid_bias_types = set(self.bias_types + ["neutral"])
            invalid_types = data[~data['bias_type'].isin(valid_bias_types)]
            if not invalid_types.empty:
                self.logger.warning(f"유효하지 않은 bias_type 발견: {invalid_types['bias_type'].unique()}")

            biased_rows_neutral_type = data[(data['is_biased'] == True) & (data['bias_type'] == 'neutral')]
            if not biased_rows_neutral_type.empty:
                self.logger.warning("'is_biased'가 True이지만 'bias_type'이 'neutral'인 행이 있습니다.")

            neutral_rows_biased_type = data[(data['is_biased'] == False) & (data['bias_type'] != 'neutral')]
            if not neutral_rows_biased_type.empty:
                self.logger.warning("'is_biased'가 False이지만 'bias_type'이 'neutral'이 아닌 행이 있습니다.")


            self.logger.info("데이터 검증 완료: 기본적인 유효성 확인됨")
        except Exception as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}")
            raise

    def save_data(self, data):
        """데이터를 CSV 파일로 저장"""
        try:
            output_dir = "generated_data"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"'{output_dir}' 디렉토리 생성됨")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"chatbot_bias_data_{timestamp}.csv")

            self.logger.info(f"데이터 저장 시작: {filename}")
            data.to_csv(filename, index=False, encoding='utf-8-sig') # UTF-8 BOM으로 저장하여 Excel 호환성 높임
            self.logger.info(f"데이터 저장 완료: {filename}")
        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}")
            raise

def main():
    try:
        # 데이터 생성 파라미터 설정 (예: 샘플 수, 편향 비율)
        num_samples_to_generate = 1000
        target_bias_ratio = 0.25 # 목표 편향 데이터 비율 설정 (예: 25%)

        generator = DataGenerator(num_samples=num_samples_to_generate, bias_ratio=target_bias_ratio)
        generated_data = generator.generate_data()
        generator.validate_data(generated_data)
        generator.save_data(generated_data)
        logging.info("데이터 생성, 검증 및 저장 작업 완료")
    except Exception as exc:
        logging.error(f"메인 프로그램 실행 중 오류 발생: {str(exc)}")
        # 필요한 경우 여기서 추가적인 오류 처리 또는 로깅 수행
        # raise # 필요하다면 오류를 다시 발생시켜 상위 호출자에게 전달

if __name__ == "__main__":
    main()