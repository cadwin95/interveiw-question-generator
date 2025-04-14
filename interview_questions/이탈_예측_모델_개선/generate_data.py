# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime

# 로깅 설정
log_filename = f"data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class DataGenerator:
    def __init__(self, num_samples=10000, churn_rate=0.05, random_state=42):
        """
        데이터 생성기 초기화

        Args:
            num_samples (int): 생성할 총 고객 샘플 수
            churn_rate (float): 실제 이탈 고객 비율 (0 ~ 1)
            random_state (int): 재현성을 위한 랜덤 시드
        """
        self.logger = logging.getLogger(__name__)
        self.num_samples = num_samples
        self.churn_rate = churn_rate
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.logger.info(f"DataGenerator 초기화 완료: num_samples={num_samples}, churn_rate={churn_rate}, random_state={random_state}")

    def _generate_features(self):
        """고객 속성(Feature) 데이터 생성"""
        self.logger.info("고객 속성 데이터 생성 시작")
        # 예시 특성: 서비스 사용 기간 (월), 월 평균 사용량, 고객 지원 문의 횟수
        # 실제로는 더 다양하고 의미있는 특성이 필요함
        tenure = np.random.randint(1, 72, size=self.num_samples) # 1개월 ~ 6년
        monthly_usage = np.random.normal(100, 30, size=self.num_samples) # 평균 100, 표준편차 30
        support_calls = np.random.poisson(1.5, size=self.num_samples) # 평균 1.5회

        # 사용량이 음수가 되지 않도록 처리
        monthly_usage[monthly_usage < 0] = 0

        features = pd.DataFrame({
            'CustomerID': range(1, self.num_samples + 1),
            'TenureMonths': tenure,
            'MonthlyUsage': monthly_usage,
            'SupportCalls': support_calls
        })
        self.logger.info(f"{len(features.columns)-1}개 고객 속성 생성 완료")
        return features

    def _generate_target(self, features):
        """이탈 여부(Target Label) 데이터 생성 (불균형 반영)"""
        self.logger.info("이탈 여부 데이터 생성 시작")
        num_churners = int(self.num_samples * self.churn_rate)
        num_non_churners = self.num_samples - num_churners

        # 이탈 가능성 점수 생성 (간단한 규칙 기반)
        # 사용 기간 짧고, 사용량 적고, 문의 많을수록 이탈 가능성 높다고 가정
        churn_propensity = (
            -0.02 * features['TenureMonths']
            -0.005 * features['MonthlyUsage']
            + 0.3 * features['SupportCalls']
            + np.random.normal(0, 0.5, self.num_samples) # 노이즈 추가
        )

        # 이탈 가능성 점수가 높은 순서대로 정렬하여 상위 num_churners 만큼을 이탈(1)로 지정
        churn_threshold = np.percentile(churn_propensity, 100 * (1 - self.churn_rate))
        target = (churn_propensity >= churn_threshold).astype(int)

        # 실제 이탈 비율이 목표 비율과 유사하도록 조정 (미세 조정 로직 추가 가능)
        current_churn_count = target.sum()
        if current_churn_count != num_churners:
            self.logger.warning(f"목표 이탈자 수({num_churners})와 실제 생성된 이탈자 수({current_churn_count})가 다릅니다. 재조정 시도.")
            # 간단한 재조정: 차이만큼 무작위로 변경 (더 정교한 방법 필요할 수 있음)
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices)
            diff = num_churners - current_churn_count
            if diff > 0: # 이탈자가 부족한 경우
                non_churner_indices = indices[target[indices] == 0]
                change_indices = np.random.choice(non_churner_indices, size=diff, replace=False)
                target[change_indices] = 1
            elif diff < 0: # 이탈자가 많은 경우
                churner_indices = indices[target[indices] == 1]
                change_indices = np.random.choice(churner_indices, size=abs(diff), replace=False)
                target[change_indices] = 0
            
            final_churn_count = target.sum()
            if final_churn_count == num_churners:
                 self.logger.info(f"이탈자 수 재조정 완료: {final_churn_count}명")
            else:
                 self.logger.warning(f"이탈자 수 재조정 후에도 차이 발생: {final_churn_count}명 (목표: {num_churners}명)")


        self.logger.info(f"이탈 여부 데이터 생성 완료 (이탈자 수: {target.sum()}, 비율: {target.mean():.4f})")
        return pd.Series(target, name='Churn')

    def generate_data(self):
        """데이터 생성 프로세스 실행"""
        try:
            self.logger.info("데이터 생성 시작")
            features = self._generate_features()
            target = self._generate_target(features)
            data = pd.concat([features, target], axis=1)
            self.logger.info("데이터 생성 완료")
            return data
        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}", exc_info=True)
            raise

    def validate_data(self, data):
        """생성된 데이터 검증"""
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("생성된 데이터가 Pandas DataFrame이 아닙니다.")
            if data.empty:
                raise ValueError("생성된 데이터가 비어 있습니다.")

            expected_columns = ['CustomerID', 'TenureMonths', 'MonthlyUsage', 'SupportCalls', 'Churn']
            if not all(col in data.columns for col in expected_columns):
                raise ValueError(f"데이터에 필요한 컬럼이 없습니다. 필요 컬럼: {expected_columns}, 현재 컬럼: {list(data.columns)}")

            if data.isnull().values.any():
                self.logger.warning("데이터에 결측치가 포함되어 있습니다.")
                # raise ValueError("데이터에 결측치가 포함되어 있습니다.") # 필요시 활성화

            actual_churn_rate = data['Churn'].mean()
            if not np.isclose(actual_churn_rate, self.churn_rate, atol=0.01): # 허용 오차 1%
                 self.logger.warning(f"실제 이탈율({actual_churn_rate:.4f})이 목표 이탈율({self.churn_rate:.4f})과 차이가 있습니다.")
                # raise ValueError(f"실제 이탈율({actual_churn_rate:.4f})이 목표 이탈율({self.churn_rate:.4f})과 차이가 있습니다.") # 필요시 활성화

            if not pd.api.types.is_integer_dtype(data['Churn']):
                raise TypeError("Churn 컬럼의 데이터 타입이 정수형(int)이 아닙니다.")
            if not set(data['Churn'].unique()).issubset({0, 1}):
                raise ValueError(f"Churn 컬럼에 예상치 못한 값({data['Churn'].unique()})이 포함되어 있습니다. 0 또는 1이어야 합니다.")

            self.logger.info(f"데이터 검증 완료. 총 샘플 수: {len(data)}, 실제 이탈율: {actual_churn_rate:.4f}")
            return True
        except Exception as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}", exc_info=True)
            raise

    def save_data(self, data, filename_prefix="churn_data"):
        """데이터를 CSV 파일로 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.csv"
            data.to_csv(filename, index=False)
            self.logger.info(f"데이터를 '{filename}'으로 저장 완료")
        except IOError as exc:
            self.logger.error(f"파일 저장 중 IO 오류 발생: {str(exc)}", exc_info=True)
            raise
        except Exception as exc:
            self.logger.error(f"데이터 저장 중 예상치 못한 오류 발생: {str(exc)}", exc_info=True)
            raise

def main():
    """메인 실행 함수"""
    try:
        logging.info("데이터 생성 프로그램 시작")
        # 데이터 생성 파라미터 설정
        generator = DataGenerator(num_samples=20000, churn_rate=0.05, random_state=123)
        
        # 데이터 생성
        generated_data = generator.generate_data()
        
        # 데이터 검증
        generator.validate_data(generated_data)
        
        # 데이터 저장
        generator.save_data(generated_data, filename_prefix="customer_churn_data")
        
        logging.info("데이터 생성 및 저장 완료")
        
    except Exception as exc:
        logging.error(f"프로그램 실행 중 오류 발생: {str(exc)}", exc_info=True)
        # raise # 필요시 주석 해제하여 에러 전파

if __name__ == "__main__":
    main()