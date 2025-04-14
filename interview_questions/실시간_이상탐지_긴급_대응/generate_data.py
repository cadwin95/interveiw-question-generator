# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime, timedelta

# 로깅 설정
# 로그 파일 이름에 현재 시간을 포함하여 중복 방지
log_filename = f'data_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class DataGenerator:
    def __init__(self, num_records_past=10000, num_records_recent=5000, fraud_ratio_past=0.01, fraud_ratio_recent=0.02, new_fraud_ratio_recent=0.015):
        """
        데이터 생성기 초기화

        Args:
            num_records_past (int): 과거 데이터 레코드 수
            num_records_recent (int): 최근 데이터 레코드 수
            fraud_ratio_past (float): 과거 데이터 내 기존 사기 비율
            fraud_ratio_recent (float): 최근 데이터 내 전체 사기 비율 (기존+신규)
            new_fraud_ratio_recent (float): 최근 데이터 내 신규 사기 비율
        """
        self.logger = logging.getLogger(__name__)
        self.num_records_past = num_records_past
        self.num_records_recent = num_records_recent
        self.fraud_ratio_past = fraud_ratio_past
        self.fraud_ratio_recent = fraud_ratio_recent
        self.new_fraud_ratio_recent = new_fraud_ratio_recent
        if new_fraud_ratio_recent > fraud_ratio_recent:
            raise ValueError("신규 사기 비율은 전체 사기 비율보다 클 수 없습니다.")
        self.old_fraud_ratio_recent = fraud_ratio_recent - new_fraud_ratio_recent

        self.logger.info(f"데이터 생성기 초기화 완료: 과거 {num_records_past}건, 최근 {num_records_recent}건")

    def _generate_base_data(self, num_records, start_date, end_date):
        """기본 거래 데이터 생성 (정상 거래 위주)"""
        data = {
            'timestamp': [start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(num_records)],
            'transaction_id': [f'T{i+random.randint(100000, 999999)}' for i in range(num_records)],
            'user_id': [f'U{random.randint(1000, 9999)}' for _ in range(num_records)],
            'amount': np.random.lognormal(mean=3.0, sigma=1.0, size=num_records).round(2), # 로그 정규 분포 거래 금액
            'location_risk_score': np.random.uniform(0.0, 0.5, size=num_records).round(4), # 지역 위험 점수 (낮음)
            'activity_freq_score': np.random.uniform(0.2, 0.8, size=num_records).round(4), # 활동 빈도 점수 (중간)
            'label': [0] * num_records # 기본값은 정상(0)
        }
        df = pd.DataFrame(data)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        self.logger.debug(f"{num_records} 건의 기본 데이터 생성 완료")
        return df

    def _introduce_old_fraud(self, df, fraud_ratio):
        """기존 사기 패턴 데이터 삽입"""
        num_fraud = int(len(df) * fraud_ratio)
        fraud_indices = random.sample(range(len(df)), num_fraud)

        for idx in fraud_indices:
            # 기존 사기 패턴 특징: 높은 금액, 높은 지역 위험도
            df.loc[idx, 'amount'] = max(df.loc[idx, 'amount'], np.random.lognormal(mean=6.0, sigma=1.0)) # 더 높은 금액
            df.loc[idx, 'location_risk_score'] = np.random.uniform(0.8, 1.0) # 높은 지역 위험
            df.loc[idx, 'label'] = 1 # 기존 사기(1) 레이블
        self.logger.debug(f"{num_fraud} 건의 기존 사기 패턴 데이터 삽입 완료")
        return df

    def _introduce_new_fraud(self, df, new_fraud_ratio):
        """신규 사기 패턴 데이터 삽입 (최근 데이터에만 적용)"""
        # 이미 사기로 레이블링된 인덱스 제외
        available_indices = df[df['label'] == 0].index.tolist()
        if not available_indices:
             self.logger.warning("신규 사기를 삽입할 정상 거래가 없습니다.")
             return df

        num_new_fraud = int(len(df) * new_fraud_ratio)
        if num_new_fraud > len(available_indices):
            num_new_fraud = len(available_indices)
            self.logger.warning(f"요청된 신규 사기 건수({int(len(df) * new_fraud_ratio)})가 가용한 정상 거래 수({len(available_indices)})보다 많아 조정합니다.")

        fraud_indices = random.sample(available_indices, num_new_fraud)

        for idx in fraud_indices:
            # 신규 사기 패턴 특징: 낮은 금액(소액 다수), 낮은 지역 위험도, 낮은 활동 빈도 (휴면 계정 등)
            df.loc[idx, 'amount'] = np.random.lognormal(mean=1.5, sigma=0.5) # 낮은 금액
            df.loc[idx, 'location_risk_score'] = np.random.uniform(0.1, 0.4) # 지역 위험도는 낮을 수 있음
            df.loc[idx, 'activity_freq_score'] = np.random.uniform(0.0, 0.2) # 낮은 활동 빈도
            # 중요: 면접 시나리오에 따라 신규 패턴 레이블은 '모름' 또는 '정상으로 오인' 상태를 반영
            # 여기서는 레이블을 2로 설정하되, 모델 예측 시 미탐되도록 처리
            df.loc[idx, 'label'] = 2 # 신규 사기(2) 레이블 (실제로는 없을 수 있음)
        self.logger.debug(f"{num_new_fraud} 건의 신규 사기 패턴 데이터 삽입 완료")
        return df

    def _add_model_predictions(self, df):
        """기존 모델의 예측 결과 시뮬레이션"""
        predictions = []
        true_positives = 0
        false_negatives_old = 0
        false_negatives_new = 0
        total_old_fraud = 0
        total_new_fraud = 0

        # 간단한 규칙 기반 또는 확률적 예측 시뮬레이션
        for _, row in df.iterrows():
            # 기존 모델은 기존 사기 패턴(1)은 잘 탐지하지만, 신규 패턴(2)은 탐지 못함
            if row['label'] == 1: # 기존 사기
                total_old_fraud += 1
                # 높은 확률로 탐지 (예: 95% Recall)
                if random.random() < 0.95:
                    predictions.append(1)
                    true_positives += 1
                else:
                    predictions.append(0) # 미탐 (False Negative)
                    false_negatives_old += 1
            elif row['label'] == 2: # 신규 사기
                total_new_fraud += 1
                # 매우 낮은 확률로 탐지 (예: 5% Recall) - 대부분 미탐
                if random.random() < 0.05:
                     predictions.append(1) # 우연히 탐지 (True Positive - New)
                     # true_positives += 1 # TP 계산시 포함 여부는 논의 필요
                else:
                     predictions.append(0) # 미탐 (False Negative - New)
                     false_negatives_new += 1
            else: # 정상(0)
                # 낮은 확률로 오탐 (예: 1% False Positive Rate)
                if random.random() < 0.01:
                    predictions.append(1) # 오탐 (False Positive)
                else:
                    predictions.append(0) # 정상 예측 (True Negative)

        df['model_prediction'] = predictions
        self.logger.info("모델 예측 결과 추가 완료")

        # Recall 계산 및 로깅 (디버깅/확인용)
        if total_old_fraud > 0:
             recall_old = true_positives / total_old_fraud
             self.logger.debug(f"시뮬레이션된 기존 사기 Recall: {recall_old:.4f} ({true_positives}/{total_old_fraud})")
        if total_new_fraud > 0:
             # 신규 사기에 대한 Recall은 별도로 보거나, 전체 Recall 계산시 포함 여부 결정 필요
             recall_new = (df[(df['label'] == 2) & (df['model_prediction'] == 1)].shape[0]) / total_new_fraud
             self.logger.debug(f"시뮬레이션된 신규 사기 Recall: {recall_new:.4f} ({df[(df['label'] == 2) & (df['model_prediction'] == 1)].shape[0]}/{total_new_fraud})")
             # 전체 사기 Recall (기존 모델 기준)
             total_fraud = total_old_fraud + total_new_fraud
             total_detected = df[(df['label'].isin([1, 2])) & (df['model_prediction'] == 1)].shape[0]
             overall_recall = total_detected / total_fraud if total_fraud > 0 else 0
             self.logger.debug(f"시뮬레이션된 전체 사기 Recall: {overall_recall:.4f} ({total_detected}/{total_fraud})")

        return df

    def generate_data(self):
        """과거 및 최근 데이터를 생성하고 병합"""
        try:
            self.logger.info("데이터 생성 시작")
            # 시간 설정
            now = datetime.now()
            recent_start_date = now - timedelta(days=7) # 최근 1주일
            past_end_date = recent_start_date - timedelta(seconds=1)
            past_start_date = past_end_date - timedelta(days=30) # 그 이전 1달

            # 과거 데이터 생성
            self.logger.info("과거 데이터 생성 중...")
            df_past = self._generate_base_data(self.num_records_past, past_start_date, past_end_date)
            df_past = self._introduce_old_fraud(df_past, self.fraud_ratio_past)
            df_past = self._add_model_predictions(df_past)
            df_past['period'] = 'past'
            self.logger.info(f"과거 데이터 생성 완료 ({len(df_past)} 건)")

            # 최근 데이터 생성
            self.logger.info("최근 데이터 생성 중...")
            df_recent = self._generate_base_data(self.num_records_recent, recent_start_date, now)
            # 최근 데이터에는 기존 사기와 신규 사기 모두 포함
            df_recent = self._introduce_old_fraud(df_recent, self.old_fraud_ratio_recent)
            df_recent = self._introduce_new_fraud(df_recent, self.new_fraud_ratio_recent)
            df_recent = self._add_model_predictions(df_recent) # 신규 사기는 대부분 미탐될 것
            df_recent['period'] = 'recent'

            # 면접 시나리오: 신규 패턴은 레이블링되지 않았을 가능성 높음
            # 실제 생성된 데이터에서는 레이블(2)을 유지하되, 분석 시 이 레이블이 없다고 가정하고 접근해야 함
            # 필요하다면 아래 코드로 레이블을 제거할 수 있음
            # df_recent.loc[df_recent['label'] == 2, 'label'] = np.nan # 또는 0으로 설정
            self.logger.info(f"최근 데이터 생성 완료 ({len(df_recent)} 건)")

            # 데이터 병합
            final_df = pd.concat([df_past, df_recent], ignore_index=True)
            final_df = final_df.sort_values(by='timestamp').reset_index(drop=True)
            self.logger.info(f"전체 데이터 생성 및 병합 완료 (총 {len(final_df)} 건)")

            return final_df

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}")
            raise

    def validate_data(self, data):
        """생성된 데이터 검증"""
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("입력 데이터는 Pandas DataFrame이어야 합니다.")
            if data.empty:
                raise ValueError("생성된 데이터가 비어 있습니다.")

            # 기본 정보 확인
            self.logger.info(f"데이터 형태(Shape): {data.shape}")
            self.logger.info(f"데이터 타입(dtypes):\n{data.dtypes}")

            # 결측치 확인
            null_counts = data.isnull().sum()
            self.logger.info(f"결측치 수:\n{null_counts[null_counts > 0]}")
            # assert not data.isnull().any().any(), "데이터에 결측치가 포함되어 있습니다." # 필요시 활성화

            # 기간별 데이터 수 확인
            period_counts = data['period'].value_counts()
            self.logger.info(f"기간별 데이터 수:\n{period_counts}")
            assert 'past' in period_counts and 'recent' in period_counts, "과거 또는 최근 데이터가 생성되지 않았습니다."

            # 레이블 분포 확인
            label_dist = data['label'].value_counts(normalize=True).sort_index()
            self.logger.info(f"전체 레이블 분포:\n{label_dist}")
            label_dist_recent = data[data['period'] == 'recent']['label'].value_counts(normalize=True).sort_index()
            self.logger.info(f"최근 레이블 분포:\n{label_dist_recent}")
            # assert 2 in label_dist_recent.index, "최근 데이터에 신규 사기 레이블(2)이 없습니다." # 레이블을 유지했을 경우 검증

            # 모델 예측 결과 확인 (Recall 저하 시뮬레이션 검증)
            recall_past = data[(data['period'] == 'past') & (data['label'] == 1) & (data['model_prediction'] == 1)].shape[0] / data[(data['period'] == 'past') & (data['label'] == 1)].shape[0] if data[(data['period'] == 'past') & (data['label'] == 1)].shape[0] > 0 else 0
            # 최근 Recall: 신규 패턴(2)을 포함한 전체 사기(1, 2)에 대한 Recall
            recent_fraud = data[(data['period'] == 'recent') & (data['label'].isin([1, 2]))]
            recent_detected_fraud = recent_fraud[recent_fraud['model_prediction'] == 1]
            recall_recent = len(recent_detected_fraud) / len(recent_fraud) if len(recent_fraud) > 0 else 0

            self.logger.info(f"과거 데이터 사기 탐지율 (Recall, Label=1): {recall_past:.4f}")
            self.logger.info(f"최근 데이터 전체 사기 탐지율 (Recall, Label=1 or 2): {recall_recent:.4f}")

            if recall_recent >= recall_past and len(recent_fraud) > 0 and data[(data['period'] == 'past') & (data['label'] == 1)].shape[0] > 0:
                 self.logger.warning("경고: 최근 데이터의 Recall이 과거 데이터보다 높거나 같습니다. 신규 패턴 미탐 시뮬레이션이 의도대로 작동하지 않았을 수 있습니다.")
            # assert recall_recent < recall_past, "최근 데이터의 Recall이 과거보다 낮아야 합니다 (시나리오 검증 실패)." # 엄격한 검증 필요시 활성화

            self.logger.info("데이터 검증 완료")
            return True

        except Exception as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}")
            raise

    def save_data(self, data, filename="generated_transaction_data.csv"):
        """데이터를 CSV 파일로 저장"""
        try:
            self.logger.info(f"데이터 저장 시작: {filename}")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("저장할 데이터는 Pandas DataFrame이어야 합니다.")

            data.to_csv(filename, index=False, encoding='utf-8-sig')
            self.logger.info(f"데이터 저장 완료: {os.path.abspath(filename)}")

        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}")
            raise

def main():
    try:
        # 데이터 생성 파라미터 설정 (필요에 따라 조정)
        generator = DataGenerator(
            num_records_past=20000,    # 과거 데이터 수 증가
            num_records_recent=10000,   # 최근 데이터 수 증가
            fraud_ratio_past=0.01,      # 과거 사기 비율
            fraud_ratio_recent=0.025,    # 최근 전체 사기 비율 (기존+신규)
            new_fraud_ratio_recent=0.015 # 최근 신규 사기 비율 (전체 사기 비율보다 작아야 함)
        )

        # 데이터 생성
        generated_data = generator.generate_data()

        # 데이터 검증
        generator.validate_data(generated_data)

        # 데이터 저장
        generator.save_data(generated_data, filename=f'transaction_data_{datetime.now().strftime("%Y%m%d")}.csv')

        logging.info("데이터 생성, 검증 및 저장 작업 완료")

    except ValueError as ve:
         logging.error(f"설정 오류: {str(ve)}")
    except Exception as exc:
        logging.error(f"프로그램 실행 중 예상치 못한 오류 발생: {str(exc)}")
        # raise # 필요하다면 에러를 다시 발생시켜 프로그램 중단

if __name__ == "__main__":
    main()