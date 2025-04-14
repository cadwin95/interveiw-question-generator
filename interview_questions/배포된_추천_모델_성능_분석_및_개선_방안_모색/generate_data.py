# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime, timedelta

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
    def __init__(self, num_users=1000, num_items=5000, start_date_str='2023-10-01', days_old_model=7, days_new_model=7):
        """
        데이터 생성기 초기화

        Args:
            num_users (int): 생성할 사용자 수
            num_items (int): 생성할 상품 수
            start_date_str (str): 데이터 생성 시작 날짜 (YYYY-MM-DD)
            days_old_model (int): 이전 모델 운영 기간 (일)
            days_new_model (int): 새 모델 운영 기간 (일)
        """
        self.logger = logging.getLogger(__name__)
        self.num_users = num_users
        self.num_items = num_items
        self.user_ids = list(range(1, num_users + 1))
        self.item_ids = list(range(1, num_items + 1))

        try:
            self.start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError as e:
            self.logger.error(f"잘못된 날짜 형식입니다. 'YYYY-MM-DD' 형식을 사용해주세요: {e}")
            raise

        self.old_model_end_date = self.start_date + timedelta(days=days_old_model)
        self.new_model_end_date = self.old_model_end_date + timedelta(days=days_new_model)
        self.total_days = days_old_model + days_new_model

        # 모델별 특성 파라미터 (시뮬레이션용)
        self.params = {
            'UserCF': {
                'complaint_rate': 0.01, # 기본 불만족 비율
                'relevance_complaint_rate': 0.005, # 관련성 부족 불만족 비율
                'already_purchased_rate': 0.002, # 이미 구매한 상품 추천 비율 (불만족 원인 중)
                'ctr_base': 0.05 # 기본 CTR
            },
            'MF': {
                'complaint_rate': 0.012, # 기본 불만족 비율 (약간 증가)
                'relevance_complaint_rate': 0.005 * 1.2, # 관련성 부족 불만족 비율 (20% 증가)
                'already_purchased_rate': 0.004, # 이미 구매한 상품 추천 비율 증가 (불만족 원인 중)
                'too_different_rate': 0.003, # 너무 다른 상품 추천 비율 (불만족 원인 중)
                'ctr_base': 0.051 # 기본 CTR (유사하거나 약간 높게 설정)
            }
        }
        self.logger.info(f"데이터 생성기 초기화 완료: {num_users} users, {num_items} items, Period: {start_date_str} to {self.new_model_end_date.strftime('%Y-%m-%d')}")

    def _generate_timestamps(self, start_date, end_date, count):
        """지정된 기간 내 랜덤 타임스탬프 생성"""
        delta = end_date - start_date
        random_seconds = [random.random() * delta.total_seconds() for _ in range(count)]
        return [start_date + timedelta(seconds=s) for s in random_seconds]

    def _generate_recommendations(self, user_id, timestamp, model_version, purchase_history_user):
        """특정 사용자에 대한 추천 목록 생성 (시뮬레이션)"""
        num_recommendations = 10 # NDCG@10 이므로 10개 추천 가정
        recommendations = []

        # 시뮬레이션: MF 모델이 '이미 구매한 상품' 또는 '너무 다른 상품'을 추천할 가능성 추가
        if model_version == 'MF':
            # 이미 구매한 상품 추천 시뮬레이션
            if random.random() < self.params['MF']['already_purchased_rate'] * 10 and not purchase_history_user.empty: # 확률 증폭
                # 최근 구매 상품 중 하나 선택 (시뮬레이션 단순화)
                recent_purchase = purchase_history_user.sort_values('timestamp', ascending=False).iloc[0]['item_id']
                if recent_purchase not in recommendations:
                    recommendations.append(recent_purchase)

            # 너무 다른 상품 추천 시뮬레이션
            while len(recommendations) < num_recommendations and random.random() < self.params['MF']['too_different_rate'] * 5: # 확률 증폭
                random_item = random.choice(self.item_ids)
                if random_item not in recommendations and random_item not in purchase_history_user['item_id'].tolist():
                     recommendations.append(random_item)

        # 나머지 추천 목록 채우기 (랜덤 아이템으로 단순화)
        while len(recommendations) < num_recommendations:
            item = random.choice(self.item_ids)
            if item not in recommendations and item not in purchase_history_user['item_id'].tolist(): # 구매 안 한 상품 중 추천
                recommendations.append(item)

        # 클릭 시뮬레이션 (CTR 기반)
        clicked = random.random() < self.params[model_version]['ctr_base']
        clicked_item_id = random.choice(recommendations) if clicked else None

        return recommendations, clicked_item_id

    def _generate_feedback(self, user_id, timestamp, model_version, recommendations, purchase_history_user):
        """사용자 피드백 생성 (시뮬레이션)"""
        feedback = None
        if random.random() < self.params[model_version]['complaint_rate']:
            feedback_type = 'Complaint'
            reason = 'Other' # 기본 이유

            # 관련성 부족 불만 사유 시뮬레이션
            if random.random() < (self.params[model_version]['relevance_complaint_rate'] / self.params[model_version]['complaint_rate']):
                # MF 모델의 특정 불만 사유 증가 반영
                possible_reasons = ['Irrelevant']
                reason_weights = [0.6]
                if model_version == 'MF':
                    possible_reasons.extend(['Already Purchased', 'Too Different'])
                    # 가중치 부여 (Already Purchased와 Too Different 확률 높임)
                    already_purchased_prob = self.params['MF']['already_purchased_rate'] * 30 # 확률 증폭
                    too_different_prob = self.params['MF']['too_different_rate'] * 20 # 확률 증폭
                    irrelevant_prob = 1.0 - already_purchased_prob - too_different_prob
                    if irrelevant_prob < 0: # 확률 합이 1을 넘지 않도록 조정
                       scale = 1.0 / (already_purchased_prob + too_different_prob)
                       already_purchased_prob *= scale
                       too_different_prob *= scale
                       irrelevant_prob = 0
                    reason_weights = [irrelevant_prob, already_purchased_prob, too_different_prob]

                elif model_version == 'UserCF':
                     possible_reasons.append('Already Purchased')
                     already_purchased_prob_ucf = self.params['UserCF']['already_purchased_rate'] * 30 # 확률 증폭
                     irrelevant_prob_ucf = 1.0 - already_purchased_prob_ucf
                     if irrelevant_prob_ucf < 0:
                         already_purchased_prob_ucf = 1.0
                         irrelevant_prob_ucf = 0
                     reason_weights = [irrelevant_prob_ucf, already_purchased_prob_ucf]


                # 가중치에 따라 이유 선택
                # numpy를 사용할 수 없으므로 random.choices 사용 (Python 3.6+) 또는 직접 구현
                # 여기서는 random.choices 를 사용한다고 가정 (만약 3.6 미만이면 다른 방식 필요)
                try:
                    reason = random.choices(possible_reasons, weights=reason_weights, k=1)[0]
                except NameError: # random.choices 가 없는 경우 (Python < 3.6)
                     # 간단한 가중 랜덤 선택 구현
                     rand_val = random.random()
                     cumulative_weight = 0.0
                     selected_reason = possible_reasons[-1] # 기본값
                     for r, w in zip(possible_reasons, reason_weights):
                         cumulative_weight += w
                         if rand_val < cumulative_weight:
                             selected_reason = r
                             break
                     reason = selected_reason


                # 'Already Purchased' 이유가 선택되었는지 확인 (실제 구매 내역과 비교)
                if reason == 'Already Purchased':
                    purchased_in_recs = any(item in purchase_history_user['item_id'].tolist() for item in recommendations)
                    if not purchased_in_recs: # 만약 추천 목록에 실제 구매한게 없다면 이유를 'Irrelevant'로 변경
                        reason = 'Irrelevant'

            feedback = {
                'user_id': user_id,
                'timestamp': timestamp,
                'model_version_active': model_version,
                'feedback_type': feedback_type,
                'reason': reason
            }
        return feedback


    def generate_data(self):
        """
        문제 시나리오에 맞는 데이터 생성: 구매 내역, 추천 로그, 사용자 피드백

        Returns:
            dict: 생성된 데이터프레임들을 담은 딕셔너리
                  {'purchase_history': pd.DataFrame, 'recommendation_log': pd.DataFrame, 'user_feedback': pd.DataFrame}
        """
        try:
            self.logger.info("데이터 생성 시작")

            # 1. 구매 내역 생성 (전체 기간)
            self.logger.info("구매 내역 데이터 생성 중...")
            purchase_records = []
            num_purchases = self.num_users * 15 # 사용자당 평균 15회 구매 가정
            purchase_timestamps = self._generate_timestamps(self.start_date, self.new_model_end_date, num_purchases)
            for i in range(num_purchases):
                user_id = random.choice(self.user_ids)
                item_id = random.choice(self.item_ids)
                purchase_records.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'timestamp': purchase_timestamps[i]
                })
            purchase_history_df = pd.DataFrame(purchase_records)
            purchase_history_df = purchase_history_df.sort_values(by='timestamp').reset_index(drop=True)
            self.logger.info(f"구매 내역 데이터 생성 완료: {len(purchase_history_df)} 건")


            # 2. 추천 로그 및 사용자 피드백 생성 (모델별 기간 구분)
            self.logger.info("추천 로그 및 사용자 피드백 데이터 생성 중...")
            recommendation_logs = []
            feedback_logs = []
            recommendation_events_per_day = self.num_users * 2 # 사용자당 하루 평균 2번 추천 슬롯 조회 가정

            for day in range(self.total_days):
                current_date = self.start_date + timedelta(days=day)
                day_start_time = current_date
                day_end_time = current_date + timedelta(days=1)

                model_version = 'UserCF' if current_date < self.old_model_end_date else 'MF'
                self.logger.info(f"{current_date.strftime('%Y-%m-%d')} 데이터 생성 중 (Model: {model_version})")

                # 해당 날짜에 활동할 사용자 샘플링 (모든 사용자가 매일 활동하지 않음)
                active_users_today = random.sample(self.user_ids, k=int(self.num_users * 0.8)) # 80% 사용자 활동 가정
                num_events_today = len(active_users_today) * 2 # 활동 사용자당 2회 추천 이벤트

                event_timestamps = self._generate_timestamps(day_start_time, day_end_time, num_events_today)

                event_idx = 0
                for user_id in active_users_today:
                     # 해당 사용자의 현재까지 구매 내역 필터링
                    user_purchases = purchase_history_df[
                        (purchase_history_df['user_id'] == user_id) &
                        (purchase_history_df['timestamp'] < event_timestamps[event_idx]) # 이벤트 시점 이전 구매내역
                    ]

                    # 2번의 추천 이벤트 생성 (예시)
                    for _ in range(2):
                        if event_idx >= len(event_timestamps): break # 생성된 타임스탬프 소진 시 중단

                        timestamp = event_timestamps[event_idx]

                        # 추천 생성
                        recommendations, clicked_item_id = self._generate_recommendations(user_id, timestamp, model_version, user_purchases)

                        recommendation_logs.append({
                            'log_id': len(recommendation_logs) + 1,
                            'user_id': user_id,
                            'timestamp': timestamp,
                            'model_version': model_version,
                            'recommended_items': recommendations, # 리스트 형태로 저장
                            'clicked': clicked_item_id is not None,
                            'clicked_item_id': clicked_item_id
                        })

                        # 피드백 생성
                        feedback = self._generate_feedback(user_id, timestamp, model_version, recommendations, user_purchases)
                        if feedback:
                            feedback['feedback_id'] = len(feedback_logs) + 1
                            feedback_logs.append(feedback)

                        event_idx += 1


            recommendation_log_df = pd.DataFrame(recommendation_logs)
            user_feedback_df = pd.DataFrame(feedback_logs)

            self.logger.info(f"추천 로그 데이터 생성 완료: {len(recommendation_log_df)} 건")
            self.logger.info(f"사용자 피드백 데이터 생성 완료: {len(user_feedback_df)} 건")

            self.data = {
                'purchase_history': purchase_history_df,
                'recommendation_log': recommendation_log_df,
                'user_feedback': user_feedback_df
            }
            self.logger.info("데이터 생성 완료")
            return self.data

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}", exc_info=True)
            raise

    def validate_data(self, data):
        """
        생성된 데이터의 유효성 검증

        Args:
            data (dict): 검증할 데이터프레임 딕셔너리
        """
        try:
            self.logger.info("데이터 검증 시작")
            required_keys = ['purchase_history', 'recommendation_log', 'user_feedback']
            if not all(key in data for key in required_keys):
                raise ValueError(f"필수 데이터 키가 누락되었습니다: {required_keys}")

            # 1. 데이터프레임 존재 및 비어있는지 확인
            for name, df in data.items():
                if not isinstance(df, pd.DataFrame):
                     raise TypeError(f"'{name}'은(는) pandas DataFrame이 아닙니다.")
                if df.empty:
                    self.logger.warning(f"'{name}' 데이터프레임이 비어있습니다.")
                else:
                    self.logger.info(f"'{name}' 데이터프레임 검증 통과 (Rows: {len(df)})")

            # 2. 필수 컬럼 확인
            required_columns = {
                'purchase_history': ['user_id', 'item_id', 'timestamp'],
                'recommendation_log': ['log_id', 'user_id', 'timestamp', 'model_version', 'recommended_items', 'clicked', 'clicked_item_id'],
                'user_feedback': ['feedback_id', 'user_id', 'timestamp', 'model_version_active', 'feedback_type', 'reason']
            }
            for name, cols in required_columns.items():
                if not data[name].empty:
                    if not all(col in data[name].columns for col in cols):
                        raise ValueError(f"'{name}' 데이터프레임에 필수 컬럼이 누락되었습니다: {cols}")
                    # 타임스탬프 타입 확인
                    if 'timestamp' in data[name].columns:
                         if not pd.api.types.is_datetime64_any_dtype(data[name]['timestamp']):
                              raise TypeError(f"'{name}' 데이터프레임의 'timestamp' 컬럼이 datetime 타입이 아닙니다.")
            self.logger.info("필수 컬럼 및 타입 검증 완료.")

            # 3. 시나리오 관련 지표 검증 (시뮬레이션 결과 확인)
            if not data['recommendation_log'].empty and not data['user_feedback'].empty:
                rec_log = data['recommendation_log']
                feedback = data['user_feedback']

                # 모델별 CTR 계산
                ctr_usercf = rec_log[rec_log['model_version'] == 'UserCF']['clicked'].mean()
                ctr_mf = rec_log[rec_log['model_version'] == 'MF']['clicked'].mean()
                self.logger.info(f"CTR (UserCF): {ctr_usercf:.4f}")
                self.logger.info(f"CTR (MF): {ctr_mf:.4f}")
                # CTR 비교 (유의미한 차이 없는지 확인 - 여기서는 대략적 비교)
                if abs(ctr_usercf - ctr_mf) > 0.01: # 예시: 1%p 이상 차이나면 경고
                    self.logger.warning(f"두 모델 간 CTR 차이가 예상보다 큽니다 (UserCF: {ctr_usercf:.4f}, MF: {ctr_mf:.4f})")
                else:
                    self.logger.info("모델 간 CTR 유사성 확인 (시뮬레이션 기준)")

                # 모델별 불만 피드백 비율 계산
                total_feedback_usercf = len(feedback[feedback['model_version_active'] == 'UserCF'])
                complaints_usercf = len(feedback[(feedback['model_version_active'] == 'UserCF') & (feedback['feedback_type'] == 'Complaint')])
                complaint_rate_usercf = complaints_usercf / total_feedback_usercf if total_feedback_usercf > 0 else 0

                total_feedback_mf = len(feedback[feedback['model_version_active'] == 'MF'])
                complaints_mf = len(feedback[(feedback['model_version_active'] == 'MF') & (feedback['feedback_type'] == 'Complaint')])
                complaint_rate_mf = complaints_mf / total_feedback_mf if total_feedback_mf > 0 else 0

                self.logger.info(f"불만 피드백 비율 (UserCF): {complaint_rate_usercf:.4f} ({complaints_usercf}/{total_feedback_usercf})")
                self.logger.info(f"불만 피드백 비율 (MF): {complaint_rate_mf:.4f} ({complaints_mf}/{total_feedback_mf})")

                # 모델별 '관련성 부족' 사유 비율 계산 (불만 피드백 중)
                relevance_complaints_usercf = len(feedback[
                    (feedback['model_version_active'] == 'UserCF') &
                    (feedback['feedback_type'] == 'Complaint') &
                    (feedback['reason'].isin(['Irrelevant', 'Already Purchased', 'Too Different'])) # 'Too Different'는 UserCF에선 거의 없어야 함
                ])
                relevance_complaint_ratio_usercf = relevance_complaints_usercf / complaints_usercf if complaints_usercf > 0 else 0

                relevance_complaints_mf = len(feedback[
                    (feedback['model_version_active'] == 'MF') &
                    (feedback['feedback_type'] == 'Complaint') &
                    (feedback['reason'].isin(['Irrelevant', 'Already Purchased', 'Too Different']))
                ])
                relevance_complaint_ratio_mf = relevance_complaints_mf / complaints_mf if complaints_mf > 0 else 0

                self.logger.info(f"'관련성 부족' 불만 비율 (UserCF): {relevance_complaint_ratio_usercf:.4f}")
                self.logger.info(f"'관련성 부족' 불만 비율 (MF): {relevance_complaint_ratio_mf:.4f}")

                # MF 모델에서 관련성 부족 불만이 UserCF 대비 증가했는지 확인 (문제 시나리오: 20% 증가)
                if relevance_complaint_ratio_mf < relevance_complaint_ratio_usercf * 1.1: # 10% 증가 미만이면 경고 (20% 목표)
                     self.logger.warning(f"MF 모델의 '관련성 부족' 불만 비율 증가가 예상보다 낮습니다 (UserCF: {relevance_complaint_ratio_usercf:.4f}, MF: {relevance_complaint_ratio_mf:.4f})")
                else:
                     increase_perc = ((relevance_complaint_ratio_mf / relevance_complaint_ratio_usercf) - 1) * 100 if relevance_complaint_ratio_usercf > 0 else float('inf')
                     self.logger.info(f"MF 모델 '관련성 부족' 불만 비율 증가 확인 (UserCF 대비 약 {increase_perc:.1f}% 증가)")

                # MF 모델에서 'Already Purchased', 'Too Different' 사유가 실제로 증가했는지 확인
                already_purchased_mf = len(feedback[(feedback['model_version_active'] == 'MF') & (feedback['reason'] == 'Already Purchased')])
                too_different_mf = len(feedback[(feedback['model_version_active'] == 'MF') & (feedback['reason'] == 'Too Different')])
                already_purchased_usercf = len(feedback[(feedback['model_version_active'] == 'UserCF') & (feedback['reason'] == 'Already Purchased')])
                too_different_usercf = len(feedback[(feedback['model_version_active'] == 'UserCF') & (feedback['reason'] == 'Too Different')]) # 거의 0이어야 함

                self.logger.info(f"'Already Purchased' 불만 건수: UserCF={already_purchased_usercf}, MF={already_purchased_mf}")
                self.logger.info(f"'Too Different' 불만 건수: UserCF={too_different_usercf}, MF={too_different_mf}")
                if not (already_purchased_mf > already_purchased_usercf and too_different_mf > too_different_usercf):
                    self.logger.warning("MF 모델의 'Already Purchased' 또는 'Too Different' 불만 사유 증가가 시뮬레이션되지 않았을 수 있습니다.")
                else:
                    self.logger.info("MF 모델 특정 불만 사유 증가 확인")

            self.logger.info("데이터 검증 완료")

        except Exception as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}", exc_info=True)
            raise

    def save_data(self, data, output_dir='generated_data'):
        """
        생성된 데이터를 CSV 파일로 저장

        Args:
            data (dict): 저장할 데이터프레임 딕셔너리
            output_dir (str): 데이터를 저장할 디렉토리 경로
        """
        try:
            self.logger.info(f"데이터 저장 시작 (경로: {output_dir})")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"'{output_dir}' 디렉토리 생성 완료")

            for name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    filepath = os.path.join(output_dir, f"{name}.csv")
                    # 리스트 형태의 컬럼은 문자열로 변환하여 저장 (CSV 호환성)
                    df_copy = df.copy()
                    if name == 'recommendation_log' and 'recommended_items' in df_copy.columns:
                         # 리스트를 콤마로 구분된 문자열로 변환
                         df_copy['recommended_items'] = df_copy['recommended_items'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)

                    df_copy.to_csv(filepath, index=False, encoding='utf-8')
                    self.logger.info(f"'{filepath}' 저장 완료")
                else:
                    self.logger.warning(f"'{name}'은(는) DataFrame이 아니므로 저장할 수 없습니다.")

            self.logger.info("데이터 저장 완료")

        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}", exc_info=True)
            raise

def main():
    """메인 실행 함수"""
    try:
        # 데이터 생성 파라미터 설정 (필요에 따라 조절)
        generator = DataGenerator(
            num_users=500,        # 사용자 수 줄여서 테스트 시간 단축
            num_items=1000,       # 상품 수 줄여서 테스트 시간 단축
            start_date_str='2023-11-01',
            days_old_model=7,     # 이전 모델 기간
            days_new_model=7      # 새 모델 기간
        )

        # 데이터 생성
        generated_data = generator.generate_data()

        # 데이터 검증
        generator.validate_data(generated_data)

        # 데이터 저장
        generator.save_data(generated_data, output_dir='simulated_recommendation_data')

        logging.info("데이터 생성, 검증 및 저장 완료")

    except ValueError as ve:
        logging.error(f"입력 값 오류: {str(ve)}")
    except Exception as exc:
        logging.error(f"프로그램 실행 중 예기치 않은 오류 발생: {str(exc)}", exc_info=True)
        # raise # 필요한 경우 에러를 다시 발생시켜 상위에서 처리하도록 함

if __name__ == "__main__":
    main()