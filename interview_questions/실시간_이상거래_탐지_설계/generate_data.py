# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime, timedelta

# 로깅 설정
log_file = 'data_generation.log'
# Check if log file exists and remove it to start fresh
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
    def __init__(self, num_users=100, num_transactions=1000, fraud_ratio=0.05, missing_ratio=0.02):
        self.logger = logging.getLogger(__name__)
        self.num_users = num_users
        self.num_transactions = num_transactions
        self.fraud_ratio = fraud_ratio
        self.missing_ratio = missing_ratio
        self.user_ids = [f'user_{i:03}' for i in range(num_users)]
        self.merchant_categories = ['online_retail', 'travel', 'electronics', 'groceries', 'restaurant', 'gas_station', 'entertainment']
        self.device_types = ['mobile_app', 'web_browser', 'pos_terminal', 'unknown']
        self.normal_ip_ranges = [f'192.168.1.{random.randint(1, 254)}' for _ in range(num_users // 2)] # Simulate common IPs
        self.foreign_ip_ranges = [f'10.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}' for _ in range(num_users // 10)] # Simulate foreign IPs

    def _generate_ip(self, user_id, is_fraud_location=False):
        if is_fraud_location and self.foreign_ip_ranges:
            return random.choice(self.foreign_ip_ranges)
        # Simple mapping for consistency (can be improved)
        user_index = int(user_id.split('_')[1])
        if user_index < len(self.normal_ip_ranges):
            return self.normal_ip_ranges[user_index]
        else:
            return f'172.16.{random.randint(1, 254)}.{random.randint(1, 254)}' # Another private range for variety

    def _introduce_missing_values(self, data):
        self.logger.info(f"데이터 품질 문제 주입 시작 (결측치 비율: {self.missing_ratio * 100}%)")
        df = pd.DataFrame(data)
        cols_to_inject = ['ip_address', 'device_info'] # Columns where missing data might occur
        for col in cols_to_inject:
            mask = np.random.choice([True, False], size=len(df), p=[self.missing_ratio, 1 - self.missing_ratio])
            df.loc[mask, col] = np.nan
        self.logger.info("결측치 주입 완료")
        return df

    def generate_data(self):
        try:
            self.logger.info("데이터 생성 시작")
            data = []
            current_time = datetime.now()
            num_fraud = int(self.num_transactions * self.fraud_ratio)
            fraud_indices = random.sample(range(self.num_transactions), num_fraud)
            fraud_type_flags = random.choices(['location', 'velocity'], k=num_fraud) # Assign types of fraud

            fraud_counter = 0
            last_fraud_user = None
            velocity_anomaly_count = 0
            velocity_anomaly_target = 0

            for i in range(self.num_transactions):
                user_id = random.choice(self.user_ids)
                timestamp = current_time + timedelta(milliseconds=random.randint(50, 500)) # Simulate real-time stream
                current_time = timestamp
                amount = round(random.uniform(5.0, 500.0), 2) # Normal transaction amounts
                merchant_category = random.choice(self.merchant_categories)
                ip_address = self._generate_ip(user_id)
                device_info = random.choice(self.device_types)
                is_fraud = 0

                if i in fraud_indices:
                    is_fraud = 1
                    fraud_type = fraud_type_flags[fraud_counter]
                    fraud_user_id = random.choice(self.user_ids) # Pick a user for fraud

                    if fraud_type == 'location' and self.foreign_ip_ranges:
                        # Simulate location anomaly: Foreign IP, potentially higher amount, relevant category
                        user_id = fraud_user_id
                        ip_address = self._generate_ip(user_id, is_fraud_location=True)
                        amount = round(random.uniform(100.0, 1500.0), 2)
                        merchant_category = random.choice(['travel', 'electronics', 'online_retail'])
                        self.logger.debug(f"위치 기반 이상 거래 생성: User {user_id}, IP {ip_address}")
                        last_fraud_user = None # Reset velocity tracking

                    elif fraud_type == 'velocity':
                         # Start or continue a velocity anomaly sequence
                        if last_fraud_user != fraud_user_id or velocity_anomaly_count == 0:
                            last_fraud_user = fraud_user_id
                            velocity_anomaly_target = random.randint(3, 7) # Number of rapid transactions
                            velocity_anomaly_count = velocity_anomaly_target

                        user_id = last_fraud_user
                        timestamp = current_time + timedelta(milliseconds=random.randint(10, 100)) # Very short time difference
                        current_time = timestamp
                        amount = round(random.uniform(1.0, 20.0), 2) # Small amounts
                        merchant_category = 'online_retail' # Often online
                        ip_address = self._generate_ip(user_id) # Could be same or slightly different IP
                        device_info = 'web_browser' # Often web
                        velocity_anomaly_count -= 1
                        self.logger.debug(f"속도 기반 이상 거래 생성 ({velocity_anomaly_target - velocity_anomaly_count}/{velocity_anomaly_target}): User {user_id}, Amount {amount}")
                        if velocity_anomaly_count == 0:
                            last_fraud_user = None # End of sequence

                    else: # Fallback if no foreign IPs defined or other case
                        is_fraud = 0 # Treat as normal if fraud condition cannot be met
                        last_fraud_user = None

                    fraud_counter += 1

                else:
                    # Reset velocity tracking if a normal transaction occurs for the last fraud user
                    if user_id == last_fraud_user:
                        last_fraud_user = None
                        velocity_anomaly_count = 0

                data.append({
                    'user_id': user_id,
                    'timestamp': timestamp,
                    'amount': amount,
                    'merchant_category': merchant_category,
                    'ip_address': ip_address,
                    'device_info': device_info,
                    'is_fraud': is_fraud # Label for evaluation
                })

            df = self._introduce_missing_values(data)
            df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure correct dtype
            self.logger.info(f"데이터 생성 완료: 총 {len(df)} 건, 이상 거래 {df['is_fraud'].sum()} 건")
            return df

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}")
            raise

    def validate_data(self, data):
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("입력 데이터가 Pandas DataFrame이 아닙니다.")

            required_columns = ['user_id', 'timestamp', 'amount', 'merchant_category', 'ip_address', 'device_info', 'is_fraud']
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"필수 컬럼 누락: {', '.join(missing_cols)}")

            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                 raise TypeError("timestamp 컬럼이 datetime 형식이 아닙니다.")
            if not pd.api.types.is_numeric_dtype(data['amount']):
                 raise TypeError("amount 컬럼이 숫자 형식이 아닙니다.")
            if not pd.api.types.is_integer_dtype(data['is_fraud']):
                 raise TypeError("is_fraud 컬럼이 정수 형식이 아닙니다.")

            # Check for excessive missing values (example threshold: 50%)
            missing_summary = data.isnull().mean()
            high_missing_cols = missing_summary[missing_summary > 0.5].index.tolist()
            if high_missing_cols:
                 self.logger.warning(f"결측치 비율이 50% 이상인 컬럼: {', '.join(high_missing_cols)}")
            else:
                 self.logger.info(f"결측치 비율 확인 완료: \n{missing_summary[missing_summary > 0]}")


            # Check for negative amounts
            if (data['amount'] < 0).any():
                self.logger.warning("금액(amount) 컬럼에 음수 값이 포함되어 있습니다.")

            self.logger.info("데이터 검증 완료")
            return True

        except Exception as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}")
            raise

    def save_data(self, data, filename="transactions.csv"):
        try:
            self.logger.info(f"데이터 저장 시작 -> {filename}")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("저장할 데이터가 Pandas DataFrame이 아닙니다.")

            # Create directory if it doesn't exist
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"디렉토리 생성: {output_dir}")

            data.to_csv(filename, index=False, encoding='utf-8')
            self.logger.info(f"데이터 저장 완료: {filename}")

        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}")
            raise

def main():
    try:
        # Configuration
        NUM_TRANSACTIONS = 5000 # Increase for more realistic stream simulation
        FRAUD_RATIO = 0.03      # Percentage of fraudulent transactions
        MISSING_RATIO = 0.01    # Percentage of missing values to inject

        generator = DataGenerator(
            num_users=200,
            num_transactions=NUM_TRANSACTIONS,
            fraud_ratio=FRAUD_RATIO,
            missing_ratio=MISSING_RATIO
        )
        data = generator.generate_data()
        generator.validate_data(data)
        generator.save_data(data, filename="simulated_transactions.csv")
        logging.info("데이터 생성 및 저장 완료")

    except Exception as exc:
        logging.error(f"프로그램 실행 중 오류 발생: {str(exc)}")
        # In a real application, you might want to exit with a non-zero code
        # raise exc

if __name__ == "__main__":
    main()