# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime, timedelta

# 로깅 설정
log_file = 'data_generation.log'
# Ensure the log file exists or can be created
try:
    # Try to open the file in append mode, which creates it if it doesn't exist
    with open(log_file, 'a') as f:
        pass
except IOError as e:
    # Handle potential errors like permission issues
    print(f"Error setting up log file {log_file}: {e}")
    # Fallback or exit if logging is critical
    # For this example, we'll proceed without file logging if it fails
    log_handlers = [logging.StreamHandler()]
else:
    log_handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)

class DataGenerator:
    def __init__(self, num_days=30):
        self.logger = logging.getLogger(__name__)
        self.num_days = num_days
        self.start_date = datetime.now().date() - timedelta(days=num_days)

    def generate_data(self):
        try:
            self.logger.info("데이터 생성 시작")

            # 날짜 생성
            dates = [self.start_date + timedelta(days=i) for i in range(self.num_days)]
            date_strings = [d.strftime('%Y-%m-%d') for d in dates] # 문자열 형식 날짜

            # 주가 데이터 생성 (종가)
            stock_prices = []
            current_price = 100.0
            for _ in range(self.num_days):
                change = random.uniform(-2.5, 2.5) # 일별 가격 변동
                current_price += change
                stock_prices.append(max(0.1, round(current_price, 2))) # 최소 가격 0.1

            stock_df = pd.DataFrame({
                'date': date_strings,
                'close': stock_prices
            })

            # 뉴스 감성 점수 데이터 생성
            sentiment_scores = [round(random.uniform(-1.0, 1.0), 4) for _ in range(self.num_days)]
            sentiment_df = pd.DataFrame({
                'date': date_strings,
                'sentiment_score': sentiment_scores
            })

            self.logger.info(f"{self.num_days}일 분량의 주가 및 감성 점수 데이터 생성 완료")
            return {'stock': stock_df, 'sentiment': sentiment_df}

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}")
            raise

    def validate_data(self, data):
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(data, dict):
                raise ValueError("입력 데이터는 딕셔너리 형태여야 합니다.")
            if 'stock' not in data or 'sentiment' not in data:
                raise ValueError("데이터 딕셔너리에 'stock' 또는 'sentiment' 키가 없습니다.")

            stock_df = data['stock']
            sentiment_df = data['sentiment']

            if not isinstance(stock_df, pd.DataFrame) or not isinstance(sentiment_df, pd.DataFrame):
                raise ValueError("데이터는 Pandas DataFrame 형태여야 합니다.")

            # stock_df 검증
            if not {'date', 'close'}.issubset(stock_df.columns):
                raise ValueError("주가 데이터에 'date' 또는 'close' 컬럼이 없습니다.")
            if stock_df['date'].isnull().any() or stock_df['close'].isnull().any():
                raise ValueError("주가 데이터에 누락된 값이 있습니다.")
            if not pd.api.types.is_string_dtype(stock_df['date']): # 문자열 날짜 확인
                 try:
                     pd.to_datetime(stock_df['date'], errors='raise')
                 except Exception as e:
                     raise ValueError(f"주가 데이터의 'date' 컬럼 형식이 올바르지 않습니다: {e}")
            if not pd.api.types.is_numeric_dtype(stock_df['close']):
                raise ValueError("주가 데이터의 'close' 컬럼은 숫자형이어야 합니다.")
            if not stock_df['date'].is_unique:
                 raise ValueError("주가 데이터의 'date' 값이 고유하지 않습니다.")


            # sentiment_df 검증
            if not {'date', 'sentiment_score'}.issubset(sentiment_df.columns):
                raise ValueError("감성 점수 데이터에 'date' 또는 'sentiment_score' 컬럼이 없습니다.")
            if sentiment_df['date'].isnull().any() or sentiment_df['sentiment_score'].isnull().any():
                raise ValueError("감성 점수 데이터에 누락된 값이 있습니다.")
            if not pd.api.types.is_string_dtype(sentiment_df['date']): # 문자열 날짜 확인
                 try:
                     pd.to_datetime(sentiment_df['date'], errors='raise')
                 except Exception as e:
                     raise ValueError(f"감성 점수 데이터의 'date' 컬럼 형식이 올바르지 않습니다: {e}")
            if not pd.api.types.is_numeric_dtype(sentiment_df['sentiment_score']):
                raise ValueError("감성 점수 데이터의 'sentiment_score' 컬럼은 숫자형이어야 합니다.")
            if not sentiment_df['date'].is_unique:
                 raise ValueError("감성 점수 데이터의 'date' 값이 고유하지 않습니다.")

            # 두 데이터프레임 간 날짜 일치 확인 (개수만 확인)
            if len(stock_df['date'].unique()) != len(sentiment_df['date'].unique()):
                 self.logger.warning("주가 데이터와 감성 점수 데이터의 날짜 수가 다릅니다.")

            self.logger.info("데이터 검증 완료")
            return True

        except Exception as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}")
            raise

    def save_data(self, data):
        try:
            self.logger.info("데이터 저장 시작")
            if not isinstance(data, dict) or 'stock' not in data or 'sentiment' not in data:
                raise ValueError("저장할 데이터 형식이 올바르지 않습니다.")

            stock_df = data['stock']
            sentiment_df = data['sentiment']
            stock_filename = 'stock_price_simple.csv'
            sentiment_filename = 'news_sentiment_simple.csv'

            stock_df.to_csv(stock_filename, index=False, encoding='utf-8')
            self.logger.info(f"주가 데이터를 '{stock_filename}'으로 저장했습니다.")

            sentiment_df.to_csv(sentiment_filename, index=False, encoding='utf-8')
            self.logger.info(f"감성 점수 데이터를 '{sentiment_filename}'으로 저장했습니다.")

        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}")
            raise

def main():
    try:
        # 생성할 데이터 일 수 설정 (예: 50일)
        generator = DataGenerator(num_days=50)
        generated_data = generator.generate_data()
        if generator.validate_data(generated_data):
            generator.save_data(generated_data)
            logging.info("데이터 생성, 검증 및 저장 완료")
        else:
             logging.warning("데이터 검증 실패로 저장을 진행하지 않습니다.")

    except Exception as exc:
        logging.error(f"프로그램 실행 중 오류 발생: {str(exc)}")
        # Re-raise the exception if you want the program to terminate on error
        # raise

if __name__ == "__main__":
    main()