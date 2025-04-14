# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime

# 로깅 설정
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = os.path.join(log_dir, f"data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class DataGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = "generated_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")

    def _generate_spam_samples(self):
        """스팸 오탐지 분석용 데이터 생성"""
        self.logger.info("Generating spam detection samples...")
        spam_samples = [
            # 오탐 (False Positive) - 실제: 정상 (뉴스레터), 예측: 스팸
            {"email_id": "fp_001", "content": "Subject: Limited Time Offer! Huge discount on summer collection. Click here for exclusive deals!", "true_label": "ham", "predicted_label": "spam"},
            {"email_id": "fp_002", "content": "Subject: Don't Miss Out! Special weekend sale - up to 70% off. Free shipping on orders over $50!", "true_label": "ham", "predicted_label": "spam"},
            {"email_id": "fp_003", "content": "Subject: Your exclusive invite: Early access to our new product launch event. Limited spots!", "true_label": "ham", "predicted_label": "spam"},
            {"email_id": "fp_004", "content": "Subject: Urgent: Claim your reward points now! Bonus offer expires midnight. Winner announced soon!", "true_label": "ham", "predicted_label": "spam"},
            # 정상 (True Negative) - 실제: 정상, 예측: 정상 (참고용)
            {"email_id": "tn_001", "content": "Subject: Meeting Reminder: Project Alpha sync tomorrow at 10 AM.", "true_label": "ham", "predicted_label": "ham"}
        ]
        df = pd.DataFrame(spam_samples)
        self.logger.info(f"Generated {len(df)} spam detection samples.")
        return df

    def _generate_caption_samples(self):
        """캡셔닝 API 개선용 데이터 생성"""
        self.logger.info("Generating image captioning samples...")
        caption_samples = [
            {"image_id": "img_complex_01", "image_description": "A busy street market in Southeast Asia with various food stalls, crowds of people, and motorbikes.", "generated_caption": "A group of people outdoors."},
            {"image_id": "img_abstract_01", "image_description": "An abstract sculpture made of twisted metal pipes painted in bright red and yellow.", "generated_caption": "A close up of an object."},
            {"image_id": "img_cultural_01", "image_description": "People wearing traditional Hanbok celebrating the Chuseok festival in South Korea.", "generated_caption": "People in costumes."},
            {"image_id": "img_specific_01", "image_description": "A close-up photo of a rare blue morpho butterfly resting on a green leaf.", "generated_caption": "A blue butterfly."}
        ]
        df = pd.DataFrame(caption_samples)
        self.logger.info(f"Generated {len(df)} image captioning samples.")
        return df

    def generate_data(self):
        """두 종류의 면접 문제에 대한 데이터를 생성"""
        try:
            self.logger.info("데이터 생성 시작")
            spam_data = self._generate_spam_samples()
            caption_data = self._generate_caption_samples()
            data = {
                "spam_detection_samples": spam_data,
                "image_captioning_samples": caption_data
            }
            self.logger.info("데이터 생성 완료")
            return data
        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}")
            raise

    def validate_data(self, data):
        """생성된 데이터의 유효성 검증"""
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(data, dict):
                raise TypeError("Generated data should be a dictionary.")
            if "spam_detection_samples" not in data or "image_captioning_samples" not in data:
                raise ValueError("Dictionary must contain 'spam_detection_samples' and 'image_captioning_samples' keys.")

            # 스팸 데이터 검증
            spam_df = data["spam_detection_samples"]
            if not isinstance(spam_df, pd.DataFrame):
                raise TypeError("Spam data should be a Pandas DataFrame.")
            expected_spam_cols = ["email_id", "content", "true_label", "predicted_label"]
            if not all(col in spam_df.columns for col in expected_spam_cols):
                raise ValueError(f"Spam DataFrame missing one or more columns: {expected_spam_cols}")
            if len(spam_df) != 5:
                self.logger.warning(f"Expected 5 spam samples, but found {len(spam_df)}.") # 요구사항은 5개지만 경고만 표시
            # 오탐/정상 레이블 확인
            fp_count = len(spam_df[(spam_df['true_label'] == 'ham') & (spam_df['predicted_label'] == 'spam')])
            tn_count = len(spam_df[(spam_df['true_label'] == 'ham') & (spam_df['predicted_label'] == 'ham')])
            if fp_count < 4 : # 최소 4개의 오탐 샘플 확인
                 self.logger.warning(f"Expected at least 4 False Positive samples, found {fp_count}")
            if tn_count < 1 : # 최소 1개의 정상 샘플 확인
                 self.logger.warning(f"Expected at least 1 True Negative sample, found {tn_count}")
            self.logger.info("Spam data validation passed.")

            # 캡셔닝 데이터 검증
            caption_df = data["image_captioning_samples"]
            if not isinstance(caption_df, pd.DataFrame):
                raise TypeError("Captioning data should be a Pandas DataFrame.")
            expected_caption_cols = ["image_id", "image_description", "generated_caption"]
            if not all(col in caption_df.columns for col in expected_caption_cols):
                raise ValueError(f"Captioning DataFrame missing one or more columns: {expected_caption_cols}")
            if len(caption_df) == 0:
                raise ValueError("Captioning DataFrame is empty.")
            self.logger.info("Captioning data validation passed.")

            self.logger.info("데이터 검증 완료")
            return True # 검증 성공 시 True 반환

        except (TypeError, ValueError) as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}")
            raise # 오류 발생 시 예외 다시 발생
        except Exception as exc:
             self.logger.error(f"예상치 못한 데이터 검증 오류 발생: {str(exc)}")
             raise

    def save_data(self, data):
        """생성된 데이터를 CSV 파일로 저장"""
        try:
            self.logger.info("데이터 저장 시작")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            spam_filename = os.path.join(self.output_dir, f"spam_samples_{timestamp}.csv")
            spam_df = data["spam_detection_samples"]
            spam_df.to_csv(spam_filename, index=False, encoding='utf-8-sig')
            self.logger.info(f"Spam data saved to {spam_filename}")

            caption_filename = os.path.join(self.output_dir, f"caption_samples_{timestamp}.csv")
            caption_df = data["image_captioning_samples"]
            caption_df.to_csv(caption_filename, index=False, encoding='utf-8-sig')
            self.logger.info(f"Captioning data saved to {caption_filename}")

            self.logger.info("데이터 저장 완료")

        except KeyError as exc:
             self.logger.error(f"데이터 저장 중 오류 발생: Missing key {str(exc)} in data dictionary.")
             raise
        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}")
            raise

def main():
    try:
        generator = DataGenerator()
        generated_data = generator.generate_data()
        if generator.validate_data(generated_data): # validate_data 가 True 를 반환할 때만 저장
            generator.save_data(generated_data)
            logging.info("데이터 생성, 검증 및 저장 완료")
        else:
             logging.error("데이터 검증 실패. 데이터 저장 안 함.") # 검증 실패 시 로그 남기기
    except Exception as exc:
        logging.error(f"프로그램 실행 중 오류 발생: {str(exc)}")
        # raise # 필요에 따라 주석 해제하여 프로그램 중단

if __name__ == "__main__":
    main()