# 필요한 패키지
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime
import uuid # 고유 ID 생성을 위해 추가

# 로깅 설정
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True) # 로그 디렉토리 생성
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
    def __init__(self, num_samples=100, output_dir="data"):
        self.logger = logging.getLogger(__name__)
        self.num_samples = num_samples
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) # 데이터 저장 디렉토리 생성

        # 샘플 데이터 생성을 위한 기본 목록 정의
        self.products = [
            "고성능 게이밍 노트북", "친환경 주방 세제", "AI 기반 영어 학습 앱",
            "프리미엄 스킨케어 세트", "스마트 홈 IoT 기기", "유기농 아기 간식",
            "전문가용 드론", "맞춤형 비타민 구독 서비스", "휴대용 미니 빔프로젝터",
            "노이즈캔슬링 블루투스 이어폰"
        ]
        self.target_audiences = [
            "20대 대학생", "30대 직장인 여성", "IT 업계 종사자", "육아 중인 부모",
            "여행을 즐기는 사람", "건강에 관심 많은 50대", "1인 가구", "MZ세대",
            "새로운 기술 얼리어답터", "환경 의식 있는 소비자"
        ]
        self.tones = ["전문적인", "친근한", "유머러스한", "감성적인", "활기찬", "진지한", "고급스러운"]
        self.forbidden_words = ["최악", "절대", "싸구려", "바보", "망함", "후회"]
        self.positive_keywords = ["혁신적인", "놀라운", "편리한", "효과적인", "만족스러운", "최고의", "추천"]
        self.negative_keywords = ["불편한", "실망스러운", "어려운", "비싼", "복잡한", "부족한"]
        self.max_length_threshold = 80 # 최대 글자 수 예시

    def _generate_single_copy(self, product, audience, tone):
        """AI가 생성했을 법한 문구를 시뮬레이션 (의도적으로 품질 이슈 포함 가능성 추가)"""
        base_copy = f"{audience}을(를) 위한 {product}! "

        # 톤앤매너 반영 시뮬레이션 (간단하게 키워드 추가)
        if tone == "전문적인":
            base_copy += "최신 기술로 구현된 성능을 경험하세요. "
        elif tone == "친근한":
            base_copy += "당신의 일상을 더 특별하게 만들어 줄 거예요. "
        elif tone == "유머러스한":
            base_copy += "이거 없으면... 글쎄요? 😉 "
        elif tone == "감성적인":
            base_copy += "소중한 당신에게 선물하세요. ✨ "
        elif tone == "활기찬":
            base_copy += "지금 바로 시작하세요! 에너지가 넘칠 거예요! "
        elif tone == "진지한":
            base_copy += "신중한 선택, 후회하지 않으실 겁니다. "
        elif tone == "고급스러운":
             base_copy += "차원이 다른 경험을 선사합니다. "
        else:
            base_copy += "특별한 혜택을 놓치지 마세요. "

        # 품질 이슈 시뮬레이션
        final_copy = base_copy
        issues = []

        # 1. 글자 수 초과 가능성 (10% 확률)
        if random.random() < 0.1:
            final_copy += "정말 길고 긴 설명입니다. 이 문구는 최대 글자 수를 초과할 가능성이 매우 높습니다. 왜냐하면 이렇게 길게 써야 초과하니까요!" * 2
            issues.append("length_exceeded")

        # 2. 금지어 포함 가능성 (10% 확률)
        if random.random() < 0.1 and len(self.forbidden_words) > 0:
            forbidden = random.choice(self.forbidden_words)
            final_copy += f" 하지만 {forbidden} 단점도 있을 수 있죠." # 금지어 삽입
            issues.append("forbidden_word")

        # 3. 긍정적 어조 유지 실패 가능성 (15% 확률, 부정 키워드 추가)
        if random.random() < 0.15 and len(self.negative_keywords) > 0:
             negative = random.choice(self.negative_keywords)
             final_copy += f" 가끔 {negative} 점이 발견될 수도 있습니다."
             issues.append("negative_tone")
        # 3-1. (보완) 긍정 키워드 미포함 가능성도 있음 (기본 문구에 긍정 키워드 부족 시)
        elif not any(p_keyword in final_copy for p_keyword in self.positive_keywords):
             if random.random() < 0.2: # 20% 확률로 긍정 부족 이슈로 판단
                 issues.append("lack_of_positive_tone")


        # 4. 입력 문맥과의 불일치 가능성 (15% 확률, 관련 없는 제품/타겟 언급)
        if random.random() < 0.15:
            wrong_product = random.choice([p for p in self.products if p != product])
            wrong_audience = random.choice([a for a in self.target_audiences if a != audience])
            context_choice = random.choice(['product', 'audience'])
            if context_choice == 'product':
                final_copy += f" 마치 {wrong_product}처럼 느껴질 수도 있어요."
                issues.append("context_mismatch_product")
            else:
                 final_copy += f" 특히 {wrong_audience}에게는 맞지 않을 수 있습니다."
                 issues.append("context_mismatch_audience")


        # 실제 생성된 문구의 (시뮬레이션된) 이슈 태그 추가
        # 실제 검증 시스템은 이 태그 없이 문구 자체만 보고 이슈를 판별해야 함
        simulated_issues = ",".join(issues) if issues else "None"

        return final_copy, simulated_issues


    def generate_data(self):
        """면접 문제에 필요한 시뮬레이션 데이터를 생성합니다."""
        try:
            self.logger.info(f"데이터 생성 시작: {self.num_samples}개 샘플")
            data = []
            for i in range(self.num_samples):
                product_info = random.choice(self.products)
                target_audience = random.choice(self.target_audiences)
                tone_and_manner = random.choice(self.tones)

                generated_copy, simulated_issues = self._generate_single_copy(product_info, target_audience, tone_and_manner)

                record = {
                    'request_id': str(uuid.uuid4()), # 고유 요청 ID
                    'generation_timestamp': datetime.now(),
                    'product_info': product_info,
                    'target_audience': target_audience,
                    'tone_and_manner': tone_and_manner,
                    'generated_copy': generated_copy,
                    'generated_copy_length': len(generated_copy), # 검증 편의를 위해 길이 추가
                    # 'simulated_issues': simulated_issues # 시뮬레이션 정보는 실제 검증 시스템 입력에는 없음 (디버깅/참고용)
                }
                data.append(record)

            df = pd.DataFrame(data)
            self.logger.info(f"데이터 생성 완료: {len(df)}개 샘플 생성됨")
            return df

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}")
            raise # 오류를 다시 발생시켜 main에서 처리하도록 함

    def validate_data(self, data):
        """생성된 데이터의 기본적인 유효성을 검증합니다."""
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("생성된 데이터가 Pandas DataFrame 형식이 아닙니다.")

            required_columns = ['request_id', 'generation_timestamp', 'product_info',
                                'target_audience', 'tone_and_manner', 'generated_copy',
                                'generated_copy_length']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼 누락: {', '.join(missing_columns)}")

            # Null 값 확인 (generated_copy는 비어있으면 안 됨)
            if data['generated_copy'].isnull().any():
                null_count = data['generated_copy'].isnull().sum()
                self.logger.warning(f"'generated_copy' 컬럼에 {null_count}개의 Null 값이 있습니다.")
                # 필요시 raise ValueError("generated_copy 컬럼에 Null 값이 포함될 수 없습니다.")

            # 데이터 타입 확인 (예시: timestamp)
            if not pd.api.types.is_datetime64_any_dtype(data['generation_timestamp']):
                 self.logger.warning("'generation_timestamp' 컬럼 타입이 datetime 형식이 아닙니다.")
                 # data['generation_timestamp'] = pd.to_datetime(data['generation_timestamp'], errors='coerce') # 변환 시도

            self.logger.info("데이터 검증 완료: 기본적인 유효성 확인됨")
            return True

        except Exception as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}")
            raise

    def save_data(self, data):
        """생성된 데이터를 CSV 파일로 저장합니다."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"marketing_copy_data_{timestamp}.csv")
            self.logger.info(f"데이터 저장 시작: {filename}")

            if not isinstance(data, pd.DataFrame):
                 self.logger.error("저장할 데이터가 DataFrame 형식이 아닙니다.")
                 raise TypeError("저장할 데이터는 Pandas DataFrame이어야 합니다.")

            data.to_csv(filename, index=False, encoding='utf-8-sig') # UTF-8 BOM으로 저장 (Excel 호환성)
            self.logger.info(f"데이터 저장 완료: {filename}")

        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}")
            raise

def main():
    """데이터 생성, 검증, 저장 프로세스를 실행합니다."""
    logging.info("="*30)
    logging.info("AI 문구 품질 검증용 데이터 생성 스크립트 시작")
    logging.info("="*30)
    try:
        # num_samples: 생성할 샘플 데이터 수
        # output_dir: 데이터 파일 저장 경로
        generator = DataGenerator(num_samples=200, output_dir="generated_data")

        # 1. 데이터 생성
        generated_data = generator.generate_data()

        # 2. 데이터 검증
        generator.validate_data(generated_data)

        # 3. 데이터 저장
        generator.save_data(generated_data)

        logging.info("모든 프로세스 완료: 데이터 생성, 검증 및 저장 성공")

    except Exception as exc:
        # main 함수 레벨에서 최종 에러 로깅
        logging.error(f"프로그램 실행 중 심각한 오류 발생: {str(exc)}")
        # 필요하다면 여기서 프로그램 종료 또는 추가적인 에러 처리 수행
        # raise # 주석 처리하면 오류 발생 시에도 프로그램이 비정상 종료되지 않음

    finally:
        logging.info("="*30)
        logging.info("데이터 생성 스크립트 종료")
        logging.info("="*30)


if __name__ == "__main__":
    main()