# 필요한 패키지
import pandas as pd
import random
import logging
import os
from datetime import datetime
import json # 데이터 검증 및 로깅을 위해 추가

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
    def __init__(self, num_articles=7, output_dir="data"):
        """
        DataGenerator 클래스 초기화

        Args:
            num_articles (int): 생성할 샘플 뉴스 기사 수 (기본값: 7)
            output_dir (str): 생성된 데이터를 저장할 디렉토리 (기본값: "data")
        """
        self.logger = logging.getLogger(__name__)
        self.num_articles = num_articles
        self.output_dir = output_dir
        self.categories = ["정치", "경제", "사회", "IT/과학", "생활/문화", "스포츠"]
        self.sentiments = [-1, 0, 1] # -1: 부정적, 0: 중립적, 1: 긍정적
        self.logger.info(f"DataGenerator 초기화 완료. 생성할 기사 수: {num_articles}, 출력 디렉토리: {output_dir}")

    def generate_data(self) -> list:
        """
        샘플 뉴스 기사 데이터를 생성합니다.

        Returns:
            list: 각 기사 정보를 담은 딕셔너리의 리스트
                예: [{'article_id': 'news001', 'title': '...', 'category': '...', 'predefined_sentiment': 0}, ...]
        """
        articles = []
        try:
            self.logger.info(f"{self.num_articles}개의 샘플 뉴스 기사 데이터 생성 시작")
            for i in range(1, self.num_articles + 1):
                article_id = f"news{i:03d}"
                category = random.choice(self.categories)
                title = f"{category} 관련 샘플 기사 제목 {i}"
                predefined_sentiment = random.choice(self.sentiments)

                article = {
                    "article_id": article_id,
                    "title": title,
                    "category": category,
                    "predefined_sentiment": predefined_sentiment
                }
                articles.append(article)
                self.logger.debug(f"생성된 기사: {json.dumps(article, ensure_ascii=False)}")

            self.logger.info("샘플 뉴스 기사 데이터 생성 완료")
            return articles
        except Exception as exc:
            self.logger.error(f"샘플 데이터 생성 중 오류 발생: {str(exc)}", exc_info=True)
            raise # 오류를 다시 발생시켜 main 함수에서 처리하도록 함

    def validate_data(self, data: list):
        """
        생성된 데이터의 유효성을 검증합니다.

        Args:
            data (list): 검증할 기사 데이터 리스트

        Raises:
            ValueError: 데이터가 유효하지 않을 경우 발생
        """
        try:
            self.logger.info("생성된 데이터 검증 시작")
            if not isinstance(data, list) or not data:
                raise ValueError("데이터가 비어있거나 리스트 형식이 아닙니다.")

            required_keys = {"article_id", "title", "category", "predefined_sentiment"}
            article_ids = set()

            for i, article in enumerate(data):
                if not isinstance(article, dict):
                    raise ValueError(f"데이터 항목 {i}가 딕셔너리 형식이 아닙니다: {article}")

                # 필수 키 존재 여부 확인
                if not required_keys.issubset(article.keys()):
                    missing_keys = required_keys - article.keys()
                    raise ValueError(f"데이터 항목 {i}에 필수 키가 누락되었습니다: {missing_keys}. 데이터: {article}")

                # 타입 검증
                if not isinstance(article["article_id"], str) or not article["article_id"]:
                    raise ValueError(f"데이터 항목 {i}의 'article_id'는 비어 있지 않은 문자열이어야 합니다: {article['article_id']}")
                if not isinstance(article["title"], str) or not article["title"]:
                     raise ValueError(f"데이터 항목 {i}의 'title'은 비어 있지 않은 문자열이어야 합니다: {article['title']}")
                if article["category"] not in self.categories:
                    raise ValueError(f"데이터 항목 {i}의 'category'가 유효한 카테고리 목록에 없습니다: {article['category']}")
                if article["predefined_sentiment"] not in self.sentiments:
                     raise ValueError(f"데이터 항목 {i}의 'predefined_sentiment'가 유효한 값(-1, 0, 1)이 아닙니다: {article['predefined_sentiment']}")

                # article_id 중복 검증
                if article["article_id"] in article_ids:
                     raise ValueError(f"데이터 항목 {i}에서 중복된 'article_id'가 발견되었습니다: {article['article_id']}")
                article_ids.add(article["article_id"])

            self.logger.info(f"{len(data)}개 데이터 항목 검증 완료. 유효함.")
        except ValueError as ve:
            self.logger.error(f"데이터 검증 실패: {str(ve)}")
            raise # 유효성 검사 오류를 다시 발생시킴
        except Exception as exc:
            self.logger.error(f"데이터 검증 중 예기치 않은 오류 발생: {str(exc)}", exc_info=True)
            raise

    def save_data(self, data: list):
        """
        생성된 데이터를 CSV 파일로 저장합니다.

        Args:
            data (list): 저장할 기사 데이터 리스트
        """
        if not data:
            self.logger.warning("저장할 데이터가 없습니다. 파일 저장을 건너<0xEB><0x9C><0x84>니다.")
            return

        try:
            self.logger.info("데이터 저장 시작")
            # 출력 디렉토리 생성 (없는 경우)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.logger.info(f"출력 디렉토리 생성: {self.output_dir}")

            # DataFrame 생성
            df = pd.DataFrame(data)

            # 파일명 생성 (날짜시간 포함)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, f"news_articles_{timestamp}.csv")

            # CSV 파일로 저장 (UTF-8 인코딩, 인덱스 제외)
            df.to_csv(filename, index=False, encoding='utf-8-sig')

            self.logger.info(f"데이터를 성공적으로 저장했습니다: {filename}")
            self.logger.info(f"저장된 데이터 미리보기 (상위 5개):\n{df.head().to_string()}")

        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}", exc_info=True)
            raise

def main():
    try:
        # 데이터 생성기 인스턴스 생성 (기사 8개 생성, 'generated_data' 폴더에 저장)
        generator = DataGenerator(num_articles=8, output_dir="generated_data")

        # 데이터 생성
        articles_data = generator.generate_data()

        # 데이터 검증
        generator.validate_data(articles_data)

        # 데이터 저장
        generator.save_data(articles_data)

        logging.info("데이터 생성, 검증 및 저장 프로세스 완료.")

    except ValueError as ve:
        logging.error(f"데이터 처리 중 유효성 검사 오류 발생: {str(ve)}")
    except Exception as exc:
        logging.critical(f"프로그램 실행 중 심각한 오류 발생: {str(exc)}", exc_info=True)

if __name__ == "__main__":
    main()