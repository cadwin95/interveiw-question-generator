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
    def __init__(self, num_samples=100):
        """
        데이터 생성기 초기화

        Args:
            num_samples (int): 생성할 샘플 데이터 수
        """
        self.logger = logging.getLogger(__name__)
        self.num_samples = num_samples
        # 실제 뉴스 기사 대신 사용할 예시 텍스트 템플릿
        self.article_templates = [
            "오늘 [회사명]은(는) [제품/서비스] 출시를 발표했습니다. 이 새로운 [제품/서비스]은(는) [핵심 기능]을 특징으로 하며, [목표 시장]을 대상으로 합니다. 회사 관계자는 '[인용문]'라고 말했습니다.",
            "[도시명]에서 발생한 [사건 종류]로 인해 최소 [숫자]명의 사상자가 발생했습니다. 당국은 현재 [사건 원인]을 조사 중이며, 시민들에게 [안전 지침]을 따를 것을 당부했습니다.",
            "최근 연구에 따르면, [연구 주제]이(가) [건강/사회적 영향]에 미치는 영향이 큰 것으로 나타났습니다. [연구 기관]의 연구진들은 [연구 결과]를 발표하며, '[권장 사항]'을 제시했습니다.",
            "스포츠계 소식입니다. [팀/선수명]이(가) [상대팀/선수명]을(를) 상대로 [점수/결과]로 승리했습니다. 경기 중 [주요 선수]은(는) [활약상]을 보여주며 팀 승리를 이끌었습니다.",
            "경제 뉴스: [국가명] 중앙은행은 기준금리를 [비율]%로 [인상/인하/동결]했습니다. 이는 [경제 상황]에 대응하기 위한 조치로, 시장에서는 [시장 반응]을 보이고 있습니다."
        ]
        # LLM 요약문 템플릿 (일부는 환각 가능성을 내포)
        self.summary_templates = [
            "[회사명]이 [제품/서비스]를 출시했습니다. 주요 특징은 [핵심 기능]입니다.", # 비교적 사실 기반
            "[도시명]에서 [사건 종류] 발생, [숫자]명 피해. 원인은 [사건 원인 추정]으로 보입니다.", # 약간의 추정 포함 가능성
            "[연구 주제] 관련 연구 결과 발표. [건강/사회적 영향]에 영향. [잘못된 권장 사항]", # 환각 가능성 (잘못된 정보)
            "[팀/선수명] 승리 소식. [주요 선수] 활약.", # 사실 기반
            "[국가명] 금리 [인상/인하/동결]. 시장 반응은 [긍정적/부정적 전망].", # 약간의 해석 포함 가능성
            "[회사명]의 신제품은 [원문에 없는 기능]도 포함하고 있습니다.", # 명백한 환각
            "[사건 종류]의 원인은 [근거 없는 추측] 때문이라는 소문이 있습니다.", # 명백한 환각 (소문)
        ]
        # Placeholder 값들
        self.placeholders = {
            "[회사명]": ["테크코", "뉴스코프", "헬스라이프", "파이낸시아", "스포츠월드"],
            "[제품/서비스]": ["AI 비서", "뉴스 앱", "건강 모니터", "투자 플랫폼", "스포츠 스트리밍"],
            "[핵심 기능]": ["자연어 처리", "실시간 업데이트", "개인 맞춤 분석", "자동 거래", "고화질 중계"],
            "[목표 시장]": ["기업 고객", "일반 사용자", "환자 그룹", "개인 투자자", "스포츠 팬"],
            "[인용문]": ["혁신을 통해 시장을 선도할 것", "정확한 정보 전달이 최우선", "사용자 건강 증진에 기여", "안정적인 수익 창출 목표", "팬들에게 즐거움을 선사"],
            "[도시명]": ["서울", "뉴욕", "런던", "도쿄", "파리"],
            "[사건 종류]": ["화재", "교통사고", "자연재해", "시위", "축제"],
            "[숫자]": ["수십", "10여", "5", "100", "수백"],
            "[사건 원인]": ["전기 누전", "운전 부주의", "기상 악화", "사회적 갈등", "안전 미비"],
            "[사건 원인 추정]": ["누전으로 추정", "과속으로 추정", "폭우로 추정", "의견 충돌로 추정", "관리 부실로 추정"], # 요약문에 들어갈 추정
            "[안전 지침]": ["외출 자제", "교통 통제 협조", "대피소 이동", "안전 거리 유지", "행사장 질서 유지"],
            "[연구 주제]": ["스마트폰 사용 시간", "규칙적인 운동", "미세먼지 농도", "온라인 학습 효과", "사회적 고립감"],
            "[건강/사회적 영향]": ["수면의 질", "심혈관 건강", "호흡기 질환", "학업 성취도", "정신 건강"],
            "[연구 기관]": ["국립보건원", "서울대학교", "한국과학기술원", "세계보건기구", "교육개발원"],
            "[연구 결과]": ["부정적 상관관계 확인", "긍정적 효과 입증", "유의미한 연관성 발견", "학습 효율 증대 확인", "우울감 증가 확인"],
            "[권장 사항]": ["사용 시간 제한 권고", "주 3회 이상 운동 권장", "마스크 착용 필수", "블렌디드 러닝 활용 제안", "사회 활동 참여 독려"],
            "[잘못된 권장 사항]": ["스마트폰 완전 금지", "매일 5시간 이상 운동", "외출 절대 금지", "모든 수업 온라인 대체", "혼자 시간 보내기"], # 환각 요약용
            "[팀/선수명]": ["레드 드래곤즈", "블루 타이거즈", "김민수", "박지현", "이영웅"],
            "[상대팀/선수명]": ["화이트 울브즈", "블랙 팬서스", "최강호", "윤세아", "강철민"],
            "[점수/결과]": ["3:1", "10점차", "역전", "완승", "신승"],
            "[주요 선수]": ["공격수 김철수", "미드필더 박하나", "수비수 이지훈", "에이스 이영웅", "신인 선수 최유리"],
            "[활약상]": ["해트트릭 기록", "결승골 성공", "무실점 방어", "MVP 선정", "데뷔전 활약"],
            "[국가명]": ["대한민국", "미국", "유럽연합", "일본", "중국"],
            "[비율]": ["0.25", "0.5", "1.0", "0.1", "0.75"],
            "[인상/인하/동결]": ["인상", "인하", "동결"],
            "[경제 상황]": ["물가 상승 압력", "경기 침체 우려", "안정적인 성장세", "수출 부진", "고용 시장 불안"],
            "[시장 반응]": ["주가 하락", "환율 변동성 확대", "투자 심리 위축", "긍정적 기대감", "관망세 유지"],
            "[긍정적/부정적 전망]": ["긍정적 전망", "부정적 전망", "혼조세 예상", "단기적 충격 예상", "장기적 안정 기대"], # 요약문에 들어갈 전망
            "[원문에 없는 기능]": ["자동 비행 기능", "시간 여행 기능", "텔레파시 기능", "무한 에너지 제공", "외계어 번역"], # 명백한 환각용
            "[근거 없는 추측]": ["외계인의 소행", "정부의 음모", "고대 유물의 저주", "경쟁사의 방해 공작", "초자연적 현상"] # 명백한 환각용
        }

    def _fill_template(self, template):
        """템플릿의 플레이스홀더를 랜덤 값으로 채웁니다."""
        filled_text = template
        # 템플릿 내 플레이스홀더를 순회하며 값 채우기 (순서 중요할 수 있음)
        sorted_placeholders = sorted(self.placeholders.keys(), key=len, reverse=True)
        
        # 각 플레이스홀더에 대해 랜덤 값 선택
        selected_values = {holder: random.choice(values) for holder, values in self.placeholders.items()}

        for holder in sorted_placeholders:
            if holder in filled_text:
                 # 이미 선택된 값 사용
                 filled_text = filled_text.replace(holder, selected_values[holder], 1) # 1번만 치환하여 동일 문서 내 일관성 유지 시도
        
        # 혹시 치환되지 않은 플레이스홀더가 있다면 기본값 처리 (이 경우는 거의 없어야 함)
        for holder in sorted_placeholders:
             if holder in filled_text:
                  filled_text = filled_text.replace(holder, f"({holder.strip('[]')})")
                     
        return filled_text

    def generate_data(self):
        """
        뉴스 기사, LLM 요약문, 환각 여부 레이블 데이터를 생성합니다.
        """
        try:
            self.logger.info(f"{self.num_samples}개의 샘플 데이터 생성 시작")
            data = []
            for i in range(self.num_samples):
                article_template = random.choice(self.article_templates)
                original_text = self._fill_template(article_template)

                # 요약문 생성 (약 30% 확률로 환각 가능성이 높은 템플릿 사용)
                is_hallucinating_intended = random.random() < 0.3
                if is_hallucinating_intended:
                    # 환각을 유도하는 템플릿 중에서 선택 (예: 템플릿 인덱스 2, 5, 6)
                    potential_hallucination_indices = [idx for idx, tmpl in enumerate(self.summary_templates) if "[잘못된 권장 사항]" in tmpl or "[원문에 없는 기능]" in tmpl or "[근거 없는 추측]" in tmpl]
                    if not potential_hallucination_indices: # 만약 환각 템플릿이 없다면 일반 템플릿 사용
                        summary_template = random.choice(self.summary_templates)
                    else:
                         summary_template = self.summary_templates[random.choice(potential_hallucination_indices)]
                else:
                    # 비교적 사실 기반 템플릿 선택
                    non_hallucination_indices = [idx for idx, tmpl in enumerate(self.summary_templates) if "[잘못된 권장 사항]" not in tmpl and "[원문에 없는 기능]" not in tmpl and "[근거 없는 추측]" not in tmpl]
                    if not non_hallucination_indices: # 만약 사실기반 템플릿이 없다면 아무거나 사용
                         summary_template = random.choice(self.summary_templates)
                    else:
                         summary_template = self.summary_templates[random.choice(non_hallucination_indices)]
                
                llm_summary = self._fill_template(summary_template)

                # 환각 여부 레이블링 (여기서는 템플릿 기반으로 단순화)
                is_hallucinating = "[잘못된 권장 사항]" in summary_template or "[원문에 없는 기능]" in summary_template or "[근거 없는 추측]" in summary_template


                data.append({
                    'article_id': f'NWS_{i+1:04d}',
                    'original_text': original_text,
                    'llm_summary': llm_summary,
                    'is_hallucinating': is_hallucinating # 실제로는 이 부분을 평가하거나 모델로 예측해야 함
                })

            df = pd.DataFrame(data)
            self.logger.info("데이터 생성 완료")
            return df

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}")
            raise

    def validate_data(self, data):
        """
        생성된 데이터의 유효성을 검증합니다.
        """
        try:
            self.logger.info("데이터 검증 시작")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("데이터는 Pandas DataFrame이어야 합니다.")

            if data.empty:
                raise ValueError("데이터프레임이 비어있습니다.")

            required_columns = ['article_id', 'original_text', 'llm_summary', 'is_hallucinating']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"필수 컬럼이 누락되었습니다: {required_columns}")

            # Null 값 체크
            if data.isnull().values.any():
                 self.logger.warning("데이터에 Null 값이 포함되어 있습니다.")
                 # 필요시 raise ValueError("데이터에 Null 값이 포함되어 있습니다.") 로 변경 가능

            # 데이터 타입 체크 (간단하게)
            if not pd.api.types.is_string_dtype(data['article_id']):
                 self.logger.warning("article_id 컬럼 타입이 문자열이 아닐 수 있습니다.")
            if not pd.api.types.is_string_dtype(data['original_text']):
                 self.logger.warning("original_text 컬럼 타입이 문자열이 아닐 수 있습니다.")
            if not pd.api.types.is_string_dtype(data['llm_summary']):
                self.logger.warning("llm_summary 컬럼 타입이 문자열이 아닐 수 있습니다.")
            if not pd.api.types.is_bool_dtype(data['is_hallucinating']):
                self.logger.warning("is_hallucinating 컬럼 타입이 불리언이 아닐 수 있습니다.")


            self.logger.info("데이터 검증 완료")
            return True

        except (TypeError, ValueError) as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}")
            raise
        except Exception as exc:
            self.logger.error(f"데이터 검증 중 예상치 못한 오류 발생: {str(exc)}")
            raise


    def save_data(self, data, filename="generated_news_data.csv"):
        """
        생성된 데이터를 CSV 파일로 저장합니다.
        """
        try:
            self.logger.info(f"데이터를 '{filename}' 파일로 저장 시작")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("저장할 데이터는 Pandas DataFrame이어야 합니다.")

            # 저장 경로 설정 (현재 스크립트 위치 기준)
            save_path = os.path.join(os.path.dirname(__file__), filename) if '__file__' in globals() else filename

            data.to_csv(save_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"데이터 저장 완료: {save_path}")

        except TypeError as exc:
             self.logger.error(f"데이터 저장 중 타입 오류 발생: {str(exc)}")
             raise
        except Exception as exc:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(exc)}")
            raise

def main():
    try:
        num_data_samples = 150 # 생성할 데이터 샘플 수 지정
        generator = DataGenerator(num_samples=num_data_samples)
        generated_data = generator.generate_data()
        if generator.validate_data(generated_data):
            generator.save_data(generated_data)
            logging.info("데이터 생성, 검증 및 저장 완료")
        else:
            logging.error("데이터 검증 실패로 저장을 진행하지 않습니다.")

    except Exception as exc:
        logging.error(f"프로그램 실행 중 오류 발생: {str(exc)}")
        # raise # 필요에 따라 에러를 다시 발생시킬 수 있음

if __name__ == "__main__":
    main()