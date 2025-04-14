import google.generativeai as genai
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Generative AI 설정
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

class CodeReviewer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def review_question(self, question_content):
        """면접 문제를 검토하고 피드백을 제공합니다."""
        prompt = f"""
        다음 면접 문제를 검토하고 피드백을 제공해주세요. 정합성(consistency)과 명확성을 중점적으로 검토해주세요.

        **검토 기준:**
        1. 문제의 명확성
           - 문제 설명이 명확하고 모호하지 않은가?
           - 용어와 개념이 정확하게 정의되어 있는가?
           - 입력과 출력이 명확하게 지정되어 있는가?

        2. 논리적 일관성
           - 문제의 각 부분이 서로 모순되지 않는가?
           - 전제 조건과 제약 조건이 일관성 있게 연결되어 있는가?
           - 평가 기준이 문제의 목표와 일치하는가?

        3. 요구사항의 정확성
           - 모든 필수 요구사항이 명시되어 있는가?
           - 요구사항이 구체적이고 측정 가능한가?
           - 불필요한 요구사항이 포함되어 있지 않은가?

        4. 해결 가능성
           - 문제가 주어진 시간 내에 해결 가능한가?
           - 필요한 정보가 모두 제공되어 있는가?
           - 해결 방법이 명확하게 존재하는가?

        **면접 문제:**
        {question_content}

        **출력 형식:**
        ### 문제 검토 결과
        [전체적인 검토 결과 요약]

        ### 명확성 검토
        - [명확성 관련 발견 사항 1]
        - [명확성 관련 발견 사항 2]
        - [명확성 관련 발견 사항 3]

        ### 일관성 검토
        - [일관성 관련 발견 사항 1]
        - [일관성 관련 발견 사항 2]
        - [일관성 관련 발견 사항 3]

        ### 요구사항 검토
        - [요구사항 관련 발견 사항 1]
        - [요구사항 관련 발견 사항 2]
        - [요구사항 관련 발견 사항 3]

        ### 개선 제안
        - [개선 제안 1]
        - [개선 제안 2]
        - [개선 제안 3]

        **중요:**
        - 각 섹션에 최소 1개 이상의 발견 사항이나 제안을 포함해주세요.
        - 발견된 문제가 있다면 [문제 수정] 태그를 사용하여 구체적인 수정 제안을 해주세요.
        - 모든 검토 결과는 마크다운 형식으로 작성해주세요.
        """
        try:
            response = model.generate_content(prompt)
            if not response.text:
                self.logger.warning("문제 검토 결과가 비어있습니다.")
                return "검토 결과가 없습니다. 다시 시도해주세요."
            return response.text
        except Exception as e:
            self.logger.error(f"문제 검토 중 오류 발생: {str(e)}")
            return f"문제 검토 중 오류가 발생했습니다: {str(e)}"

    def review_code(self, code_content, question_content):
        """생성된 코드를 검토하고 피드백을 제공합니다."""
        prompt = f"""
        다음 면접 문제에 대한 코드를 검토하고 피드백을 제공해주세요. 정확성과 일관성을 중점적으로 검토해주세요.

        **면접 문제:**
        {question_content}

        **코드:**
        {code_content}

        **검토 기준:**
        1. 코드의 정확성
           - 코드가 문제의 요구사항을 정확히 구현하는가?
           - 입력과 출력이 문제의 명세와 일치하는가?
           - 예외 상황이 적절히 처리되는가?

        2. 코드의 일관성
           - 코드 스타일이 일관되게 유지되는가?
           - 변수명과 함수명이 일관된 규칙을 따르는가?
           - 코드 구조가 논리적으로 일관성이 있는가?

        3. 요구사항 구현 검증
           - 모든 필수 기능이 구현되어 있는가?
           - 구현된 기능이 문제의 요구사항과 정확히 일치하는가?
           - 누락된 요구사항이 없는가?

        4. 에러 처리
           - 예외 상황이 적절히 처리되는가?
           - 에러 메시지가 명확하고 유용한가?
           - 로깅이 적절히 구현되어 있는가?

        **출력 형식:**
        ### 코드 검토 결과
        [전체적인 검토 결과 요약]

        ### 정확성 검토
        - [정확성 관련 발견 사항 1]
        - [정확성 관련 발견 사항 2]
        - [정확성 관련 발견 사항 3]

        ### 일관성 검토
        - [일관성 관련 발견 사항 1]
        - [일관성 관련 발견 사항 2]
        - [일관성 관련 발견 사항 3]

        ### 요구사항 구현 검증
        - [구현 검증 관련 발견 사항 1]
        - [구현 검증 관련 발견 사항 2]
        - [구현 검증 관련 발견 사항 3]

        ### 개선 제안
        - [개선 제안 1]
        - [개선 제안 2]
        - [개선 제안 3]

        **중요:**
        - 각 섹션에 최소 1개 이상의 발견 사항이나 제안을 포함해주세요.
        - 발견된 문제가 있다면 [코드 수정] 태그를 사용하여 구체적인 수정 제안을 해주세요.
        - 모든 검토 결과는 마크다운 형식으로 작성해주세요.
        """
        try:
            response = model.generate_content(prompt)
            if not response.text:
                self.logger.warning("코드 검토 결과가 비어있습니다.")
                return "검토 결과가 없습니다. 다시 시도해주세요."
            return response.text
        except Exception as e:
            self.logger.error(f"코드 검토 중 오류 발생: {str(e)}")
            return f"코드 검토 중 오류가 발생했습니다: {str(e)}"

    def generate_review_report(self, question_dir):
        """문제와 코드를 검토하고 종합 보고서를 생성합니다."""
        try:
            # 문제 디렉토리 경로 확인
            if not os.path.exists(question_dir):
                self.logger.error(f"문제 디렉토리가 존재하지 않습니다: {question_dir}")
                return None

            self.logger.info(f"검토 중: {question_dir}")

            # 문제 파일 읽기
            question_file = os.path.join(question_dir, "question.md")
            if not os.path.exists(question_file):
                self.logger.warning(f"question.md 파일이 존재하지 않습니다: {question_file}")
                return None

            with open(question_file, "r", encoding="utf-8") as f:
                question_content = f.read()

            # 코드 파일 읽기
            code_file = os.path.join(question_dir, "generate_data.py")
            code_content = ""
            if os.path.exists(code_file):
                with open(code_file, "r", encoding="utf-8") as f:
                    code_content = f.read()

            # 문제 검토
            question_review = self.review_question(question_content)
            
            # 코드 검토
            code_review = None
            if code_content:
                code_review = self.review_code(code_content, question_content)

            # 검토 보고서 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            review_path = os.path.join(question_dir, f"review_{timestamp}.md")
            
            with open(review_path, "w", encoding="utf-8") as f:
                f.write(f"**검토 일시:** {datetime.now().isoformat()}\n\n")
                f.write(f"**문제 디렉토리:** {os.path.basename(question_dir)}\n\n")
                
                # 문제 검토 결과 기록
                if question_review:
                    f.write("## 문제 검토 결과\n\n")
                    f.write(question_review + "\n\n")
                    
                    # 문제 수정사항 추출 및 적용
                    f.write("## 문제 수정사항\n\n")
                    question_fixes = self._extract_fixes(question_review, "[문제 수정]")
                    for fix in question_fixes:
                        f.write(f"- {fix}\n")
                    
                    # 문제 파일 수정
                    if question_fixes:
                        self._apply_question_fixes(question_file, question_fixes)
                
                # 코드 검토 결과 기록
                if code_review:
                    f.write("\n## 코드 검토 결과\n\n")
                    f.write(code_review + "\n\n")
                    
                    # 코드 수정사항 추출 및 적용
                    f.write("## 코드 수정사항\n\n")
                    code_fixes = self._extract_fixes(code_review, "[코드 수정]")
                    for fix in code_fixes:
                        f.write(f"- {fix}\n")
                    
                    # 코드 파일 수정
                    if code_fixes:
                        self._apply_code_fixes(code_file, code_fixes)

            self.logger.info(f"검토 보고서가 생성되었습니다: {review_path}")
            return review_path

        except Exception as e:
            self.logger.error(f"검토 보고서 생성 중 오류 발생: {str(e)}")
            return None

    def _extract_fixes(self, review_content, fix_type):
        """검토 내용에서 수정사항을 추출합니다."""
        fixes = []
        lines = review_content.split("\n")
        
        for line in lines:
            if fix_type in line:
                # 수정사항만 추출 (토큰 제거)
                fix = line.replace(fix_type, "").strip()
                if fix:
                    fixes.append(fix)
        
        return fixes

    def _apply_question_fixes(self, question_file, fixes):
        """문제 파일에 수정사항을 적용합니다."""
        try:
            with open(question_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 수정사항 적용
            for fix in fixes:
                # 문제 제목 수정
                if "문제 제목" in fix:
                    # 문제 제목 섹션 찾기
                    start_idx = content.find("### 문제 제목")
                    if start_idx != -1:
                        # 다음 섹션 시작 전까지가 문제 제목
                        end_idx = content.find("###", start_idx + 1)
                        if end_idx == -1:
                            end_idx = len(content)
                        # 새로운 문제 제목으로 교체
                        content = content[:start_idx] + "### 문제 제목\n\n" + fix.replace("문제 제목: ", "") + "\n\n" + content[end_idx:]
                
                # 문제 설명 수정
                elif "문제 설명" in fix:
                    # 문제 설명 섹션 찾기
                    start_idx = content.find("### 문제 설명")
                    if start_idx != -1:
                        # 다음 섹션 시작 전까지가 문제 설명
                        end_idx = content.find("###", start_idx + 1)
                        if end_idx == -1:
                            end_idx = len(content)
                        # 새로운 문제 설명으로 교체
                        content = content[:start_idx] + "### 문제 설명\n\n" + fix.replace("문제 설명: ", "") + "\n\n" + content[end_idx:]
                
                # 평가 포인트 수정
                elif "평가 포인트" in fix:
                    # 평가 포인트 섹션 찾기
                    start_idx = content.find("### 평가 포인트")
                    if start_idx != -1:
                        # 다음 섹션 시작 전까지가 평가 포인트
                        end_idx = content.find("###", start_idx + 1)
                        if end_idx == -1:
                            end_idx = len(content)
                        # 평가 포인트를 리스트 형식으로 변환
                        points = fix.replace("평가 포인트: ", "").split(";")
                        points_text = "\n".join([f"- {point.strip()}" for point in points if point.strip()])
                        # 새로운 평가 포인트로 교체
                        content = content[:start_idx] + "### 평가 포인트\n\n" + points_text + "\n\n" + content[end_idx:]
                
                # 제약 조건 수정
                elif "제약 조건" in fix:
                    # 제약 조건 섹션 찾기
                    start_idx = content.find("### 제약 조건")
                    if start_idx != -1:
                        # 다음 섹션 시작 전까지가 제약 조건
                        end_idx = content.find("###", start_idx + 1)
                        if end_idx == -1:
                            end_idx = len(content)
                        # 제약 조건을 리스트 형식으로 변환
                        constraints = fix.replace("제약 조건: ", "").split(";")
                        constraints_text = "\n".join([f"- {constraint.strip()}" for constraint in constraints if constraint.strip()])
                        # 새로운 제약 조건으로 교체
                        content = content[:start_idx] + "### 제약 조건\n\n" + constraints_text + "\n\n" + content[end_idx:]
            
            with open(question_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            self.logger.info(f"문제 파일 수정 완료: {question_file}")
        except Exception as e:
            self.logger.error(f"문제 파일 수정 중 오류 발생: {str(e)}")

    def _apply_code_fixes(self, code_file, fixes):
        """코드 파일에 수정사항을 적용합니다."""
        try:
            with open(code_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 수정사항 적용
            for fix in fixes:
                # 함수 추가
                if "함수 추가" in fix:
                    # 함수 내용 추출
                    func_content = fix.replace("함수 추가: ", "")
                    # main 함수 전에 추가
                    main_idx = content.find("def main()")
                    if main_idx != -1:
                        content = content[:main_idx] + func_content + "\n\n" + content[main_idx:]
                
                # 함수 수정
                elif "함수 수정" in fix:
                    # 함수 이름과 내용 추출
                    func_name = fix.split(":")[1].strip()
                    func_content = fix.split(":")[2].strip()
                    # 함수 찾기
                    func_start = content.find(f"def {func_name}")
                    if func_start != -1:
                        # 함수 끝 찾기
                        func_end = content.find("def ", func_start + 1)
                        if func_end == -1:
                            func_end = len(content)
                        # 함수 내용 교체
                        content = content[:func_start] + func_content + "\n\n" + content[func_end:]
                
                # 코드 수정
                elif "코드 수정" in fix:
                    # 수정할 코드와 새로운 코드 추출
                    old_code = fix.split("->")[0].replace("코드 수정: ", "").strip()
                    new_code = fix.split("->")[1].strip()
                    # 코드 교체
                    content = content.replace(old_code, new_code)
                
                # 클래스 수정
                elif "클래스 수정" in fix:
                    # 클래스 이름과 내용 추출
                    class_name = fix.split(":")[1].strip()
                    class_content = fix.split(":")[2].strip()
                    # 클래스 찾기
                    class_start = content.find(f"class {class_name}")
                    if class_start != -1:
                        # 클래스 끝 찾기
                        class_end = content.find("class ", class_start + 1)
                        if class_end == -1:
                            class_end = len(content)
                        # 클래스 내용 교체
                        content = content[:class_start] + class_content + "\n\n" + content[class_end:]
                
                # 메서드 수정
                elif "메서드 수정" in fix:
                    # 클래스 이름, 메서드 이름, 내용 추출
                    parts = fix.split(":")
                    class_name = parts[1].strip()
                    method_name = parts[2].strip()
                    method_content = parts[3].strip()
                    # 클래스와 메서드 찾기
                    class_start = content.find(f"class {class_name}")
                    if class_start != -1:
                        class_end = content.find("class ", class_start + 1)
                        if class_end == -1:
                            class_end = len(content)
                        class_content = content[class_start:class_end]
                        # 메서드 찾기
                        method_start = class_content.find(f"def {method_name}")
                        if method_start != -1:
                            method_end = class_content.find("def ", method_start + 1)
                            if method_end == -1:
                                method_end = len(class_content)
                            # 메서드 내용 교체
                            new_class_content = class_content[:method_start] + method_content + "\n\n" + class_content[method_end:]
                            content = content[:class_start] + new_class_content + content[class_end:]
            
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            self.logger.info(f"코드 파일 수정 완료: {code_file}")
        except Exception as e:
            self.logger.error(f"코드 파일 수정 중 오류 발생: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='면접 문제 및 코드 정합성 검토 도구')
    parser.add_argument('--dir', type=str, required=True, help='검토할 문제 디렉토리 경로')
    args = parser.parse_args()
    
    # interview_questions 폴더 경로 추가
    question_dir = os.path.join("interview_questions", args.dir)
    
    reviewer = CodeReviewer()
    review_path = reviewer.generate_review_report(question_dir)
    
    if review_path:
        print(f"\n검토가 완료되었습니다. 보고서는 다음 위치에서 확인할 수 있습니다:")
        print(f"- {review_path}")
    else:
        print("\n검토 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 