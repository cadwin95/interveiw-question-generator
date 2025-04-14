import os
import re

def read_question_title(question_path):
    try:
        with open(question_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 1. HTML 주석에서 [title] 토큰을 찾음
            title_match = re.search(r'<!--\s*\[title\](.*?)-->', content)
            if title_match:
                return title_match.group(1).strip()
                
            # 2. "### 문제 제목" 다음 줄을 찾음
            title_match = re.search(r'###\s*문제\s*제목\s*\n(.*?)\n', content)
            if title_match:
                return title_match.group(1).strip()
                
            # 3. "### 최종 문제" 다음 줄을 찾음
            title_match = re.search(r'###\s*최종\s*문제\s*\n(.*?)\n', content)
            if title_match:
                return title_match.group(1).strip()
                
            # 4. 일반 마크다운 제목 형식 (### 또는 ##)
            title_match = re.search(r'^#+\s*(.*?)\n', content)
            if title_match:
                return title_match.group(1).strip()
                
            return None
    except Exception as e:
        print(f"Error reading {question_path}: {e}")
        return None

def sanitize_folder_name(title):
    if not title:
        return None
    # 특수문자 제거 및 공백을 언더스코어로 변경
    sanitized = re.sub(r'[^\w\s-]', '', title)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    return sanitized

def main():
    base_dir = "interview_questions"
    
    # 각 폴더에 대해 처리
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            question_path = os.path.join(folder_path, "question.md")
            if os.path.exists(question_path):
                title = read_question_title(question_path)
                if title:
                    new_name = sanitize_folder_name(title)
                    if new_name and new_name != folder:
                        new_path = os.path.join(base_dir, new_name)
                        try:
                            os.rename(folder_path, new_path)
                            print(f"Renamed '{folder}' to '{new_name}'")
                        except Exception as e:
                            print(f"Error renaming {folder}: {e}")

if __name__ == "__main__":
    main() 