# AI 면접 문제 생성기 (Interview Question Generator)

이 도구는 Google Gemini AI를 활용하여 다양한 직업군과 주제에 맞춘 고품질 면접 문제를 자동으로 생성합니다. 여러 단계의 검증과 최적화 과정을 통해 실전에서 활용 가능한 면접 문제와 평가 기준을 제공합니다.

## 주요 기능

- 8개 직업군에 맞춤화된 면접 문제 생성
- 다양한 AI/기술 관련 주제 지원
- 9단계 검증 및 개선 프로세스
- 마크다운과 JSON 형식의 결과 저장
- 면접관을 위한 가이드라인 자동 생성
- 문제 해결을 위한 환경 설정 및 샘플 코드 생성

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install google-generativeai argparse
```

2. Google Gemini API 키 발급:
   - [Google AI Studio](https://ai.google.dev/)에서 API 키 발급
   - 코드 내 API 키 부분을 자신의 키로 변경

## 사용 방법

### 기본 사용법
```bash
python assesment.py
```

### 사용 가능한 매개변수

| 매개변수 | 설명 |
|---------|------|
| `--test-type` | 문제 생성 모드 선택 (`creative`, `complex`, `default`) |
| `--no-save` | 결과 파일 저장 안 함 |
| `--topic` | 특정 주제 지정 |
| `--topic-id` | 주제 ID로 선택 |
| `--list-topics` | 사용 가능한 주제 목록 표시 |
| `--job` | 직업군 지정 (기본값: ai_engineer) |
| `--job-id` | 직업군 ID로 선택 |
| `--list-jobs` | 사용 가능한 직업군 목록 표시 |

### 예시

1. 사용 가능한 직업군 목록 보기:
```bash
python assesment.py --list-jobs
```

2. 사용 가능한 주제 목록 보기:
```bash
python assesment.py --list-topics
```

3. 특정 직업군과 주제로 면접 문제 생성:
```bash
python assesment.py --job data_scientist --topic "비트코인 미래 시세 예측"
```

4. ID를 사용하여 직업군과 주제 선택:
```bash
python assesment.py --job-id 2 --topic-id 3 --test-type complex
```

## 생성 프로세스

각 면접 문제는 다음 9단계를 통해 정교화됩니다:

1. **초기 문제 기획**: 선택된 주제와 직업군에 맞는 기본 면접 문제 생성
2. **논리 검증**: 문제의 논리적 일관성과 명확성 검토
3. **난이도 조정**: 적절한 난이도로 문제 수준 조정
4. **창의성 강화**: 문제에 창의적 요소 추가
5. **복잡성 증가**: 난해한 요소 추가로 심층적 사고 유도
6. **단순화**: 불필요한 복잡성 제거 및 문제 정교화
7. **최종 검토**: 문제의 품질과 평가 기준 확정
8. **데이터 생성 코드**: 문제 해결에 필요한 환경 및 데이터 생성 코드 제공
9. **면접관 가이드**: 면접관이 참고할 수 있는 주요 개념 및 평가 포인트 제공

## 출력 결과

생성된 결과는 `interview_questions` 디렉토리에 저장됩니다:

- **JSON 파일**: 모든 단계의 결과를 포함한 데이터
- **마크다운 파일**: 구조화된 보고서 형식의 문서

## 지원하는 직업군

- AI 엔지니어
- 데이터 사이언티스트
- 백엔드 개발자
- 프론트엔드 개발자
- DevOps 엔지니어
- 보안 엔지니어
- 제품 매니저
- UX 디자이너

## 커스터마이징

새로운 주제나 직업군을 추가하려면 `interview_topics` 또는 `job_positions` 딕셔너리를 수정하세요.

## 주의사항

- Gemini API 키 보안에 주의하세요.
- 면접 문제를 그대로 사용하기보다 검토 후 활용하는 것을 권장합니다.
- 생성된 문제의 난이도와 적합성은 실제 면접 상황에 맞게 조정이 필요할 수 있습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 
