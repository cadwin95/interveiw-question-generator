<!-- [title]주가_방향_예측 -->

### 문제 제목
주가 방향 예측

### 문제 설명
**상황:**
금융 시장 예측을 위해 주가 데이터(종가)와 해당 일자의 뉴스 감성 점수 데이터를 활용합니다. 이 두 가지 정보를 결합하여 다음 거래일의 주가 상승/하락 방향을 예측하는 간단한 모델을 구현해야 합니다.

**요구사항:**
1.  제공된 `stock_price_simple.csv` (일별 종가)와 `news_sentiment_simple.csv` (일별 뉴스 감성 점수) 데이터를 로드하고, `date`를 기준으로 병합하세요.
2.  당일의 `close`(종가)와 `sentiment_score`(감성 점수)를 특징(Features)으로 사용하고, 다음 거래일의 종가가 당일 종가보다 상승했는지 여부 (상승 시 1, 그 외 0)를 타겟 변수(Label)로 생성하세요. (마지막 날 데이터는 타겟 생성 불가로 제외)
3.  생성된 특징과 타겟 변수를 사용하여 **Logistic Regression** 모델을 학습시키고, 전체 데이터에 대한 **Accuracy(정확도)**를 계산하여 출력하세요.

### 평가 포인트
-   데이터 로딩 및 병합 정확성 (Pandas 활용 능력)
-   특징 및 타겟 변수 생성 로직의 정확성
-   Logistic Regression 모델 구현 및 Accuracy 계산 능력 (Scikit-learn 활용 능력)

### 제약 조건
-   제한 시간: 30분
-   사용 모델: Logistic Regression으로 제한
-   주요 사용 라이브러리: Pandas, Scikit-learn 권장

## 데이터 생성

데이터 생성 코드는 `generate_data.py` 파일에 저장되어 있습니다.

### 실행 방법
다음 방법 중 하나를 선택하여 실행할 수 있습니다:

1. 자동 실행 스크립트 사용:
```bash
./run_data_generation_20250414_152244.sh
```

2. Python 스크립트 직접 실행:
```bash
python generate_data.py
```

### 로그 확인
실행 로그는 `logs/data_generation.log` 파일에서 확인할 수 있습니다.

