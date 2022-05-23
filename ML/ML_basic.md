## 프로세스
> 1. 문제정의
2. 데이터수집
3. 데이터탐색
4. 데이터전처리
5. 모델링
6. 모델평가
7. 결과보고서 작성

### 문제정의
통계적 배경지식이 중요(업무 base)

1개의 데이터셋에서 1개의 문제를 해결하는 것을 기본으로 함

문제정의 예시
> 기존고객 개인정보를 바탕으로 한 신규고객 고객군 분류  
- 종속변수 : 신규고객 고객군 분류
- 설명변수 : 기존고객 개인정보(성별, 나이, 지역, 소득, 신용정보, 가입정도등등)
- 모델 : 분류문제를 해결하기 위한 classification

복잡한 문제를 해결해야 할 경우 위의 1개 문제 = 1개 데이터셋의 원칙을 지키면 혼동이 덜 함   

### 데이터수집
분석에 필요한 설명변수, 종속변수에 따라 데이터 수집

> 사내데이터 활용  
데이터셋 활용  
크롤링  
API  
    
실제 수집이 불가능한 데이터가 있을 수 있음

### 데이터탐색

pandas, 시각화등 다양한 EDA 방법을 사용하여 수집데이터를 흝어보는 과정

수집된 데이터가 문제에 적합하지 않거나 사용이 불가능한 경우가 있을 수 있음

> 기술통계  
결측치, 이상치 탐색  
공분산 탐색  

### 데이터전처리
수집된 데이터를 모델링이 가능한 형태로 가공

추후 데이터전처리 특강에서 다룰예정

> 데이터 병합  
데이터 형변환  
더미화  
클래스불균형 해결  
차원축소  
재샘플링

### 모델링
실제로 가장 손쉬운 과정

문제해결에 알맞는 모델 탐색 및 최적모델 탐색

사람의 공수가 아닌 하드웨어 스펙에 따라 작업시간 단축

> 모델선택  
모수추정  
모델학습  
파라메터 서칭  

### 모델평가
고객군 분류가 되어있지 않은 신규고객데이터를 모델에 적용하여

기존분류 고객과 차이를 여러가지 방법으로 산출하여 평가

> 예측, 분류/오분류 평가  
설명력 평가  

### 결과보고서 작성
문제정의부터 모델평가까지 데이터분석 프로젝트 전 과정에 대한 진행경과, 결과, 문제점, 해결방법을 기술

> 시각화  
활용방안  
운영방안

```python
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y)
# train_x, train_y
# test_x, test_y

from sklearn.linear_model import LinearRegression
model = LinearRegression()
# model = LogisticRegression()

model.fit(train_x, train_y) # 모델 학습

from sklearn.metrics import mean_squared_error
import numpy as np

predicted = model.predict(test_x) # 모델 평가
print(predicted) # 예측 y

# 정답 y와 예측 y를 비교합니다. - 점수(=모델의 성능)를 측정 -> RMSE는 낮을수록 좋다!
RMSE = np.sqrt(mean_squared_error(test_y, predicted)) 

# Root Mean Square Error : RMSE
# regression 회귀모델 평가할떄 가장 많이 사용되는 평가지표 -rmse 낮을수록 좋은것
# classification 분류모델 평가할때 가장 많이 사용되는 평가지표 - accuracy, precision, recall, f1-score 높을수록 좋은 것
```
