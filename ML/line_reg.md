# 회귀분석: Regression Analysis
- 1. 회귀분석의 목적
	- 독립변수로 종속변수를 예측하는 것
	- 종속변수는 수치형 데이터

- 2. 회귀분석 종류
	- 단순회귀 : simple regression
	- 다중회귀 : multiple regression
	- 다항회귀 : polynomial regression
		- 독립변수 1개이상, 종속변수 1개, 2차원 이상의 수식

## 모든 데이터가 직석위에 있지 않은경우 : 최소제곱법
#### 추세선의 수식
- $\hat{y} = a + bx$
 - $\hat{y}$ : 추세선 : 종속변수의 예측값
 - $x$ : 독립변수
 - $a$ : y절편(constant)
 - $b$ : 기울기(slope)
  - $\frac{\Delta{y}}{\Delta{x}}$ : y의 증가량 / x의 증가량
 - $a, b$ : 회귀계수

#### 모든 데이터의 오차를 구하는 수식 (MSE)
$$ \sum_{i=1}^n e_i^2 = \sum_{i=1}^n(y_i - a - bx_i)^2$$

#### MSE가 가장 작아지는 a와 b를 구하는 방법
- 경사하강법(gradient descent)
 -  a와 b의 미분값으로 기울기 구하기
$$ \sum_{i=1}^n e_i^2 = \sum_{i=1}^n(y_i - a - bx_i)^2$$
$$ \frac{\partial{L}}{\partial{b}} = -2 \sum_{i=1}^{n}(y_i - a - bx_i)x_i  $$

## 선형회귀 모델링
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression() # hyper-parameter 기입가능
model.fit(feature, target)
```
## 모델 성능평가 : MSE, RMSE, MAE, MAPE
```python
from sklearn.metrics import mean_absolute_error # MAE

pred = np.round(model.predict(feature))
mean_absolute_error(target, pred)
```

## 회귀분석 결과표
```python
# train/test 구분X
import statsmodels.api as sm
feature = sm.add_constant(feature)
model = sm.OLS(target, feature).fit()
pred = np.dot(feature, model.params)

print(model.summary2())
```
- 결과표 정리 (예시)
```
R-squared:     0.989

            Coef.    Std.Err.       t       P>|t| 
--------------------------------------------------
const      -0.0134     0.0354     -0.3800   0.7039
x1         48.3830     0.0354   1368.6758   0.0000
```
### 결과표 해석
- 1. Coef.
    - 회귀계수
    - 추세선의 파라미터
- 2. Std.Err.
    - 표준오차
    - 모집단의 여러개의 표본평균에 대한 표준편차
        - 표준오차가 작으면 회귀계수가 우연일 확률이 낮다.
        - 표준오차가 크면 회귀계수가 우연일 확률이 크다.
- 3. t
    - t-test
    - 회귀계수가 우연인지 아닌지에 대한 확률(표준오차와 반대)
        - t-test의 절대값이 클수록 우연일 확률이 낮다. (0과 멀리 떨어질수록)
        - t-test의 절대값이 작을수록 우연일 확률이 크다. (0에 가까울수록)
- 4. P>|t|
    - 유의확률 : probability value
    - 0.05 이하면 유의하다로 판단 
    - 연구가설이 맞다는것을 증명
- 5. R-squared
    - 모델의 분산 설명력
    - 높을수록 모델이 데이터를 잘 설명 -> 잘 예측

### R-squared
- 값이 높을수록 모델이 데이터를 잘 설명한다고 할수 있습니다.
- $R^2 = \frac{설명된분산}{종속변수의 전체분산} = 1 - \frac{SSE}{SST}$
- SST(sum of square total)
    - $SST = \sum(y_i-\bar{y_i})^2$
- SSE(sum of square error)
    - $SSE = \sum(y_i-\hat{y_i})^2$
- $R^2 = 0$ : 모델의 설명력 0%
- $R^2 = 1$ : 모델의 설명력 100%
- SSE가 클수록 $R^2$가 0에 근접
- SST가 클수록 $R^2$이 1에 근접
- 추세선의 데이터가 평균에서 멀리 떨어진 데이터 일수록 오차의 영향이 작아짐

#### R-squared가 높으면 항상 좋은가
- 잔차도(residual plot)가 랜덤하게 분포하는것을 확인해야 합니다
- 독립변수를 무조건 많이 넣어도 R-squared를 증가시킵니다.
- 높은 R-squared는 과적합(overfitting) 문제로부터 자유롭지 못합니다.

#### 결과 요약
- 회귀분석모델에서 좋은 모델을 만들기 위해서 feature selection과 nomalization을 잘해야 함
- feature selection을 잘 하기 위해서 요약표를 참고
- 요약표에서 중요한 내용
    - R-squared
        - 모델의 분산 설명력 의미 
        - 0 ~ 1 사이의 값을 가지며 1과 가까울수록 모델의 정확도가 올라감
    - coef. 
        - 회귀계수
        - 너무 0과 가까우면 모델에 영향을 거의 안 주게 되므로 계산량만 증가시킴
    - std.err.
        - 표준 에러
        - 회귀모델에서 평균에 대한 추세선의 데이터 퍼짐 정도를 나타냄
        - 높을수록 추세선의 신뢰도가 떨어짐
        - 그렇다고 낮은게 항상 좋은것은 아님 > 회귀계수가 0과 가까우면 낮게 나옴
