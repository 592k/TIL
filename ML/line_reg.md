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
