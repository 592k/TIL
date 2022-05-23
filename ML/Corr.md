# 상관계수 (Correlation Coefficient)
	import numpy as np

### 2. 분산(variance)
- 편차제곱의 평균

$ variance = \frac{\sum_{i=1}^n{(x_i-\bar{x})^2}}{n}, (\bar{x}:평균) $

```pyton
def variance(data):

	avg = np.average(data)
	var = 0
	for num in data:
		var += (num - avg) ** 2
	return var / len(data)

variance(data)
variance(data) ** 0.5

np.var(data)
np.std(data)
```

### 3. 공분산 (covariance)
- 평균 편차

$ covariance = \frac{\sum_{i=1}^{n}{(x_i-\bar{x})(y_i-\bar{y})}}{n}, (\bar{x}:x의 평균, \bar{y}:y의 평균) $

```python
def covariance(data1, data2):

	x_ = np.average(data1)
	y_ = np.average(data2)

	cov = 0
	for idx in range(len(data1)):
		cov += (data1[idx] -x_) * (data2[idx] - y_)
	return cov / (len(data1) - 1)
	
covariance(data1, data2)
```

### 4. 상관계수 (correlation coefficient)
- 공분산의 한계 극복
- -1 ~ 1 , 0에 가까울수록 상관도가 적다.
$ correlation-coefficient = \frac{공분산}{\sqrt{{x분산} \cdot {y분산}}} $

-최종 상관계수

$ r = \frac{\sum(x-\bar{x})(y-\bar{y})}{\sqrt{{\sum(x-\bar{x})^2}\cdot{\sum(y-\bar{y})^2}}} $

```python
def cc(data1, data2):

	x_ = np.average(data1)
	y_ = no.average(data2)

	cov, xv, yv = 0, 0, 0

	for idx in range(len(data1)):
		cov += (data1[idx] - x_) * (data2[idx] - y_)
		xv += (data1[idx] - x_) ** 2
		yv += (data2[idx] - y_) ** 2

	return cov / ((xv*yv) ** 0.5)


cc(data1, data2)
np.corrcoef(dtat1, data2)[0][1]
```
```python
import pandas as pd

# multiple columns dataframe -> corr -> 각 컬럼별 상관관계를 구할 수 있습니다
df1.corrwith(df2))
```

### 5. 결정계수 (cofficient of determination : R-squared)
- 상관계수의 제곱
- 클 수록 회귀분석을 통해 예측할수 있는 수치의 정도가 더 정확
```python
# pandas dataframe의 데이터를 pickle 파일로 저장하기
import pickle
with open("./data/dataframe.pkl", "rb") as f:
	datas = pickle.load(f)
```
