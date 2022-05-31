## 다중선형회귀분석 (multiple regression)
feature filtering -> feature 수 줄이기

#### 다중공선성(Multicollinearity)
- feature들 사이의 상관계수 확인
```python
sns.heatmap(df.corr(), annot=True)
```

- VIF (Variance inflation factor)
- $$VIF_i = \frac{1}{1-R_i^2}$$
- VIF가 10이상인 경우 다중공선성으로 판단
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.DataFrame({
    "VIF Factor": [variance_inflation_factor(features_df.values, idx) for idx in range(features_df.shape[1])],
    "features": features_df.columns,
})
```
모델 생성해서 요약표로 확인
```python
import statsmodels.api as sm

feature_2 = sm.add_constant(df[["TV", "radio"]])
model_2 = sm.OLS(df["sales"], feature_2).fit() # summary output 가능
print(model_2.summary2())
```

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
    - t
        - t-test
        - 회귀계수가 우연인지 확인하는 지표
        - 0과 가까울수록 우연일 확률이 높음
    - P>|t|
        - p-value
        - t-test를 0 ~ 1 사이의 수치로 변환 한것
        - 대체로 0.05 이하의 수치를 가지면 해당 feature는 유의한것으로 판단
- 잔차도
    - 실제 데이터와 예측 데이터의 오차에 대한 분산
    - 그래프로 그려봤을때 일정한 잔차가 나오는 모델이 좋은 모델
	
