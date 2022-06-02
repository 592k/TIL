## Random Forest

### 1. 의사결정나무 모델의 단점
- 중간에 에러가 발생하면 다음단계로 에러 전파
- 학습 데이터의 미세한 변동에도 최종 결과에 크게 영향
- 최종 노드 갯수를 늘리면 과적합 위험 : Low Bias, Large Variance
> -  Bias -> 편향, underfitting check
> -  Variance -> 분산, overfitting check

Random Forest 사용으로 보완

### 2. 앙상블 (여러개의 Decision Tree 모음)
- 여러 모델의 예측을 다수결 법칙과 평균을 이용
- 여러 모델의 장점(각 모델의 장점을 알아내기 어려움..) 을 결합
- 조건을 만족시켜야 하나의 모델을 사용할 때보다 성능이 우수함
> 1. Base 모델이 서로 독립적일 때 (feature와 model이 독립적)
> 2. 무작위 예측보다 성능이 좋은 경우
> 3. 비슷한 수준의 model을 사용

- Ensemble 관련모델
1. 확률기반 모델: 나이브 베이즈
2. 커널기반 모델: SVM

>- linear regression, logistic regression X
>- rf(xgboost, adaboost), nb, svm

각 모델의 특성을 파악 해서 앙상블

- 앙상블 모델의 에러율
$$e_{ensemble} = \sum^N_{i=[N/2]} \begin{bmatrix}N\\i\\\end{bmatrix} e^i(1-e)^N-1$$


$$ e: Base 모델의 오류율, N : Base 모델의 수 $$

- Base 모델의 오류율보다 앙상블 모델의 오류율이 작아지는 0.5 이하의 오류율을 가지는 모델들을 사용해야 한다.
- ex) 1+1+1+1+.. => 1.2

- 의사결정나무 모델은 Base 모델로 활용도가 높음
> - 데이터크기가 큰 경우도 모델을 빨리 구축가능
> - 데이터 분포에 대한 전제가 필요하지 않음

### 3. Random Forest 개요
- 다수의 Decision Tree model에 의한 예측을 종합하는 "앙상블"방법
- 하나의 Decision Tree model보다 좋은 예측
- 관측치에 비해 변수의 수가 많은 고차원 데이터에서 <u>**중요 변수 선택기법**</u>으로 활용

- **Bootstrap** 기법을 이용하여 다수의 train data 생성
- train data로 decision tree 모델 구축 (무작위 feature 사용) (복원 추출)
- 여러개의 모델에 대한 결과를 병합하여 예측

- 아이디어 : Diversity, Random
    * 상이성 확보
        - Bagging : 여러개의 Train data를 생성하여 각 데이터 마다 개별의 의사나무모델 구축
    * 랜덤성 확보
        - Random Subspace : 의사결정나무 모델 구축시 변수를 무작위로 선택

### 4. Bagging
- Bootsrap Aggregting
- 다수의 학습 데이터셋을 생성해서 다수의 모델을 만들고 다수의 모델을 하나의 모델로 결합하는 방법

Bootstrap sampling -> Model training -> Model forecasting -> Result aggregating

#### Aggregating
1. 방법1: Hard Voting
    - Predicted class lable 갯수로 결정
    - $\sum_{j=1}^{n}I(\hat{y}_j=0) = 4, \sum_{j=1}^{n}I(\hat{y}_j=1) = 6$
        - 예측된 label이 0보다 1이 더 많으므로 1로 예측
2. 방법2: Soft Voting
    - Training Accuracy의 가중평균으로 예측값 결정
    - $\frac{\sum(TrainAcc)I(\hat{y}=0)}{\sum(TrainAcc)}=0.424$
    - $\frac{\sum(TrainAcc)I(\hat{y}=1)}{\sum(TrainAcc)}=0.576$
        - 예측된 label의 가중평균이 0보다 1이 더 높으므로 1로 예측
3. 방법3: 각 모델의 확률이 일정해야 편향되지 않음
    - 모든 모델에서 각 lable의 확률값의 평균으로 결정
    - $\frac{\sum{P(y=0)}}{n}=0.371, \frac{\sum{P(y=1)}}{n}=0.629$
        - 1로 예측한 확률의 평균이 높기 때문에 1로 예측

### )) Boosting
- **정확성 향상**을 위해서 사용
- 앞서 만든 모델이 오분류로 인한 케이스에 더 높은 가중치를 부여하여 모델 수정
- 부스팅을 하는 목적은 '정확성을 향상'시키기 위함에 있다. 앞서 만든 모델이 오분류로 인한 케이스에 더 높은 가중치를 부여함으로써 이를 더 잘 해결할 수 있는 모델이 되도록 모델을 수정해 나간다. 연속형 변수(numeric data)는 가중 평균(weighted aveage, median)으로 예측 결과값을 합친다. 배깅이 병렬적(parallel)으로 학습한다면 부스팅은 순차적(sequential)으로 모델만든다고 할 수 있다. 정답보다 오답에 더 높은 가중치를 줌으로써 오답에 더욱 집중 할 수 있게 한다. 이로써 정확도가 더 높아질 수 있지만 그만큼 이상치에 취약해질 수 있다. 대표적인 알고리즘에는 AdaBoost, xgBoost, GBM 등이 있다. 이중에서 xgBoost를 가장 많이 사용한다.

- 회기 모형에 대한 Boosting 기법(Adaptive Boosting AdaBoost)
> 1. 표본에 모형을 적합시켜 예측치와 오차를 산출
> 2. 오차가 큰 개체에는 큰 가중치, 오차가 작은 개체에는 작은 가중치
> 3. 변경된 가중치로 모형에 재적합, 후 재실행 (일정 횟수 반복)
> 4. 최종으로 이제까지 도출된 모형 총합.

### 5. Random Subspace
- 하나의 모델 만들기
    * 전체 feature에서 특정 갯수(hyper parameter)개의 변수를 선택
    * Information Gain이 가장 높은 변수 하나를 선택해서 노드를 생성
    * 위의 방법을 반복적으로 수행해서 트리를 형성(hyper parameter:tree의 level)
    * 전체 feature에서 선정된 특정갯수의 feature로만 트리를 생성함
- 위의 방법으로 여러개의 모델을 생성

### 6. 실습 예제
- hypter parameter
    - n_estimators : 트리의 갯수 : 많을수록 과적합을 피할수 있다 : 2000개 이상
    - max_depth : 트리의 레벨
    - max_features : 최대 변수의 갯수
    - stratify : y 데이터의 비율을 맞춤 (e.g. y=1 70%, y=0 30%, -> sampling y=1 70%, y=0 30%)
    - n_estimators= Tree의 갯수(default=100)

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
```
```python
# 샘플 데이터
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=5, random_state=2).fit(X_train, y_train)
```


