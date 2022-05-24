## Step 1. 데이터 확인
- 어떤 feature인가?
- 각 feature의 의미
- train set, test set 구분
- Categorical or Numerical 구분

#### titanic
```python
import seaborn as sns
import pandas as pd

titanic = sns.load_dataset('raw/titanic') # 내장 데이터셋 사용 방법
```
데이터 형태
- Categorical
	- Nominal : survived, sex, embarked
	- ordinal : pclass

- Numerical
	- Discret : sibsp, parch
	- Continous : age, fare
-> categorical and numerical 혼합되어 사용

비어있거나 (NaN) 틀린 값 확인
	|numerical|categorical|
	|---|---|
	|0 값으로 대체|"empty", "null" 등으로 대체|
	|중간값으로 대체|가장 많이 나오는 category로 대체|
	|작은 수의 경우, 해당 instance 삭제|많이 비어있는 해당 feature 사용X|
##### numerical 데이터
```python
titanic.describe()
```
##### categorical 데이터
```python
titanic.describe(include=['O'])
```

## Step 2. 가설 검증
> e.g.
- [가설 1] 구명정에는 여자와 아이가 우선으로 탑승했다.
- [가설 2] 1등석의 승객들은 탈출에 용이했다.
```python
# 시각화
sns.countplot(x='Survived', hue='Sex', data=titanic)

grid = sns.FacetGrid(titanic,col='Survived',row='Pclass')
grid.map(sns.distplot,'Age',kde=False);
```

## Step 3. 결측치 처리 (Missing data)
결측치 처리방법
- 1. 무시한다.
- 2. 제거한다. (drop)
- 3. 다른 값으로 대체한다, (fill)

##### 통계적으로 차이가 나지 않도록 평균(mean)에 분산(std) 만큼 차이 나는 사이의 임의값을 넣어주기
```python
import numpy as np

std = np.std(list)
mean = np.mean(list)

rand_value = np.random.uniform(mean - std, mean + std, 4)
```
##### titanic['Age']
```python
mean = titanic['Age'].mean()       # age 의 mean 값을 구함
std = titanic['Age'].std()         # age 의 std 값을 구함
size = titanic['Age'].isna().sum() # age 에 nan 값이 몇개인지를 구함 


rand_age = np.random.randint(mean - std, mean + std, size = size) # mean 에서 std 만큼 떨어져 있는 랜덤한 값들을 size 만큼 반환
```

## Step 4. Feature Engineering
>  e.g.
- `Braund Mr. Owen Harris`에서 `Mr만 추출
- `.split('기준')
```python
# 정규표현식 사용하기
for data in titanic:
  titanic['Title'] = titanic['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) 
```
```python
# cut() 이용하여 구간설정
titanic['age_band'] = pd.cut(titanic['Age'], 5)    #5개의 구간으로 잘라, ageband 라는 새로운 열 생성
```

## Step 5. 데이터 검수
```python
titanic.info()

# int로 변환
titanic['Fare'] = titanic['Fare'].astype(int)     # float에서 int로 바꿔줍니다
```
##### `map`또는 `labelencoder`를 이용해 title을 numerical 하게 바꾼후, 기존 drop
 - ① 먼저 title 안에 어떤 값이 있는지 확인 - `데이터셋.title.unique()`
 - ② 중복되는 표현을 통일 - `데이터열.replace('이전', '이후')`
 - ③ numerical 데이터로 인코딩
 - ④ name 항목 드랍

```python
# LabelEncoder
from sklearn.preprocessing import LabelEncoder    # LabelEncoder를 불러오기
encoder = LabelEncoder()      # encoder라는 변수를 선언
encoder.fit(titanic['Title']) # title 열에 맞게 인코딩
titanic['Title'] = encoder.transform(titanic['Title']) # 인코딩 결과를 실제 행에 적용
```
## Step 6. Scikit-learn 모델로 성능 검증하기
`scores(x,y)` 함수는 x 에 features 를, y 에 예측하고자 하는 output 을 넣어주면
4개의 모델 

**logistic regression, SVM, KNN, Random Forest** 로 훈련한 정확도를 보여줍니다. 
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# max_iter = model.fit에서 피팅 시에 최대 시도 횟수 (기본 100)
model = LogisticRegression(max_iter=1000)   
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model.fit(x_train,y_train)
Y_pred = model.predict(x_test)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```
