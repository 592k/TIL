# ML training
## ML FLOW (Scikit-Learn)
머신러닝 구현에 특화된 라이브러리 -> 딥러닝 X
```python
# sklearn을 사용하여 하나의 ML model을 불러와서 분류 모델을 학습하고 평가하는 예시

# 1. 사용할 모델을 불러옵니다.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. 모델 객체를 선언합니다.
model = RandomForestClassifier()

# 3. training data로 학습을 진행합니다.
model.fit(X_train, y_train)

# 4. test data로 inferenxce를 진행합니다.
pred = model.predict(X_test)

# 5. Evaluation metric으로 평가를 진행합니다.
print("Accuracy : %.4f" % accuracy_score(y_test, pred))

>> Accuracy : 0.8976
```

### Classification (titanic)
- Survived 예측

결측치 처리
```python
# Embarked, Age
# 1. Embarked
# train.Embarked.value_counts() -> S 가장 많음
# 같은 Ticket, Fare, Cabin
# train.loc[(train.Pclass ==1) & (train.Sex=='female'), 'Embarked'].value_counts()
train.loc[train.Embarked.isnull(), 'Embarked'] = 'S'

# 2. Age
# Age fill -> mean() ### EDA
train.Age = train.Age.fillna(train.Age.mean())
```
```python
# train.Cabin.value_counts() -> 147, 좌석의 위치에 따른 생존 여부확인이 유의미 하지 않음 => drop

# PassengerId, Name, Ticket, Cabin, => drop
train = train.drop(columns=['PassengerId','Name','Ticket','Cabin'])
```

#### Feature Engineering
1. Cagegorical feature encoding
2. Normalizaion
```python
# categorical feature --> One-hot Encoding, Ordinal Encoding

# 1. Ordinal Encoding -> Ordinal feature 변환, e.g. 학력, 선호도 ..
# 2. One-hot Encoding -> Nominal feature 변환 e.g. 성별, 부서, 출신학교.. (구분)

train_OHE = pd.get_dummies(train, columns=['Sex','Embarked']) # 1 or 0 -> 정보차이 X
```
```python
# Normalization --> Min-Max scaling
X = train_OHE.drop(columns='Survived')
y = train_OHE.Survived
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# scaler.fit()
# scaler.transform()
# X_scaled = scaler.fit_transform(X)

# 특정 X 만 scale
temp = scaler.fit_transform(X.loc[:,['Age', 'Fare']]) # -> dataframe만 가능
X['Age'] = temp[:,0]
X['Fare'] = temp[:,1]
```

#### Training
```python
# Classification
from sklearn.linear_model import SGDClassifier # 1. Linear Classifier
from sklearn.linear_model import LogisticRegression # 2. Logistic Regression
from sklearn.tree import DecisionTreeClassifier # 3. Decision Tree
from sklearn.ensemble import RandomForestClassifier # 4. Randcom Forest

# 평가 지표
from sklearn.metrics import accuracy_score
```
```python
clf = SGDClassifier()
clf2 = LogisticRegression()
clf3 = DecisionTreeClassifier()
clf4 = RandomForestClassifier()

clf.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)

pred = clf.predict(X)
pred2 = clf2.predict(X)
pred3 = clf3.predict(X)
pred4 = clf4.predict(X)

print('1. Linear Classifier, Accureacy for training : %.4f' % accuracy_score(y, pred))
print('2. Logistic Regression, Accureacy for training : %.4f' % accuracy_score(y, pred2))
print('3. Decison Tree, Accureacy for training : %.4f' % accuracy_score(y, pred3))
print('4. Random Forest, Accureacy for training : %.4f' % accuracy_score(y, pred4))
```

#### Test (Predict)
```python
# test data에 같은 feature engineering을 적용해줍니다.
test = test.drop(columns=['PassengerId', 'Name','Ticket','Cabin'])

# fillna -> train data의 mean으로 채워줘야 한다.
test.Age = test.Age.fillna(train.Age.mean())
test.Fare = test.Fare.fillna(train.Fare.mean())

# Categorical feature encoding
test_OHE = pd.get_dummies(data=test, columns=['Sex','Embarked'])

# Nomalization
temp = scaler.transform(test_OHE.loc[:, ['Age','Fare']]) # -> train의 scaler사용, test data로 fit 하지 않음
test_OHE.Age = temp[:,0]
test_OHE.Fare = temp[:,1]
```
```python
# prediction
result = clf.predict(test_OHE)
result2 = clf2.predict(test_OHE)
result3 = clf3.predict(test_OHE)
result4 = clf4.predict(test_OHE)
```
#### Kaggle 제출을 위한 submission 파일 생성
```python
# prediction
result = clf.predict(test_OHE)
result2 = clf2.predict(test_OHE)
result3 = clf3.predict(test_OHE)
result4 = clf4.predict(test_OHE)

# 결과 파일인 submission.csv를 생성합니다.
submission['Survived'] = result4

submission.to_csv('submission.csv', index=False)
```

### Regression (california_housing)

sklearn.datasets으로 데이터 불러오기
- 결측치 없는 데이터 (data.info() 확인)
- 모든 columns 사용
```python
from sklearn.datasets import fetch_california_housing

X = fetch_california_housing(as_frame=True)['data']
y = fetch_california_housing(as_frame=True)['target']
data = pd.concat([X, y], axis=1)
```
#### EDA
- outlier 확인
- 데이터 분포 확인 
```python
# feature distribution
#sns.histplot(data=data, x="AveRooms")
#data.AveOccup.value_counts()
#plt.figure(figsize=(10, 6))
#sns.boxplot(data=data.loc[:, ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "AveOccup", "Latitude", "Longitude", "MedHouseVal"]])
#sns.boxplot(data=data.loc[:, [ "AveRooms", "AveOccup"]])
#plt.figure(figsize=(8, 6))
#sns.heatmap(data.corr(), annot=True)

#data.loc[data.AveRooms > 100, :] # 1914, 1979 row 제거
#data.loc[data.AveOccup > 200, :] # 3364, 13034, 16669, 19006 row 제거
# AveBedrms, Longitude column 제거
data = data.drop(index=[1914, 1979, 3364, 13034, 16669, 19006]) ## remove outlier
data = data.drop(columns=["AveBedrms", "Longitude"]) ## remove collinearity
```
#### Training
- train-test split
- Standardization
- Model training
- Hyper-parameter tuning
- Evaluation
```python
# 학습을 위한 training / validation / test dataset 나누기
from sklearn.model_selection import train_test_split

# 트테트테
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0xC0FFEE)

# 6 : 2 : 2 = train : validation : test
```
```python
# feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```
```python
# 학습을 위한 라이브러리 세팅
from sklearn.linear_model import LinearRegression   # 1. Linear Regression
from sklearn.linear_model import Lasso              # 2. Lasso
from sklearn.linear_model import Ridge              # 3. Ridge
from xgboost.sklearn import XGBRegressor            # 4. XGBoost
from lightgbm.sklearn import LGBMRegressor          # 5. LightGBM

from sklearn.metrics import mean_squared_error
```
```python
## training
reg = LinearRegression() # 모델생성
reg2 = Lasso()
reg3 = Ridge()
reg4 = XGBRegressor()
reg5 = LGBMRegressor()

reg.fit(X_train, y_train) # fit
reg2.fit(X_train, y_train)
reg3.fit(X_train, y_train)
reg4.fit(X_train, y_train)
reg5.fit(X_train, y_train)

pred_train = reg.predict(X_train) # train_predict
pred_train2 = reg2.predict(X_train)
pred_train3 = reg3.predict(X_train)
pred_train4 = reg4.predict(X_train)
pred_train5 = reg5.predict(X_train)

pred_val = reg.predict(X_val) # val_predict
pred_val2 = reg2.predict(X_val)
pred_val3 = reg3.predict(X_val)
pred_val4 = reg4.predict(X_val)
pred_val5 = reg5.predict(X_val)

mse_train = mean_squared_error(y_train, pred_train) # 모델 평가
mse_val = mean_squared_error(y_val, pred_val)
mse_train2 = mean_squared_error(y_train, pred_train2)
mse_val2 = mean_squared_error(y_val, pred_val2)
mse_train3 = mean_squared_error(y_train, pred_train3)
mse_val3 = mean_squared_error(y_val, pred_val3)
mse_train4 = mean_squared_error(y_train, pred_train4)
mse_val4 = mean_squared_error(y_val, pred_val4)
mse_train5 = mean_squared_error(y_train, pred_train5)
mse_val5 = mean_squared_error(y_val, pred_val5)


print("1. Linear Regression\t, train=%.4f, val=%.4f" % (mse_train, mse_val))
print("2. Lasso\t\t, train=%.4f, val=%.4f" % (mse_train2, mse_val2))
print("3. Ridge\t\t, train=%.4f, val=%.4f" % (mse_train3, mse_val3))
print("4. XGBoost\t\t, train=%.4f, val=%.4f" % (mse_train4, mse_val4))
print("5. LightGBM\t\t, train=%.4f, val=%.4f" % (mse_train5, mse_val5))

### -> 함수를 만들어서 사용
```
```python
# prediction!
result = reg.predict(X_test)
result2 = reg2.predict(X_test)
result3 = reg3.predict(X_test)
result4 = reg4.predict(X_test)
result5 = reg5.predict(X_test)

# Summary!
print("---------- Linear Regression ---------")
print('MSE in training: %.4f' % mean_squared_error(y_test, result))

print("---------- Lasso ---------")
print('MSE in training: %.4f' % mean_squared_error(y_test, result2))

print("---------- Ridge ---------")
print('MSE in training: %.4f' % mean_squared_error(y_test, result3))

print("---------- XGBoost ---------")
print('MSE in training: %.4f' % mean_squared_error(y_test, result4))

print("---------- LightGBM ---------")
print('MSE in training: %.4f' % mean_squared_error(y_test, result5))
```
#### Hyper-parameter tuning
1. Human Search 직접
2. Grid Search(GridSearchCV) : 주어진 hp의 조합을 모두 돌려보는 방식.
3. Bayesian Optimization(hyperopt, optuna, ...) : hyper-parameter를 최적화하는 베이지안 방식을 사용.
```python
# GridSearchCV
from sklearn.model_selection import GridSearchCV

# reg5 = LGBMRegressor(<hyper-parameter>) -> max_depth, learning_rate, n_estimators
# hyper-parameter default
'''
max_depth = -1 (무제함)
learning_rate = 0.1
n_estimators = 100
'''

param_grid = {
    "max_depth" : [3, 4, 5, 6, 7, -1], # 6. ## 다른 code 참조
    "learning_rate" : [0.1, 0.01, 0.05], # 3
    "n_estimators" : [50, 100, 200] # 3
} # 6x3x3 = 54

# GridSerchCV(<model>, parameter,scoring=<mse function>)
gcv = GridSearchCV(reg5, param_grid, scoring='neg_mean_squared_error', verbose=1)

gcv.fit(X_train, y_train)

print(gcv.cv_results_) # 전체 확인

print(gcv.best_estimator_)
print(gcv.best_params_)

>> LGBMRegressor(n_estimators=200)
>> {'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 200}

final_model = gcv.best_estimator_
```

### Clustring (instacart)
- unsupervised machine learning
- user-products 관계

index = user_id, columns = product_name, values = counts
```python
X = pd.crosstab(index=temp.user_id, columns=temp.department)

# tSNE : 2차원으로 변환하여, 시각화에 용이
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
tsne_data = tsne.fit_transform(X)

plt.scatter(tsne_data[:, 0], tsne_data[:, 1], s=10, c='y')
```
```python
# K-means
from sklearn.cluster import KMeans                    # 1. K-means
from sklearn.cluster import AgglomerativeClustering   # 2. Hierarchical Agglomerative Clustering
from sklearn.cluster import DBSCAN                    # 3. DBSCAN
from sklearn.cluster import SpectralClustering        # 4. Spectral Clustering

from sklearn.metrics import silhouette_score

model = KMeans(n_clusters=4) # centroid 4개

# 예측
pred = model.fit_predict(X) # unsupervised -> y가 없음
print("Silhouette Score : %.4f" % silhouette_score(X, pred))
```
```python
# 최적의 K
# # Elbow method : 정해진 K에 대해서 SSE를 계산한 다음, SSE가 가장 많이 꺾이는 K(elbow)가 optimal K라고 판단
sse = []
silhouettes = []

for K in range(2, 11):
  model = KMeans(n_clusters=K) # K : 2 ~ 10
  pred = model.fit_predict(X)
  sse.append(model.inertia_) # SSE
  silhouettes.append(silhouette_score(X, pred))
    
plt.figure(figsize=(8, 6))
plt.title("Find optimal K with elbow method", fontsize=14)
plt.xlabel("Number of Clusters(K)", fontsize=10)
plt.ylabel("SSE", fontsize=10)
plt.plot(range(2, 11), sse, lw=3)
plt.show()

# Silhouette score : 같은 클러스터에 속하는 데이터중 가장 먼 데이터와의 거리와 다른 클러스터에 속하는 데이터 중 가장 가까운 데이터와의 거리 비율을 계산한 지표. [-1, 1]
# 가장 큰 silhouette scoure 선택
plt.figure(figsize=(8, 6))
plt.title("Find optimal K with silhouette score", fontsize=14)
plt.xlabel("Number of Clusters(K)", fontsize=10)
plt.ylabel("Silhouette Score", fontsize=10)
plt.plot(range(2, 11), silhouettes)
plt.show()
```
```python
# summary
# clustering 확인, 해석이 필요
X["Cluster_label"] = pred

group1 = X.loc[X.Cluster_label == 0, :]
group2 = X.loc[X.Cluster_label == 1, :]
group3 = X.loc[X.Cluster_label == 2, :]
group4 = X.loc[X.Cluster_label == 3, :]

group1.mean()

# 각 그룹의 평균을 확인
# user - product -> 각 그룹 별 평균적으로 가장 많이 구매한 제품
```

