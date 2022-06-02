## K-means Clustering
- unsupervised learning
- 각 군집에 할당된 포인트들의 평균좌표를 이용해 중심점을 반복해서 업데이트 하는 방법
- 알고리즘 수행 절차
    1. k개의 중심 좌표 생성
    2. 각 데이터와 중심 좌표 사이의 거리를 계산하여 가까운 중심 좌표로 군집을 할당
    3. 할당된 군집을 기반으로 새로운 중심 좌표를 계산
    4. 모든 데이터의 클러스터 할당이 바뀌지 않을때까지 반복

#### sklearn 사용
- n_clusters: 군집의 갯수
- init: 초기화 방법. "random"이면 무작위, "k-means++"이면 K-평균++ 방법
- n_init: 초기 중심위치 시도 횟수. 디폴트는 10이고 10개의 무작위 중심위치 목록 중 가장 좋은 값을 선택
- max_iter: 최대 반복 횟수
- random_state: 시드값

#### k-means++
- 최초 랜덤 포인트를 효율적으로 뽑는 방법
- K-평균++ 방법 : 처음 랜덤으로 뽑힌 포인트에서 가장 먼 지점에 두번째 랜덤 포인트를 설정

##### 예시
```python
from sklearn.cluster import KMeans

# df_1(클러스터 1 데이터), df_2(클러스터 2 데이터) 데이터 결합
df_datas = pd.concat([df_1, df_2]).reset_index(drop=True)

# KMeans 모델 객체 생성 및 학습
model = KMeans(n_clusters=2, init="random", n_init=1, max_iter=1).fit(df_datas[["x", "y"]])

# 모델을 사용한 결과 예측
pred = model.predict(df_datas[["x", "y"]])

# 최종 클러스터 중심 좌표 출력
model.cluster_centers_
```

#### 최적의 k 값 찾기
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- WSS : inertia_
    - 가장 가까운 군집 중심까지의 표본의 제곱 거리의 합.

```python
# yellowbrick 패키지 사용 -> feature의 수가 많아질수록 k 값을 찾는것이 정확하지 않습니다.
from yellowbrick.cluster import KElbowVisualizer

visualizer = KElbowVisualizer(model, k=(1, 20))
```
- ARI
    - Adjusted Rand Index
    - 0 ~ 1사이의 값을 갖고 1이 가장 좋은 성능을 의미
- AMI
    - Adjusted Mutual Information : Adjusted Mutual Information between two clusterings
    - 확률변수간 상호의존성을 측정한 값
- ARI와 AMI는 종속변수가 있어야 구할수 있음
- Silhouette Score
    - cluster간 거리가 멀고, cluster 내 데이터간 거리는 가깝게

```python
# ARI, AMI값은 라벨 데이터가 있기때문에 확인이 가능 
# -> 높은 값을 선택하면 좋은 K값을 선택할수 있음
# ARI, AMI
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

# Silhouette Score: cluster 간 거리가 멀고, cluster 내 데이터간 거리는 가깝게
from sklearn.metrics.cluster import silhouette_score

ari_datas, ami_datas, silhouette_datas = [], [], []

n_datas = range(5, 15) 

for n in n_datas:

    model = KMeans(n_clusters=n, random_state=0)
    model.fit(digits.data)
    
    # 예측 데이터
    pred = model.predict(digits.data)
    
    # 평가 지표 데이터 저장
    # ARI
    ari_data = adjusted_rand_score(digits.target, pred) # 실제 데이터가 있어야 평가지표 구할수 있음
    # AMI
    ami_data = adjusted_mutual_info_score(digits.target, pred) # 실제 데이터가 있어야 평가지표 구할수 있음
    # Silhouette
    silhouette_data = silhouette_score(digits.data, pred) # 실제 데이터가 없어도 평가지표 구할수 있음

    # 데이터 저장
    ari_datas.append(ari_data)    
    ami_datas.append(ami_data)
    silhouette_datas.append(silhouette_data)
    
    # 데이터 출력
    # print("n : {},\t ARI: {},\tAMI : {},\tSilhouette Score: {}".format(n, ari_data, ami_data, silhouette_data))
    print("n : {},\t Silhouette Score: {}".format(n, silhouette_data))
'''    
# 그래프 출력

# plt.plot(n_datas, ari_datas, label="ARI")
# plt.plot(n_datas, ami_datas, label="AMI")
plt.plot(n_datas, silhouette_datas, label="Silhouette")

plt.xticks(n_datas)
plt.legend()

plt.show()
'''
```

