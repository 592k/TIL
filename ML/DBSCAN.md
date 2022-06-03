## DBSCAN
- k-means 클러스터링의 특징
    - 단순하고 강력한 클러스터링 방법 (DBSCAN = sensitive)
    - 군집의 모양이 원형이 아닌경우에 잘 동작하지 않음
    - k 값을 사용자가 지정
- DBSCAN 클러스터링의 특징
    - Density-based Spatial Clustering of Applications with Noise
    - 데이터의 밀집 정도를 사용
    - 데이터의 군집형태와 관계없이 클러스터링이 가능
    - 군집의 갯수를 사용자가 지정할 필요가 없음
    - Density-based clustering 중 가장 성능이 우수
    - <u>DBSCAN (k 초기 값 찾기) -> K-means</u>
    - eps-neighbors와 MinPts를 사용하여 군집구성

- DBSCAN 알고리즘
    * eps-neighbors : 한 데이터를 중심으로 **epsilon($\epsilon$)거리 이내의 데이터**들을 한 군집으로 구성
    * MinPts : 한 군집은 Minpts보다 많거나 같은 수의 데이터로 구성됨
    * 데이터의 분류
        - Core : 군집을 찾는데 기준이 되는 데이터
        - Border : 군집에 포함은 되지만 군집을 찾는데 기준이 되지 못하는 데이터
        - Noise : 아웃라이어 데이터
- DBSCAN hyper parameter 설정 및 특징
    * **MinPts 설정**
        - eps값 내에 있는 데이터의 수
        - eps값 내에 있는 데이터의 수가 MinPts개 이상이면 Core 데이터가 됨
        - 간단한 설정 : 변수의 수 + 1로 설정
        - 3이상으로 설정 -> 15이상
    * **eps 설정**
        - 너무 작으면 많은 데이터가 노이즈로 구분됨
        - 너무 크면 군집이 하나가 될수 있음
        - K-nearest neighbor의 거리를 그래프로 나타낸후 거리가 급격하게 증가하는 지점을 eps 값으로 설정
        - Knn, 데이터의 k의 설정 값과 거리 (Euclidean Distance)에 따라 분류(k= 홀수)
```python
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=<eps>).fit(<data>)
```
#### 최적의 eps 값 설정
- K-nearest neighbor의 거리를 그래프로 나타낸후 거리가 급격하게 증가하는 지점을 eps 값으로 설정
- 모든 데이터의 케이스에 적용되는것은 아님
```python
# K-nearest neighbor를 이용하여 최적화된 eps값 구하기
from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(<data>)
distances, indices = nbrs.kneighbors(<data>)

# 기울기 값 구하기
distances_g = np.gradient(distances[:, 1])
distances_g[:5]

# 기울기가 최대가 되는 위치값 구하기
max_index = np.argmax(np.gradient(distances_g))

# 기울기가 최대가 되는 위치 값에있는 eps 값 구하기
opt_eps = distances[max_index][1]

# 최적화된 eps 값으로 모델 학습
# 최적의 eps 값으로 모델학습
model = DBSCAN(eps=opt_eps).fit(<data>)
```
* MinPts
    - 각 cluster 마다 평균적인 데이터 수를 확인 -> 최적의 MinPts 찾기
    - eps-neighbors가 더 중요
####  DBSCAN의 장단점
- 장점
    - 군집의 수(K)를 설정할 필요가 없음
    - 다양한 모양의 군집이 형성될 수 있으며 군집끼리 겹치는경우가 없음
    - 하이퍼 파라미터가 2개로 작은편
- 단점
    - 데이터는 하나의 군집에 속하게 되므로 시작점에 따라 다른 모양의 군집이 형성됨
    - eps 값에 의해 성능이 크게 좌우됨 : eps 값의 테스트를 많이 해보아야 함
    - 군집별 밀도가 다르면 군집화가 잘 이루어지지 않음

### HDBSCAN 
- Hierarchical DBSCAN
- 하이퍼 파라미터에 덜 민감한 DBSCAN 알고리즘
- https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
    - min_cluster_size를 기준으로 클러스터를 생성
    - 생성된 클러스터의 core 데이터와 다른 클러스터의 core 데이터의 거리에 대한 계층을 만듦
```python
import hdbscan

model = hdbscan.HDBSCAN(min_cluster_size=25).fit(datas)
pred = model.fit_predict(datas)

df = pd.DataFrame(datas)
df["labels"] = pred

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)

# 클러스터를 몇개로 하면 좋을지 알려줌
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
plt.show()
```
* 추가)
    - Hierarchical Clustering => 밀도 기반
    - Gaussian Mixture => 각 클러스터가 정규분포를 따르도록 클러스터링한다.
