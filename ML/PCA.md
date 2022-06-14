## PCA (Principal Component Analysis)
- PCA란 입력데이터들의 공분산 행렬(convariance matrix)에 대한 고윳값 분해
- 기존 데이터의 <u>**분포를 최대한 보존**</u>하면서 고차원 공간의 데이터들을 저차원 공간으로 변환하는 차원축소기법
- 데이터의 분산을 최대한 보존하는 축(서로 직교하는 새 기저)을 찾아 고차원 공간의 데이터를 <u>**선형 연관성이 없는**</u> 저차원 공간으로 변환하는 기법
- 기존의 변수를 조합하여, 서로 연관성이 없는 새로운 변수, 즉 주성분(principal component)를 만들어낸다.
- 참조: https://tyami.github.io/machine%20learning/PCA/

### 사용예시

- 얼굴인식
- 정규화
- 노이즈를 없애는 도구
- 다중공선성이 존재할 때 상관도가 높은 변수를 축소
- 연관성 높은 변수를 제거하여 연산속도 및 결과치 개선
- 다양한 센서데이터를 주성분분석하여 시계열로 분포나 추세를 분석하고 고장징후탐지

### scikit-learn PCA 
 **sklearn.decomposition.PCA**

- components_: 주성분 축
- n_components_: 주성분의 수
- mean_: 각 성분의 평균
- explained_variance_ratio_: 각 성분의 분산 비율

```python
from sklearn.preprocessing import StnadarcScaler

scaled = StandardScaler().fit_transform(dataframe)

from sklearn.decomposition import PCA

pca = PCA(n_components=2) # PCA로 변환할 차원의 수를 의미하는 생성 파라미터
pca.fit(scaled)
pca_data = pca.transform(scaled)
```

* PCA란 데이터의 변동성을 가장 잘 설명할 수 있는 축을 찾아 데이터 차원을 축소하는 기법
* PCA Component별 원본 데이터의 변동성(분산)을 얼마나 반영하고 있는지 확인!

```python
# 전체 변동성에서 개별 PCA components별로 차이나는 변동성의 비율을 제공
print(pca.explained_variance_ratio)
```
