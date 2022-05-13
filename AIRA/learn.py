# Proximity

'''
np.square() 제곱
np.sqrt() 루트

a @ b = a 와 b 내적
'''

def euclidean_distance(a,b):
	x = np.sqrt(np.sum(np.square(a-b)))
	return x
f = lamda x,y : np.sqrt(np.sum(np.square(s-y)))

def cosine_similarity(a,b):
	x = (a@b)/(np.linalg.nor(a)*np.linalg.norm(b) + 1e-12)
	return x

'''
distances = []
for idx, row in data.iterrows():
	distances.append(euclidean_distance(target, row))
+
target = num
-> distances 리스트의 target data 변환
np.argmin(distances)
np.argmax(distances)
'''
# 공분산 계산 방법
# np.cov()
# 단위에 의존하지 않고 상관관계를 이해해야함 -> 상관계수

