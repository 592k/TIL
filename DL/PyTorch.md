## PyTorch
### PyTorch Introduction
- Numpy와 매우 비슷한 문법
- 동적으로 back-propagation 경로 생성 가능
- 너무나도 훌륭한 documentation
- 난이도에 비해 자유도 높음
- 단점 :
	- TF에 비해 낮은 상용화 지원
### Dataset
- $D = {(x_i,y_i)}_{i=1}^N$
- x와 y는 n차원과 m차원의 벡터로 표현될 수 있다.
- Example
	- Tubular Dataset
		- 한 row가 한개의 sample
		- column 갯수가 벡터의 차원을 의미
### Tensor
- 0: scalar, 1: vector, 2: matrix
- 3차원 이상
- Mini-batch: Consider Parallel Operations
- e.g.
		- CV
		- |x| = (images, height, weight)
		- |x| = (#img, #ch, h, w)

