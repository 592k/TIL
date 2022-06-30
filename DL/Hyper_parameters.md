## Hyper-parameters
- 모델 외부의 설정 값으로, <u>사용자에 의해서 결정</u>됨
- 모델의 성능을 좌우
- 최적의 값은 다 다름, Heuristic한 방법으로 search
> - Network Depth & Width
	- Network의 capacity를 결정
		- 너무 깊으면 overfitting
		- 너무 얕으면 관계 또는 함수를 알 수 없음
> - Laerning Rate
> - ReLU, LeakyReLU
	- 각도
- 등등 ...
- **실험결과를 잘 정리해서** 설정해야한다.
#### 효율적인 실험방법
- baseline 구축!
- Hyper-parameter 별 결과물 관리
	- 성능 및 모델 파일
	- 모델 파일 이름에 저장
		- model.n_layers-10.n_epochs-100.act-leaky_relu.loss-xxx.accuracy-xx.pth
		- 하지만 Table 정리 필요
	- 실험관리 프레임워크
		- MLFlow
		- WanDB
