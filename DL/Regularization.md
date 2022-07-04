## Regularization
- Overfitting을 피하기위해, generalization error를 낮추기위한 방법
- training error가 높아질 수 있다.
	- training error 최소화를 방해하는 형
	- 최소화 term과 최대화 term 균형을 찾도록함
- Generalization이 잘 된 모델은 noise에 강인함

### 기법
- 데이터
	- Data Augmentation
		- Noise injection
- Loss 함수
	- Weight Decay
- Neural Network layer
	- Dropout
	- Batch Normalizaion
- 학습방식, 추론 방식
	- Early Stopping
	- Bagging & Ensemble

#### Weight Decay
- L2 Norm을 통해 weight parameter가 원점에서 멀어지는 것을 방지
	- Bias는 penalty 대상에서 제외
- Weight parameter는 노드와 노드간의 관계를 나타냄
	 - 숫자가 커질수록 강한 관계
- 전체적인 관계의 강도를 제한하여 출력 노드가 다수의 입력 노드로부터 많이 배우지 않도록 제한
> - Loss 함수 수정을 통한 regularization 방식의 대표
> 	- Original objective term과 반대로 최적화 되는 regularization term
> 	- 두 term을 동시에 최소를 만드는 과정에서 overfitting을 방지
>	- 두 term 사이의 균형을 유지하는 것이 관건: hyper-parameter를 통해 조절
#### Data Augmentation
- 데이터 늘리기
- 핵심 특징을 간직한 채, noise를 더하여 데이터를 확장
- 더욱 noise robust한 모델을 얻음
- 규칙 증강(augment)은 옳지 않음
	- 모델이 그 규칙을 배움
	- Randomness가 필요
- 예시
	- Salt & Pepper Noise
	- Rotation
	- Flipping
	- Shifting
	- Dropping
	- Exchange
- Generative Models
	- Autoencoder or GAN 통해 이미지 학습 후 생성
> - 한계
- 기존 데이터를 통해 새로운 지식을 배울 수 없다.
- 최적화 측면에선 유리
#### DropOut

