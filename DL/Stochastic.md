## Stochastic Gradient Descent (SGD)
- < 1 parameter update by **GD** from full sample's loss>
	- Too Expensive
- 1st update from random <u>k sample loss</u>
- 2nd update from anothor random <u>k sample loss</u>
- random 일부 데이터를 sampling해서 update

### Epoch & Iteration
- 1 Epoch
	- 모든 데이터셋의 샘플들이 forward & backward 되는 시점
	- Epoch의 시작에 데이터셋을 random shuffling 해준 후, 미니배치로 나눈다.
- 1 Iteration
	- 한 개의 미니배치 샘플들이 forward & backward 되는 시점
- 따라서 Epoch와 Iteration의 <u>이중의 for loop</u>이 만들어지게 됨
	- 파라미터 전체 업데이트 횟수: #epochs x #iterations
### Summary
- 전체가 아니라 일부 샘플의 loss에 대한 gradient descent
- 1 epoch당 파라미터 update 횟수 증가
- batch_size가 작아질 수록 실제 gradient와 달라질 것
	- 큰 batch_size가 학계 추세
		- 큰 배치사이즈로 인한 비용이 줄어듬
		- 매우 크면 성능은 악화

