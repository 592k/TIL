## Deep leanring
Deep Nuearl Networks  
- 인공신경망(Aritificial Neuarl Networks)을 이어받음
- 비선형 함수로, 패턴 인식 능력이 월등
- 활용사례
	- 음성인식
	- 기계번역
	- 자율주
	- 객체 인식, 이미지 분류
	- 사진 합성, 사진 보정, Super Resolution
	- 데이터 분석

> 인공지능 모델
- x가 주어졌을 때, y를 반활하는 함수
- 파라미터(weight parameter)
	- 함수가 동작하는 방식을 결정
> 좋은 인공지능
- 일반화(Generalization)를 잘하는 모델
- Unseen 데이터에 대해서 좋은 prediction을 하는 Model
### Working Process
1. 문제 정의
	- 문제를 단계별로 나누고, simplify 하여야 한다.
	- x와 y가 정의 되어야 한다.
2. 데이터 수집
	- 문제의 영역에 따라 수직 방법이 다름
		* NLP,CV : crawling
		* 데이터 분석 : 실제 수집한 데이터
	- 필요에 따라 Labeling
		* supervised learning
3. 데이터 전처리 및 분석
	- 신경망에 넣어주기 위한 형태로 가공하는 과정
		* 입출력 값을 정제(cleaning & normalization)
	- EDA
	- CV의 경우, 데이터증강(augmentation)
		* e.g. rotation, flipping, shifting, ..
4. 알고리즘 적용
	- 가설설정, 가설을 위한 모델 적용
5. 평가
	- Test set 구성
		* 실제 데이터와 가깝게
	- 정량적 평가와 정성적 평가로 나뉨
		* Extrinsic evaluation or Intrinsic evaluation
6. 배포
	- 학습과 평가가 완료됨 모델 weights파일 배포
	- RESTful API 등을 통해 wrapping 후 배포
	- 데이터 분포 변화에 따른 모델 업데이트 및 유지/보수가 필요할 수 있음

### Appendix: Basic Math
1. 지수와 로그
2. Summation & Product
3. Argmax
	- Pick the argument that makes max value.
	- argmin 반대

