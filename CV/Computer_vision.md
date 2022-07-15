# Computer Vision
## Linear Algebra
Mathmatics in vector space  
Image Understanding = Imgae description(Vector) + Decision Making(Function)
- Vector   
- Linear Dependency
- Basis
- Matrix
  - Square matrix
  - Identity matrix
  - Matrix operations [addition, subtraction, Multiplication, Transpose]
  - Rank of matrix
    - Linearly independent
  - Inverse Matrix
    - Square and non-singular
- Eigen Decomposition
  - computing inverse matrix
  - solving optimization problems
- ### [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

## Probability
noisy and uncertain  
- Random variable
- Probability Axioms
  - Sample sapce : the set of possible outcomes
  - Event space : A power set whose elemnet is a set of sample space
  - Probability measure (P):
    - A function P : E -> R with the following conditions
- Conditional Probabilty (조건부 확률)
- Bayes' Theorem
  - $posterior \propto linklihood \times prior$
- Gaussian Distributions(정규분포)

## OpenCV
- cv 오픈 소스 라이브러리
- OpenCV-python 을 통해 python  포팅도 되어있다.
  - GPU operation 지원을, 명시적으로 python과 연계하여 하지 않는다 
```python
import cv2

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # bgr -> rgb
```
- image cropping
- image masking

## PIL
- Python Imaging Library
- CV 보다는 이미지 처리에 중점을 둔 라이브러리
- 픽셀 단위 이미지 조작, 마스킹, 투명도 제어, 윤곽보정 및 검출, 등
  - Imgage read and visualize
  - Image cropping, rotating, scaling
  - Image interpolation (upsampling, downsampling)
```python
from PIL import Image
```
이미지 처리 그 자체만으로는 PIL라이브러리가 훨씬 간단하고 편리하다.  
하지만, 복잡한 컴퓨터비전 알고리즘의 적용이나, 딥러닝/머신러닝 모덺을 개발할 떄에는 PIL 의 함수만으로는 수행할 수 없다.  
- ### Scikit-image
  - numpy를 기반으로 동작하기 때문에, 좀 더 numpy와의 호환성이 좋다.