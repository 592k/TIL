# Object Detection
How to obtain the classification function $f(x)$?
- Detection = Localization + Classification  
## Classical Object Detection
### 1. Support Vector Machine
- Supervised leaerning
- Finding the hyperplane that separates two classes with the largest margin
- Linear SVM
   - Separable training examples
   - Margin
   - Max-margin solution
   - Identical, but easier-to-solve problem
   - Quadratic Programming (QP)
   - Soft margin
   - kernel method
   - Non-Linear SVM
     - Polynomial kernel, Gaussian kernel, Histogram intersection kernel, ...
   - Hyper-Paramters Tuning
     - Hyper-parametesrs
     - Kernel parameters
   - Multi-Class SVM
     - One-versus-all
     - One-versus-one
### 2. Histogram of Oriented Gradients (HOG)
- Gradient imgae computation
- Use color information when available
- Weighted vote into quantized orientations
- Soft voting
- Cell
  - A unit area to compute orientation histogram
- Block
   - Partially overlapped
- Window
  - Concatenating all block descriptors
* ### HOG + SVM
  * Classification by linear SVM with soft margin

## Pedestrian Detection