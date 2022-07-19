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

## Finding object bounding box
- Box localization (Object proposals) + Box classification (CNN)
- Motivation
  - Capture all acales
  - Diversification
  - Fast to compute
### EdgeBox for Object Detection
- Gradient orentation 을 기반으로 edge group을 표현하고, 이를 이용해서 bounding box score를 계산함
- Evaluation metrix 제안 : Intersection on Union (IoU)
### Region-based CNN (RCNN)
- each proposal using any architecture
- Bounding box regression
- MEan average precision(mAP): 53.7%
- Fast R-CNN
  - A single feature computation and ROI pooling
  - Bounding boc regression
  - multi-task loss
  - RoI pooling
    - conceptual illustration
    - cell-wise pooling for feature aggregation
### Faster R-CNN
- Fast RCNN + Regio Proposal Network
- 9 anchors per location
- Groundtruth label per anhor
- Trained with <u>a binary classification loss</u> for anchor selection and <u>a regression loss for box refinement</u>
### Mask R-CNN
- framwork for instance segmentation
- RoI Align & Head Architecture
## One-Stage Detector: YOLO
You Only Look Once  
- Loss functions
  - Classification loss
  - Localization loss
  - Confidence loss
- Non-maximal suppressions
  - 중복되는 박스 제거
### YOLO v2
- Better
  - Accuracy, mAP 개선
- Faster
  - DarkNet
  - classification task pre-train
  - detection task 로 학습
  - 
- Stronger
  - Hierarchical classification
  - Dataset combination with WordTree
  - Joint  classification and detection
- Real-time object detection system
### YOLO v3
- Bounding box prediction
  - 하나의 object에 하나의 anchor box 할당
- class prediction
- Prediction across across scales
- Feature extractor
  - DarkNet-53
### YOLO v4
- Practically, one-stage detector의 개선
### **Evaluation Metrics of Object Detection**
- IoU
- Recall : 정답 박스를 맞춘 수/ 전체 정답 박스 수
- Precision : 예측한 박스 중에 정답을 맞춘 수/ 예측한 박스 수
- AP (Average Precision)
- mAP

## Deconvolution Network (DeconvNet)
- Overall architecture: Convolutional encoder-decoder
- Unpooling
- Deconvolution
### U-Net
- U-Net for semantation (encoder-decoder) : for medical image processing
- 