## Pre-processing
- Resize
- Color
  - RGB, Gray
- Normalization
  - scaling, mainmax, average, ...
  
## OpenCV 예시
```python
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR
img.shape
img = cv2.resize(img, (200,200))
img = img[:100,:100]
...
```

## Class imbalance
양, 다양성 ⬆️  
1. Oversampling
   - 단순한 중복 추출, 데이터의 양⬆️, 다양성 x
2. Data augmentation
   - transform, 다양성 ⬆️ -> epoch 수 늘려야 한다.
   - 다른 class에도 적용 해야함, 데이터의 비율은 그대로..
3. Focal loss
   - Pt : 등장확률 
   - 적은 비중을 가진 class에 가중치를 더 준다.

## Overfitting
generalization ⬆️ 가 목표  
- overfitting 방지
  - 일반화(Regularization) 항 추가 (L1, L2)
    - $COST = \frac{1}{n} \displaystyle\sum_{i=1}^{n}{loss}$
    -  $COST = \frac{1}{n} \displaystyle\sum_{i=1}^{n}({loss + \frac{\lambda}{2}w^2})$
    -  weight decay
 - 앙상블(Ensemble)
   - model1 + model2 + model3 + ...(voting)
   - 외부 데이터가 제한되는 대회에서 주로 사용
 - Dropout

## Image augmentation
데이터 셋의 다양성 증가
- original &rarr;
  - flip, blur, HueSturationValue, GaussNoise, CoarseDropout, Gray, ...
- YOLOv4
  - MixUp, CutMix, Mosaic, Blur, RandomRain
