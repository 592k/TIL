# Representation learning
- Image representation + decision making
- Convolutional layers + Pooling operatoins + MLP (Fully-connected layers)
  - Pre-training the feature extractor (backbone)
  - Post-training (fine-tune) the classifier (or other task-specific branches)
> Metric
  - A function that quantifies a distance between every pair of elements in a set, thus inducing a measure of similarity
## Similarity in Computer Vision
- Information retrieval
- Face identification / Verification
- Person re-identification
- Visual object tracking
- Local patch matching for stereo imaging*
- Visual representation learnging
  - Unsupervised Patch Tracking
> Metric Learning
>   - Learning $D$ and/or $f$ satisfying the relations from data

## Classical Metric Learning
- 대표적 : Eulidean distance
- But, when considering the data manifold
  - Mahalanobis distance $D_M$
    - $D_M(x_i,x_j) = \sqrt{(x_i - x_j)^TM(x_i - x_j)}$
    - Estimating $M$ of $D_M$ from data
    - Estimating covariance matrix of data $\sum$ usinf its inverse as $M = \sum^-1$

## Deep Metric Learning
Learning representation from data
- Siamese Network
  - Siamese architecture
  - Contrastive loss
  - Distance measure D : Euclidean distance
  - Minimize of L
- Treiplet Network
  - Relation among a triplet
  - $L_2$ normalization of embedding feature
  - Weight sharing

## Sampling Matters
- Distance Weighted Sampling (DWS)
  - offers a wide range of training samples
  - Margin-based loss
- Quadruplet Network
  - **Multi-object tracking**
  - A typical form: A linear combination of two triplet rank losses

## Applications of Deep Metric Learning
- Finding nearest ones among training examples in a specific space
- Label transfer
- Image retrieval
  - Content-based image retrieval
    - A most straightforward way
    - Working even for unseen calsses or concepts
    - Margin-based
- Face Verification
- Person Re-identification
  - Identifying people across differnet cameras
- Online Visual Tracking
  - Particle filtering
    - Comparing the inital target to candidate boxes
- Unsupervised Representation Learning
  - Context prediction
  - Inpainting 
  - Unsupervised Patch Tracking