### Overfitting
- Drop out
- Weight decay
- Early stopping
### Network Weight Initialization
- Learning from scratch
  - Small (and adequate) random numbers
  - Problem for deep learning
- Finetunning an existing network
### Optimization Techniques of NN
- Stochastic Gradient Descent (SGD)
- AdaGrad
- Adam
- ...
- Learning rates are adaptively determined for each parameter

# Models
## AlexNet
- Similar with **LeNet-5**
- 7 hidden layers, 650K neurons, 60m parameters
- larger amount of data
- ReLu and droup out
- Conv-Pool-LRN-Conv-Pool-LRN-Conv-Conv-Conv-Pool-FC-FC-FC
  - LRN : Local Response Normalization

```python
import torch, torchvision 
import torchvision.models as models

alexnet = models.alexnet(pretrained=True)

alexnet.eval() ## freeze weights.

alexnet.train() ## trainable weights.

alexnet.eval()

logit = alexnet(image)

cifar10 = torchvision.datasets.CIFAR10(root='./', download=True)
for img, gt in cifar10:

    print(img.size, gt)

    plt.imshow(img)
    plt.show()
    
    ### TODO: transforms the image to pytorch tensor and forward into the alexnet model.
    
    img = normalizer(to_tensor(img))
    img = img.unsqueeze(0)
    
    print(img.shape)
    
    break
```

## VGG
- 16 and 19 layers
- simpler architecture 
- Only 3X3 conv filters with stride 1, 2X2 max-pooling and a few FC layers
- Input(224x224)
- Learning rate 0.01
- Gaussian distribution

```python
import torch, torchvision
import torchvision.models as models

models.vgg16()

cifar10 = torchvision.datasets.CIFAR10(root='./', download=True)

to_tensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                ])

vgg16 = models.vgg16(pretrained=True)

# dataloader
cifar10 = torchvision.datasets.CIFAR10(root='./', download=True, transform=to_tensor)  ## torch.utils.data.Dataset 

dataloader = torch.utils.data.DataLoader(cifar10, batch_size=8, shuffle=True, num_workers=2)

for idx, data in enumerate(dataloader):
#     
    img, gt = data
    
    print(img.shape, gt.shape)
    
#     for im in img:
#         plt.imshow(torchvision.transforms.ToPILImage()(im))
#         plt.show()
    
    ## TODO: run the vgg16 network
    logits = vgg16(img)
    
    print(logits.shape)
    
    break
```

## Explainable CNN

```python
from explainable_cnn import CNNExplainer
import pickle, torch
from torchvision import models

model = models.vgg16(pretrained=True)
```

## ResNet
- Depth is of crucial importance
- ReLu Activation
- Batch Normalization
  - Internal covariate shift
  - $\hat{x} = \frac{x - \mu}{\theta}$ normalization
  - $y = \gamma\hat{x} + \beta$ scaling & shifting
  - Normalizing each layer, for each mini-batch
  - Greatly accelerate training
  - Less sensitive to initalization
  - Improve regularization
- Degradation Problem
  - +identity
- simple but just deep

```python
models.resnet18()

### Model
resnet18 = models.resnet18(pretrained=True)

## Dataset
to_tensor = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
                                          )

cifar10 = torchvision.datasets.CIFAR10(root='./', download=True, transform=to_tensor)

dataloader = torch.utils.data.DataLoader(cifar10, batch_size=8, shuffle=True, num_workers=2)

for idx, data in enumerate(dataloader):
    
    img, gt = data
    
    print(img.shape)
    
    scores = resnet18(img)
    
    print(scores.shape)
    break

# Fine tunning
## TODO: 1. replace the last FC layer for cifar10
### Hint: 1000 -> 10

## TODO: 2. fine tuning the last classifier (FC layer) using the cifar 10 training set.

## TODO: 3. evaluation of the cifar 10 test set.
```