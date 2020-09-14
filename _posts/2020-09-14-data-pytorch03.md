---
layout: post
title:  "파이토치로 시작하는 딥러닝 기초 - Part3.CNN"
subtitle:   "deep-learning-part3"
categories: data
tags: pytorch
comments: true
---

- Convolution Neural Network에 대한 소개  
  - Convolution  
  - MNIST  
  - Pytorch Visdom  
  - Pytorch Datasets & Custom Datasets  
  - CIFAR-10  
  - VGG & ResNet  

---  

# PART 3: CNN     
## Lab10-1 Convolution  
- Convolution  
  : 이미지 위에서 Stride값 만큼 Filter(Kernel)을 이동시키면서 겹쳐지는 부분의 각 원소의 값을 곱해서 모두 더한 값을 출력하는 연산  

- Stride  
  : Filter를 한 번에 얼마나 이동할 것인가.  
  
- Padding  
  : Zero-Padding 생각! (Input이미지 주변을 0으로 둘러쌈)  
  
- PyTorch 상의 2D Convolution 생성하기!  
  - `torch.nn.Conv2d`  
  
    > torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation, groups=1, bias=True)  
    
    ![pytorch03-01](https://user-images.githubusercontent.com/43376853/93089096-c615b600-f6d5-11ea-8f77-43cae671e351.png)  
    
    - 경우에 따라서는 3x3처럼 정사각형 모양의 Filter가 아니라, (3,1)과 같은 형태로 선언하는 그런 경우도 있음.  
    - 이럴 때는 kernel_sze 부분에 (3,1)와 같이 괄호를 쳐서 표기하면 됨  
    
- nn.Conv2d(1,1,3)을 만들었다고 가정  
  - Input Type: __torch.Tensor__ 타입이어야 한다.  
  - Input Shape: (N x C x H x W) 형태여야 함  
                 (batch_size, channel, height, width)  
                 
  - Output의 크기는?  
    : 다음과 같은 마법의 공식(?) 존재 (와웅 신기하다~~~ 짱짱맨)    
    ![pytorch03-02](https://user-images.githubusercontent.com/43376853/93090617-fd856200-f6d7-11ea-8389-502122f84d3a.png)  

    - 첫 번째 예제를 진행해보면  
      ![pytorch03-03](https://user-images.githubusercontent.com/43376853/93091931-b8fac600-f6d9-11ea-9dbf-490aed179729.png)  
      (짱 신기함...)   
      
      
 - Neuron과 Perceptron의 관계?  
  ![pytorch03-04](https://user-images.githubusercontent.com/43376853/93092903-fdd32c80-f6da-11ea-8e8f-17edd870f44c.png)  
  
  - 하지만 Filter 또한 Bias를 가질 수 있기 때문에 실제 출력은 다음과 같이 이루어진다.  
    ![pytorch03-05](https://user-images.githubusercontent.com/43376853/93093130-468ae580-f6db-11ea-9dde-9941c97395ca.png)  

- Pooling  
  - 이미지의 사이즈를 줄이기 위해  
  
  - Cf) Fully Connected 연산을 대체하기 위해 Average Pooling을 사용하는 경우도 존재  

  - `torch.nn.MaxPool2d`  
  
    > torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)  
      - kernel_size만 잘 설정해주면 되는 편 (나머지는 다 default 값이 있기 때문에)  
      
- ex)  
  
  ```  
  import torch
  import torch.nn as nn
  
  inputs = torch.Tensor(1,1,28,28)
  conv1 = nn.Conv2d(1, 5, 5)  
  pool = nn.MaxPool2d(2)
  out = conv1(inputs)  
  out2 = pool(out)
  
  #out.size()
  #out2.size()
  ```  
  
  ![pytorch03-06](https://user-images.githubusercontent.com/43376853/93094051-5fe06180-f6dc-11ea-9ac4-e9360e997cad.png)  
  

- 한 가지 더!! (About Conv2d)  
  ![pytorch03-07](https://user-images.githubusercontent.com/43376853/93094285-ae8dfb80-f6dc-11ea-8948-c58cfb4a863d.png)  
  
  - 'Convolution'이 아니라 'Cross-correlation Operator'라고???  
    ![pytorch03-08](https://user-images.githubusercontent.com/43376853/93095529-2ad50e80-f6de-11ea-9281-3f248a03c8d3.png)  
    
    - 이 그림에서 볼 수 있듯, 'Cross-correlation'이라고 적힌 이유는, Filter를 뒤집지 않고 계산하기 때문!  
    - 딥러닝 공부하면서 Convolution과 Cross-correlation을 크게 고민할 일은 없을 것  
    
    
---  

#### References  
[파이토치로 시작하는 딥러닝 기초 Part3](https://www.edwith.org/boostcourse-dl-pytorch/joinLectures/24017)    
  

  
  
 
      
  
