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
    
    
## Lab10-2 Mnist CNN  

- 학습 단계? (얼마든지 자의적으로 정의할 수 있는 부분)  
  
  ```  
  1. 라이브러리 가져오기 (torch, torchivision, matplotlib 등)  
  2. GPU 사용 설정 & Random seed 설정  
    -> cuda 사용가능할 때는 GPU 사용하여 연산, 그렇지 않으면 CPU 사용 
  3. 학습 parameter 설정 (learning_rate, training_epochs, batch_size 등) 
  4. Dataset 가져오기 & Data loader 만들기 
  5. 학습 모델 만들기 (class CNN(torch.nn.Module))
  6. Loss function (Criterion) 선택 & 최적화 도구 선택 Optimizer 
  7. 학습 및 Loss Check (Criterion의 Output)
  8. 학습된 모델의 성능 확인
  ```  
  ![pytorch03-09](https://user-images.githubusercontent.com/43376853/93193902-6c1efa00-f782-11ea-858b-2620a369fb8b.png)  

- 오늘 만들어 볼 CNN 구조?  
  ![pytorch03-10](https://user-images.githubusercontent.com/43376853/93194015-91136d00-f782-11ea-870e-4544c6ac7779.png)  
 
  - 그냥 Manual하게 손수 다 해보면?   
  ![pytorch03-11](https://user-images.githubusercontent.com/43376853/93196524-84dcdf00-f785-11ea-9126-3f0360b1c3de.png)  

  [코드 링크](https://github.com/data-weirdo/data-weirdo-playground/blob/master/PyTorch%20Basic/4.%EA%B0%84%EB%8B%A8%ED%95%9C%20CNN%20%EB%AA%A8%EB%8D%B8.ipynb)  
  
- 한 가지 의문: "Layer를 더 깊게 쌓으면 더 좋은 결과가 나오지 않을까?" 라는 궁금증   
  ![pytorch03-13](https://user-images.githubusercontent.com/43376853/93204007-51538200-f790-11ea-8851-5fff5dcffdd3.png)  
  
  - 결과: 이전의 모델보다 Accuracy가 떨어진다.  
  
    > 결론: 모델을 쌓을 때에는 모델을 깊게 쌓는 것도 중요하지만, 모델을 얼마나 효율적으로 쌓는가가 더 중요하다.  
 
 
---    

## Lab10-3 Visdom  

[코드 링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-10_3_1_visdom-example.ipynb)  

- 설치: `pip install visdom`  
- Visdom 서버 켜기: `pythom -m visdom.server`  
  -> Local Host 서버가 켜지게 됨! (해당 주소를 주소창에 입력하면 됨!, ex: http://localhost:8097)  
  
- Lab 10-3 목표: "Visdom 사용법 익히고 MNIST-CNN loss graph까지 적용해보기!"  
  - 다음과 같이 여러 이미지들을 띄워볼 수 있음!  
    ![pytorch03-14](https://user-images.githubusercontent.com/43376853/93209540-3cc7b780-f799-11ea-979b-25eabf244775.png)   
  
  - 선들도 다음과 같이 쉽게 그려볼 수 있음. (짱 신기하다..)  
    ![pytorch03-15](https://user-images.githubusercontent.com/43376853/93227258-24629780-f7af-11ea-9edf-b56ab9a4b0af.png)  

---  

## Lab10-4 ImageFolder1   
[코드 링크1](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-10_4_1_ImageFolder_1.ipynb)  
[코드 링크2](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-10_4_2_ImageFolder_2.ipynb)    

- 나만의 사진을 이용해서 딥러닝 태스크를 진행한다면?  
  - 데이터셋 준비방법 예시  
    ![pytorch03-16](https://user-images.githubusercontent.com/43376853/93292738-354bf100-f821-11ea-8184-963af2b5d67e.png)  
    - 클래스 별로 폴더 만들어서 이미지를 넣어두면 됨.  
    
- cf) `%ls`(현재 위치에서 커맨드 입력하기) -> 하위폴더들이 나옴  (cf. `% pwd`: 현재 워킫 디렉터리 볼 수 있음)  
      
- ImageFolder를 쓰면, 내 디렉터리의 사진들을 이용해서 모델링을 해볼 수 있음.  
  ![pytorch03-17](https://user-images.githubusercontent.com/43376853/93294599-f2404c80-f825-11ea-89e6-2e3250489eab.png)  

- 104-2는 내 사진이 없으므로.. 스킵.. ㅋ_ㅋ  

---  

## Lab10-5 Advance CNN(VGG)  

- Oxford VGG(Visual Geometry Group)에서 만든 Network  
- 다양한 형태들 존재  
  ![pytorch03-18](https://user-images.githubusercontent.com/43376853/93670217-c1397380-fad4-11ea-973e-d8fff722f533.png)  

- VGG16  
  ![pytorch03-19](https://user-images.githubusercontent.com/43376853/93670232-ea5a0400-fad4-11ea-8ab0-46647a558119.png)  
  
- `torchvision.models.vgg`    
  - vgg11 ~ vgg19까지 만들 수 있도록 되어 있음  
  - 3x224x224 입력을 기준으로 만들도록 되어 있음  
  
- [Vggnet Full code](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)  
- [Vgg with Cifar-10](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-10_5_2_Aadvance-CNN(VGG_cifar10).ipynb)  
  - `optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)`  
    : epoch을 돌리는 동안, learning rate을 조금씩 감소시키면서 학습  
    : step_size: lr_scheduler step을 5번할 때마다, learning rate에 0.9를 곱해달라는 것.  
    
---  

## Lab10-6-1 Advance CNN(RESNET-1) 

- 이 두 Block을 모두 만들어 볼 것!  
![pytorch03-20](https://user-images.githubusercontent.com/43376853/93710168-74f73d80-fb7f-11ea-8587-57b3ffb96c62.png)   

- 다양한 아키텍쳐들이 시도가 되었었는데, 이런 것들도 한 번 만들어 보겠다.  
![pytorch03-21](https://user-images.githubusercontent.com/43376853/93710181-98ba8380-fb7f-11ea-87d1-eb506ead731c.png)  

![pytorch03-22](https://user-images.githubusercontent.com/43376853/93710511-76763500-fb82-11ea-857e-70611cd49374.png)  

- `torchvision.models.resnet`   

- Downsample  
  - Stride가 2일 때, feature size가 줄어드니까, Identity 값도 함께 낮추어주기 위해 사용  
  - ResNet 코드에서는, 더불어 채널 사이즈를 맞추어주는 용도로도 사용함.  


- ResNet50 아키텍쳐 [Reference](https://cv-tricks.com/keras/understand-implement-resnets/)  
  ![pytorch03-23](https://user-images.githubusercontent.com/43376853/94026237-5c438d80-fdf4-11ea-990c-9d9376e2e3d9.png)  
  


---  

#### References  
[파이토치로 시작하는 딥러닝 기초 Part3](https://www.edwith.org/boostcourse-dl-pytorch/joinLectures/24017)    
[Detailed Guide to Understand and Implement ResNets](https://cv-tricks.com/keras/understand-implement-resnets/)  
  

  
  
 
      
  
