---
layout: post
title:  "파이토치로 시작하는 딥러닝 기초-Basic ML"
subtitle:   "deep-learning-part1"
categories: data
tags: deep_learning
comments: true
---

- 기본적인 머신러닝에 대한 정리    

---  

# 실습환경설정  
## 도커 환경 설정  
- 도커 왜 써야 하나?  
  - A라는 사람의 컴퓨터에서는 되는데, B라는 사람의 컴퓨터에서는 안 되는 그런 상황을 해결  
  
- `도커(Docker)`란?  
  - 컨테이너 기반의 가상화 시스템 (Container-Based Virtualization System)  
    - 가상화 (Virtualization)?  
      - 실제로는 없는 것을 마치 존재하는 것처럼 보여주는 기술.  
      - 클라우드 시장의 핵심 기술  
        → 물리적인 서버 하나를 여러 개의 가상서버로 쪼개서 각각을 빌려준다.  
        
  - 하나의 컴퓨터에서 여러 개의 독립된 OS를 두는 가상화: 속도 ↓  
    - 리눅스: '독립된 여러 개의 OS를 띄우지 말고, Host OS 위에 Docker를 설치해서 어느 컴퓨터에서든 똑같이 돌아갈 수 있게 하자!'  
    
- 해당 강의에서 왜 Docker를 언급?  
  - 컨테이너 이미지(Image)만 다운 받으면, 모두가 똑같은 환경에서 딥러닝 활용 가능   
  - 만약에, 뭔가 꼬였다? 해당 컨테이너 날리고 새로운 컨테이너 만들면 됨  
    
- 도커 설치 방법  
  - cf. Windows, MAC OS: 모두 도커 사용할 수 있긴 하지만, Docker의 원래 대상은 `Linux`를 위한 것  
    - 그래서 별도의 Virtual Machine이나 Hypervisor 사용  
      → Linux 만큼의 성능은 나오지 않을 수 있다.  
      → GPU를 사용할 수 없다.  
      
  - Windows 7, 8, 10 (64-bit) 설치 가정  
    - `Docker Toolbox`를 설치할 것  
    
      ```  
      1. 구글에서 "Docker Toolbox" 검색  
      2. Download & Install (설치 완료시: Docker Quickstart Terminal이라는 이름의 파일 생김)   
      3. 관리자 실행  
      4. 설치 완료 후, 화면에 'docker run hell-world'를 입력했을 때 
         Hello from Docker라는 명령어가 나오면 성공한 것  
      ```  
  - MacOS   
    
      ```  
      1. macOS 버전 체크 - 10.10 이상인가?  
      2. If 1 is satisfied → Docker.com 으로 이동해서 Get Started
      3. Download for MAC  
      4. 로그인 후 Get Docker  
      5. Open and Install  
      6. 완료 후 Windows의 Step4와 동일하게.  
      ```  
  - Ubuntu  
    - 리눅스 커널 사용하기 때문에 매우 간단.  
  
      ```  
      터미널을 열고 
      1. curl -fsSL https://get.docker.com > docker.sh
      2. sudo sh docker.sh  
      3. sudo docker run hello-world  
      ```  
---   
    
# PART 1: Machine Learning & PyTorch Basic    
## Lab1. Tensor Manipulation 1&2  
- [Lab1 코드 링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-01_tensor_manipulation.ipynb)  

  ```  
  소개 내용  
  - Vector, Matrix, and Tensor  
  - NumPy Review  
  - PyTorch Tensor Allocation  
  - Matrix Multiplication  
  - Other Basic Ops  
  ```  
  
- Vector: 1차원  
- Matrix: 2차원  
- `Tensor`: 3차원  

- Tensor Shape (텐서의 크기에 대한 이해 중요.)   
  - 2D Tensor  
    - `(batch size, dim)`  
      ![](https://wikidocs.net/images/page/52460/tensor3.PNG)  
    - ex) (64, 256) 형태가 대부분일 것  
    
  - 3D Tensor1  (ex. Computer Vision)  
    - `(batch size, width, height)`  
      ![](https://wikidocs.net/images/page/52460/tensor5.PNG)  
      
  - 3D Tensor2 (ex. NLP, 시계열 데이터)  
    - `(batch size, length, dim)`  
      ![](https://wikidocs.net/images/page/52460/tensor6.PNG)  
      - 1 Batch size = 1 Batch Size   
      - Length는 Time Step으로 해석 가능  
      
- Numpy Review  
  - 1D Array with NumPy  
    2D Array with NumPy  
    ↓ Pytorch 버전: (`torch.FloatTensor()`)   
  - 1D Array with PyTorch (dim, shape, size)  
    2D Array with PyTorch (dim, size)  
  
  - Shape, Rank, Axis   
  - Broadcasting  
    : 자동으로 실행되기 때문에 사용자 입장에서는 조심해야.  
    - Same Shape  
    - Vector + Scalar  
    - 2x1 Vector + 1x2 Vector  
  - Mul vs. Matmul    
    - 딥러닝은 행렬곱을 많이 사용하므로 Matmul에 주의  
  - Mean  
    - 원하는 차원에서만 평균 구하기? (feat. `dim()`)  
  - Sum  
  - Max, Argmax  
  
  - `View` ★    
    - numpy의 reshape  
    - View 함수를 잘 쓰는 것 또한 DL에서 굉장히 중요  
  - Squeeze  
    - 역시 dim을 파라미터로 줄 수도 있다.  
  - Unsqueeze  
    - dim을 꼭 명시해주어야 한다. (ex. x.unsqueeze(dim=1))  
  - Scatter   
  - Type Casting  
    - ex) torch.LongTensor(), torch.ByteTensor()  
  - Concatenation  
    - `torch.cat`  
  - Stacking  
    - Concatenation의 조금 더 편리한 Version  
  - Ones and Zeros Like  
    - `torch.ones_like()`  
    - `torch.zeros_like()`  
  - In-place Operation  
    - inplace operation에는 `_`가 붙는다!  
    - x.mul(2.) vs x.mul_(2.)  
    - Cf. PyTorch는 사실 Garbage Collector가 효율적으로 잘 설계되어 있기 때문에, in-place 사용하더라도 속도 면에서 크게 이점이 없을 수도 있다.  
  
  - Zip  
  
  - 등등 다양한 연산들이 존재  

---  

## Lab2. Linear Regression  
- [Lab2 코드 링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-02_linear_regression.ipynb)  
- 공부 시간과 점수의 상관관계? 
- y=Wx+b  (W: Weight, b: Bias)  
  
  ```  
  x_train = torch.FloatTensor([[1], [2], [3]])
  y_train = torch.FloatTensor([[2], [4], [6]])
  
  # Weight와 Bias를 0으로 초기화 
  # requres_grad = True: 학습할 것임을 명시하는 것. 
  W = torch.zeros(1, requires_grad=True)
  b = torch.zeros(1, requires_grad=True)
  hypothesis = x_train * W + b
  
  # MSE 계산 
  cost = torch.mean(hypothesis - y_train) ** 2)
  
  # Gradient descent  
  optimizer = optim.SGD([W, b], lr=0.01)
  
  optimizer.zero_grad() # gradient 초기화
  cost.backward() # gradient 계산
  optimizer.step() # 개선
  ```
  
- Full Training Code  

  ```  
  ### Part A: 한 번만
  x_train = torch.FloatTensor([[1], [2], [3]])
  y_train = torch.FloatTensor([[2], [4], [6]])

  W = torch.zeros(1, requires_grad = True)
  b = torch.zeros(1, requires_grad = True)

  optimizer = optim.SGD([W,b], lr=0.01)
  
  ### Part B: 반복!
  nb_epochs = 1000  
  for epoch in range(1, nb_epochs+1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
  ```  
  
  - Part A: 한 번만!
    > 1. 데이터 정의  
    > 2. Hypothesis 초기화
    > 3. Optimizer 정의  
    
  - Part B: 반복!  
    > 1. Hypothesis 예측  
    > 2. Cost 계산  
    > 3. Optimizer로 학습  
    
---       
    
## Lab3. Deeper Look at GD  
- [Lab3 코드 링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-03_minimizing_cost.ipynb)  

- 회귀의 Cost function: MSE  
- Gradient Descent  
  W := W-α∇W  
  
- Full Code  

  ```  
  # 데이터
  x_train = torch.FloatTensor([[1], [2], [3]])  
  y_train = torch.FloatTensor([[1], [2], [3]]) 
  # 모델 초기화  
  W = torch.zeros(1, requires_grad=True)
  # optimizer 설정  
  optimizer = optim.SGD([W], lr=0.15)  
  
  nb_epochs = 10
  for epoch in rangE(np_epochs+1):
  
    # H(x) 계산  
    hypothesis = x_train * W  
    # cost 계산  
    cost = torch.mean((hypothesis - y_train) ** 2)  
    
    print('Epoch {:4d}/{} W: {:.3f} Cost: {:.6f}'.format(
      epoch, nb_epochs, W.item(), cost.item()
    ))
    
    # cost로 H(x) 계산  
    optimizer.zero_grad()  
    cost.backward()
    optimizer.step()
  ```  
  
---  

## Lab4-1. Multivariable Linear Regression 
- [Lab4-1 코드 링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-04_1_multivariable_linear_regression.ipynb)  

- Multivariable Linear Regression  
  - Simple LInear Regresseion과 사고는 다를 게 전혀 없음.  
  - x_train, y_train의 선언만 차원을 달리 해주는 것 밖에 없음.  
  
- 근데 어쨌든, 이렇게 직접 데이터들을 다 적어준다는 건, 차원이 커지면 상당히 비효율적인 일  
  → PyTorch는 `nn.Module`이라는 편리한 모듈 제공  
  
    ```  
    # Example  
    import torch.nn as nn
    
    class MultivariateLinearRegressionModule(nn.Module):  
      def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        
      def forward(self, x):
        return self.linear(x)
    ```    
    
    ```  
    # 데이터
    x_train = torch.FloatTensor([[73, 80, 75],
                               [93, 88, 93],
                               [89, 91, 90],
                               [96, 98, 100],
                               [73, 66, 70]])
    y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
    # 모델 초기화
    model = MultivariateLinearRegressionModel()
    # optimizer 설정
    optimizer = optim.SGD(model.parameters(), lr=1e-5)

    nb_epochs = 20
    for epoch in range(nb_epochs+1):
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 20번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
    ```  
## Lab4-2. Loading Data  
- [Lab4-2 코드 링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-04_2_load_data.ipynb)  

- 복잡한 ML 모델의 학습을 위해서는 엄청나게 많은 양의 데이터가 필요  
  - 데이터 多: 데이터를 한 번에 학습시키는 것은 불가능해짐  
    → 해결책: Mini-Batch Gradient Descent  
      ![](https://wikidocs.net/images/page/55580/%EB%AF%B8%EB%8B%88%EB%B0%B0%EC%B9%98.PNG)  
      
    - 단, Batch gradient descent 방법에 비해 Mini-batch Gradient Descent 방법은  
      매끄럽게 학습이 되지는 않음 (mini batch # vs. cost 그래프)  
      
- 코드 등장  
  - torch.utils.data.Dataset  
  - \__len\__()  
  - \__getitem\__()    



#### Reference
[파이토치로 시작하는 딥러닝 기초](https://www.edwith.org/boostcourse-dl-pytorch/lecture/42994/)  
