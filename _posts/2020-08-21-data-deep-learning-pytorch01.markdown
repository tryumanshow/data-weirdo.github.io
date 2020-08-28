---
layout: post
title:  "파이토치로 시작하는 딥러닝 기초 - Part1.Basic ML"
subtitle:   "deep-learning-part1"
categories: data
tags: deep_learning
comments: true
---

- PyTorch 사용에 관한 학습을 시작하기 전 먼저 알아야 할 기본적인 개념들     

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

---  

## Lab5. Logistic Regression    
- [Lab5 코드 링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-05_logistic_classification.ipynb)  

- Logistic Regression: Classification 문제  
- 문제 정의: Binary Classification  
  - Sigmoid: ![](https://latex.codecogs.com/gif.latex?H%28X%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-W%5ETX%7D%7D%20%5Capprox%20P%28X%3D1%29)    
  ![](https://wikidocs.net/images/page/22881/%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C%EA%B7%B8%EB%9E%98%ED%94%84.png)  
  
  - Cost Function   
    ![](https://latex.codecogs.com/gif.latex?cost%28W%29%20%3D%20-%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%7Bylog%28H%28x%29%29&plus;%281-y%29log%281-H%28x%29%29%7D)  
  
- 사용될 모듈 Import  
  
  ```  
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim 
  ```  

- Sigmoid 함수는 PyTorch에서 제공 : `torch.sigmoid()`  
- Costfunction도 PyTorch에서 제공: `F.binary_cross_entropy(hytothesis, y_train)`  

- 전체 훈련 과정 예시  
   
  ```  
  x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
  y_data = [[0], [0], [0], [1], [1], [1]]
  x_train = torch.FloatTensor(x_data)
  y_train = torch.FloatTensor(y_data)
  
  # 모델 초기화  
  W = torch.zeros((2,1), requires_grad=True)
  b = torch.zeros(1, requires_grad=True)
  # optimizer 설정  
  optimizer = optim.SGD([W,b], lr=1)
  
  nb_epochs = 1000
  for epoch in range(nb_epochs + 1):
      
      # Cost 계산  
      hypothesis = torch.sigmoid(x_train.matmul(W) + b) 
      cost = F.binary_cross_entropy(hypothesis, y_train)
      
      # cost로 H(x) 개선
      optimizer.zero_grad() # 기존에 구해놓은 gradient가 있으면 초기화를 해주어야. 
      cost.backward()
      optimizer.step()
      
      # 100번마다 로그 출력하기
      if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
        ))
      ```  
      
- 일련의 과정들을 Class로 좀 더 세련되게 표현하기  
  
  ```  
  class BinaryClassifier(nn.Module): # nn.Module을 상속받음  
    def __init__(self):
      super().__init__()
       self.linear = nn.Linear(8, 1)
       self.sigmoid = nn.Sigmoid()
       
    def forward(self, x):
      return self.sigmoid(self.linear(x))
  
  model = BinaryClassifier()
  ```  
  
  - cf. [nn.Linear()](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)  
  
  ```  
  # optimizer 설정
  optimizer = optim.SGD(model.parameters(), lr=1)

  nb_epochs = 100
  for epoch in range(nb_epochs + 1):

      # H(x) 계산
      hypothesis = model(x_train)

      # cost 계산
      cost = F.binary_cross_entropy(hypothesis, y_train)

      # cost로 H(x) 개선
      optimizer.zero_grad()
      cost.backward()
      optimizer.step()

      # 20번마다 로그 출력
      if epoch % 10 == 0:
          prediction = hypothesis >= torch.FloatTensor([0.5])
          correct_prediction = prediction.float() == y_train
          accuracy = correct_prediction.sum().item() / len(correct_prediction)
          print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
              epoch, nb_epochs, cost.item(), accuracy * 100,
          ))
  ```  
  
  
---  

## Lab6. Softmax Classification  
- [Lab6 코드 링크-1](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-06_1_softmax_classification.ipynb)  
- [Lab6 코드 링크-2](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-06_2_fancy_softmax_classification.ipynb)  

- ![](https://latex.codecogs.com/gif.latex?P%28class%3Di%29%20%3D%20%5Cfrac%7Be_i%7D%7B%5Csum%7Be_j%7D%7D)  
- Softmax 함수는 PyTorch에서 제공: `F.softmax()`  
  
- Cross Entropy  
  - 두 확률분포가 얼마나 비슷한지를 나타내는 수치  
    ![](https://latex.codecogs.com/gif.latex?H%28P%2CQ%29%20%3D%20-%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20P%28x%29%7D%5BlogQ%28x%29%5D%20%3D%20-%5Csum_%7Bx%5Cin%20X%7DP%28x%29logQ%28x%29)  
    
    
  - [Low-level의 구현](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-06_1_softmax_classification.ipynb)    
  - [High-level의 구현](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-06_2_fancy_softmax_classification.ipynb)   
    - `F.log_softmax(z, dim=1)`  
    - `F.nll_loss(F.log_softmax(z, dim=1), y)`  # negative loglikelihood  
    - `F.cross_entropy(z, y)`  
    
    ```  
    # High-level Implementation with nn.Module 
    
    class SoftmaxClassifierModel(nn.Module):  
      def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output이 3
        
      def forward(Self, x):
        return self.linear(x)
        
    model = SoftmaxClassifierModel()
    ```  
    
  
- 요컨대  
  - Binary Classification: Sigmoid 사용  
  - Multinomial Classification: Softmax 사용  
 
---  


## Lab7-1. Tips  
- [Lab7 코드 링크-1](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-07_1_tips.ipynb)  


- MLE (Maximum Likelihood Estimation)  
  - 관찰한 데이터를 가장 잘 설명하는 pdf의 `parameter`를 찾는 것  
    - 이 parameter를 찾는 방법 (Optimization via Gradient Ascent)  
      ![](https://latex.codecogs.com/gif.latex?%5Ctheta%20%5Cleftarrow%20%5Ctheta%20-%20%5Calpha%20%5Cbigtriangledown_%7B%5Ctheta%7DL%28x%3B%5Ctheta%29)  
      - 'Local Maxima' 찾기  
      
- Overfitting  
  - MLE는 숙명적으로 Overfitting이 따르게 된다.  
    (주어진 데이터에 대해 그 데이터를 가장 잘 설명하는 pdf를 찾다보니 당연히 overfitting이 발생)  
  - Overfitting을 최소화하는 방법?  
    : Training Set(.0.8) / Validation Set(0~0.1) / Test Set(0.1~0.2)으로 Observation을 나누기.  
    
  - Training Set으로 훈련을 하면 epoch이 늘어날수록 Train Loss가 꾸준히 줄어든다.  
  - 하지만 Validation Set은 epoch이 늘어날수록 Validation Loss가 줄다가 다시금 늘어난다.  
    - 여기서 다시금 늘어나는 그 지점부터가 `Overfitting`이 일어나기 시작하는 것  
      
  
  - Overfitting을 방지하는 방법?  
    
    ```  
    1. 데이터를 더 많이 모은다.  
    2. Feature 개수를 줄인다.  
    3. Regularization 
    ```  
    
    - Regularization  
      
      ```  
      - Early Stopping (Validation Loss가 더 이상 낮아지지 않을 때까지.)
      - Reducing Network Size  (딥러닝의 경우)
      - Weight Decay  
      - Dropout
      - Batch Normalization 
      ```  
      
      - DeepLearning에 한해 Dropout과 Batch Normalization이 가장 많이 사용된다~  
      ```  
      
- DNN 학습하는 의사결정 과정?  
  
  ```  
  1. 신경망 아키텍쳐를 만든다. (물론 Input, Output size는 고정)  
  2. 모델을 훈련하고, 그 모델이 overfitting 된 모델인지 확인한다. 
    - 만약 Overfitted: Drop-out이나 Batch-Normalization과 같은 Regularization 실행 
    - 만약 Not overfitted: 모델 크기 증가 (Deeper & Wider)
  3. Step2를 반복  
  ```  
- '좋은' Learning Rate는 데이터와 모델에 따라 굉장히 달라질 수가 있기 때문에,   
  딱히 '어떤 값을 사용해라!' 라고 할 수는 없고, 다만 lr=1e-1와 같은 값을 사용했을 때  
  학습이 너무 느리다든지, 아니면 큰 값을 썼는데 발산을 해버린다든지, 그런 경우들을  
  잘 고려하여 적절하 숫자를 잘 찾아내자.  
  
  
- 데이터 전처리  
  - 굉장히 중요~  
  - 데이터 전처리를 하면 학습이 훨씬 더 수월해짐  
    - ex. Standardization  
      ![](https://latex.codecogs.com/gif.latex?x_j%5E%7B%27%7D%20%3D%20%5Cfrac%7Bx_j-%5Cmu_j%7D%7B%5Csigma_j%7D)  
  - NN은 전처리를 하지 않으면, 값이 큰 특정 column의 학습에만 힘을 쏟을 것.  
    - NN 모델을 잘 만들고, 코딩을 잘 하는 것도 중요하지만, 그것만큼이나 데이터의 성격을 파악하고 전처리를 잘 해주는 것 또한 굉장히 중요.  
      ∵ MLE를 Gradient Descent 방법을 사용해서 구하는데, 최적화가 원활히 이뤄지지 않으면 최적의 파라미터를 찾을 수 없다. 
    
  
## Lab7-2. MNIST Introduction    
- [Lab7 코드 링크-2](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-07_2_mnist_introduction.ipynb)  

- MNIST 데이터 셋  
  - 손으로 쓰여진 0~9의 숫자 이미지  
  - 우편번호 자동 인식이 목적  
  - Train set: 60000장의 Image & Label  
  - Test set: 10000장의 Image & Label  
  
  ```  
  28 x 28 Image  
  1 channel gray image  
  0~9 digits  
  ```  

- `torchvision` 패키지   
  - 유명한 데이터 셋들, 모델 아키텍쳐들, 다양한 Transformation들로 구성됨  
    [참고](https://pytorch.org/docs/stable/torchvision/index.html)  
  
  - 코드 부분 주석   
  
    ```  
    import torchvision.datasets as dsets
    
    mnist_train = dsets.MNIST(root="MNIST_data/", 
                              train=True, 
                              transform = transform.ToTensor(), 
                              download=True)
    ```  
    
    - root: 어디에 MNIST 데이터가 있는가,  
    - train=True: MNIST 데이터의 Trainset을 불러오겠다.  
    - transform: MNIST 데이터셋을 불러올 때 어떤 Transform을 적용해서 불러올 거냐?  
      - 일반적으로 PyTorch의 경우 이미지는 0~1 사이의 값을 갖게 되고, 순서는 Channel - Height - Width  
      - 일반적인 이미지는 0~255의 값을 갖게 되고, Height - Width - Channel의 순서  
      - To Tensor는 후자를 PyTorch에 맞게 전자로 바꾸어 준다.  
    - download=True: 만약 root에 MNIST 데이터가 존재하지 않으면 다운로드 받겠다.   

    ```  
    data_loader = torch.utils.DataLoader(DataLoader=mnist_train, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        drop_last=True)
    ```  
    
    - DataLoader: 어떤 데이터를 load할 것인가.  
    - batch_size: 이미지를 불러올 때 몇 개씩 잘라서 불러 올래?  
    - shuffle = True: 섞어서 불러올래? (True)  
    - drop_last = True: 배치 사이즈만큼 잘라서 불러올 때 뒤에 남는 데이터들을 자르자.  
    
    ```  
    for epoch in range(training_epochs):
    ...  
      for X, Y in data_loader: # X: image, Y: label 
        X = X.view(-1, 28*28).to(device) # view를 이용해서 28 by 28을 784로 바꾸어 준다.  
    ```  
  
    [Epoch  Batch size / Iteration 참고](https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks)  
    - Epoch  
      : 훈련 셋 전체가 학습에 한 번 사용이 된다? = 1 epoch  
    - Batch Size   
      : Training Set을 몇 개 단위로 자를 거냐?  
    - Iteration  
      : Batch를 몇 번 학습에 사용?  

    - ex) 
      1000개의 Training Set  
      Batch Size: 500  
      -> Epoch 한 번 도는 데에 2번의 Iteration 소요  

  - Classifier 학습하기  
    
    ```  
    linear = torch.nn.Linear(784, 10, bias=True).to(device) # ∵ MNIST 데이터 이미지 shape: 784  
    
    training_epochs = 15
    batch_size = 100
    
    criterion = torch.nn.CrossEntropyLoss().to(device) # PyTorch: CrossEntropyLoss가 자동으로 Softmax 계산  
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)  
    
    for epoch in range(training_epochs):
      avg_cost = 0
      total_batch = len(data_loader)  
      for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device)
        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch  
        
      print("Epoch: ", "%04d" % (epoch+1), "cost = ", "{:.9f}.format(avg_cost))"
      
    ```  
      
- Test 하기  
  
  ```  
  With torch.no_grad(): # 'Gradient는 계산하지 않겠다.'(Test시 항상 사용하는 습관 들이자.)
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prdiction.float().mean()
    pritn("Accuracy: ", accuracy.item())
  ```  
  
- Image를 이용하기 때문에 Visualization도 필요~  

  ```  
  import matploblib.pyplot as plt
  import random  
  
  r = random.randint(0, len(mnist_test) - 1)
  X_single_data = mnist_test.test_data[r: r+1].view(-1, 28*28).float().to(devicee)
  Y_single_data = mnist_test.test_labels[r: r+1].to(device)
  
  print("Label: ", Y_single_data.item())
  single_prediction = linear(X_single_data)
  print("Prediction: ", torch.argmax(single_prediction, 1).item())
  
  plt.imshow(mnist_test.test_data[r:r+1].view(28,28), cmap='Greys', interpolation='nearest')
  plot.show()
  ```  
    
---  

#### Reference
[파이토치로 시작하는 딥러닝 기초](https://www.edwith.org/boostcourse-dl-pytorch/lecture/42994/)  
