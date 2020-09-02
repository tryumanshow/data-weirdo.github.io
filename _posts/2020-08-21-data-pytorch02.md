---
layout: post
title:  "파이토치로 시작하는 딥러닝 기초 - Part2.DNN"
subtitle:   "deep-learning-part2"
categories: data
tags: pytorch
comments: true
---

- Deep Neural Network에 대한 소개  
  - [Perceptron](#-lab8-1.-perceptron)   
  - [Multi Layer Perceptron](#-lab8-2.-multi-layer-perceptron)  
  - [ReLU](#-lab9.-relu)  
  - [Weight Initialization](#-lab9-2.-weight-initialization)  
  - [Dropout](#-lab9-3.-dropout)  
  - [Batch Normalization](#-lab9-4.-batch-normalization)  

---  

# PART 2: DNN     
## Lab8-1. Perceptron    
- [Lab8-1 코드 링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-08_1_xor.ipynb)  

- Perceptron: 인공신공망의 한 종류  
  ![](https://upload.wikimedia.org/wikipedia/commons/f/ff/Rosenblattperceptron.png)  
  - Linear Classifier를 우해 만들어진 모델  
    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Perceptron_example.svg/600px-Perceptron_example.svg.png)  
  - AND, OR 문제를 해결하기 위해 만들어짐  
    ![](https://upload.wikimedia.org/wikipedia/commons/5/5d/Logic-gate-and-us.png)  
    ![](https://commons.wikimedia.org/wiki/File:Logic-gate-or-us.png)  
    - 이 두 Case는 Linear Classifier로 분류 가능  
    
  - 하지만, Perceptron 하나로는 XOR 문제는 풀 수 없음.  
    ![](https://upload.wikimedia.org/wikipedia/commons/c/c9/Logic-gate-xor-us.png)  
    
  ![](https://www.researchgate.net/profile/Peter_Ranon/publication/49654564/figure/tbl1/AS:601702222794759@1520468397225/Truth-Table-of-AND-OR-and-XOR-Gates_W640.jpg)  
    
    -> Multi Layer Perceptron 등장!  
    
---  

## Lab8-2. Multi Layer Perceptron    
- [Lab8-2 코드 링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-08_2_xor_nn.ipynb)  
    
- Multi Layer Perceptron  
  ![](https://texample.net/media/tikz/examples/PNG/neural-network.png)  
  - MLP를 통해 XOR 문제 해결이 가능  
  
- MLP를 학습할 수 있는 방법?  
  -> `Backpropagation`  
    ![](https://commons.wikimedia.org/wiki/File:Perceptron_XOR.jpg)  
    
  - Example Code  
  
    ```  
    X = torch.FloatTensor([0,0], [0,1], [1,0], [1,1]]).to(device)  
    Y = torch.FloatTensor([[0]], [1], [1], [0]]).to(device)
    
    # NN layers  
    linear1 = torch.nn.Linear(2, 2, bias=True)
    linear2 = torch.nn.Linear(2, 1, bias=True) # Linear1, Linear2: MLP
    sigmoid = torch.nn.Sigmoid()
    model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)
  
    # Cost 정의 & Optimizer
    criterion = torch.nn.BCELoss().to(device) # BCE: Binary Cross Entropy 
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    for step in range(1001):
      optimizer.zero_grad()
      hypothesis = model(X)
      cost = criterion(hypothesis, Y)
      cost.backward()
      optimizer.step()
      if stop % 100 == 0:
        print(step, cost.item())
    ```  
    
  - XOR NN 조금 더 깊게 쌓아 보자.   
    
    ```  
    X = torch.FloatTensor([0,0], [0,1], [1,0], [1,1]]).to(device)  
    Y = torch.FloatTensor([[0]], [1], [1], [0]]).to(device)
    
    # NN layers  
    linear1 = torch.nn.Linear(2, 10, bias=True)
    linear2 = torch.nn.Linear(10, 10, bias=True)
    linear3 = torch.nn.Linear(10, 10, bias=True)
    linear4 = torch.nn.Linear(10, 1, bias=True)  # 4 MLP 
    sigmoid = torch.nn.Sigmoid()
    
    ...  
    ```  
    : 앞의 두 개 짜리보다 Loss가 훨씬 더 작아진다~  
    
--  

## Lab9. ReLU    
- Activation Function을 Sigmoid로 사용했을 때의 문제점 극복 -> ReLU  
  - Sigmoid: Graidient를 계산하면서 문제가 생김  
    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sigmoid-function-2.svg/800px-Sigmoid-function-2.svg.png)  
    - -Inf, +Inf 쪽으로 가면 갈수록 그 기울기가 0에 가까워져서,  
      Backpropagation이 학습이 안되는 문제가 발생  
      더군다나 이 Sigmoid Function이 여러 개가 중첩될 때,  
      Input Layer에 가까워질수록 학습이 안 됨.  (`Vanishing Gradient` Problem)  
    - `torch.nn.sigmoid(x)`  
   
  - ReLU  
    - f(x) = max(0, x)
      - 0보다 큰 영역에서는 gradient가 1, 작은 영역에서는 0
      ![](https://t1.daumcdn.net/cfile/tistory/990A6A335981FFA437)  
    - Sigmoid보다 퍼포먼스 good  
    - 다만 ReLU 또한 0보다 작은 영역에서는 gradient가 0이 되는 문제점이 있기는 함  
      `torch.nn.relu(x)`  
      
  - 이 외에도 다양한 activation function들이 존재  
    ![](https://miro.medium.com/max/666/1*nrxtwp6rzqdFhgYh0x-eVw.png)  
    - `torch.nn.sigmoid(x)`  
    - `torch.nn.tanh(x)`  
    - `torch.nn.relu(x)`  
    - `torch.nn.leaky_relu(x, 0.01)`  등등  
    
#### Optimizer in PyTorch      
- 다양한 Optimizer 존재  ([torch.optim](https://pytorch.org/docs/master/optim.html#module-torch.optim))  
  [참고링크](http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html)  
  
  ```   
  - torch.optim.SGD
  - torch.optim.Adadelta  
  - torch.optim.Adagrad  
  - torch.optim.Adam  
  - torch.optim.SparseAdam  
  - torch.optim.Adamax 
  - torch.optim.ASGD
  - torch.optim.LBFGS
  - torch.optim.RMSprop  
  - torch.optim.Rprop 
  ```  
  
  - 각 Optimizer들의 직관적 이해 [링크](https://www.slideshare.net/yongho/ss-79607172)    
    ![](https://image.slidesharecdn.com/random-170910154045/95/-49-1024.jpg?cb=1505089848)  
  
  - 예시 코드  
    [Lab 9-1](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-09_1_mnist_softmax.ipynb)  
    [Lab 9-2](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-09_2_mnist_nn.ipynb)  
    [Lab 9-3](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-09_3_mnist_nn_xavier.ipynb)  
    [Lab 9-4](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-09_4_mnist_nn_deep.ipynb)  
    [Lab 9-5](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-09_5_mnist_nn_dropout.ipynb)  
    [Lab 9-6](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-09_6_mnist_batchnorm.ipynb)  
    [Lab 9-7](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-09_7_mnist_nn_selu(wip).ipynb)  
      
     
---  

## Lab9-2. Weight Initialization  
- Geoffrey Hinton 曰 "우리는 이제껏 weight initialization을 굉장히 멍청한 방법으로 행하고 있었다."  
  - Weight를 잘 Initialize하는지가 딥러닝의 성능에 상당한 영향을 미침.  

- 어떻게 Initialize?  
  - 모든 weight가 9이면 안 된다. -> Backpropagation 자체가 안됨    
  
  #### Method1. RBM  
  - 2006년, Hinton 교수님께서 RBM(Restricted Boltzmann Machine)을 제안  
    Hinton et al. (2006) "A Fast Learning Algorithm for Deep Belief Nets"  
    
    ![torch02-01](https://user-images.githubusercontent.com/43376853/91835080-dbc1bf00-ec83-11ea-8c0b-af3ab627cc75.png)  
    - 이에, Pre-training의 개념을 적용하여 Weigth Initialize  
      - Layer마다 학습하고, Fine-tuning  

    - 현재는 그리 많이 쓰이지는 않는다.  
    
  #### Method2. Xavier / He Initialization  
  - RBM에 비해 상당히 간단하게 Weight 초기화  
  - Layer의 특성에 따른 Initialization  
  
  ##### Xavier Initialization  
  - 방식1. Normal distribution으로 weight 초기화  
    ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20W%20%5Csim%20N%280%2C%20Var%28W%29%29)  
    ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20Var%28W%29%20%3D%20%5Csqrt%7B%5Cfrac%7B2%7D%7Bn_%7Bin%7D&plus;n_%7Bout%7D%7D%7D)  
    
  - 방식2. Uniform distribution으로 weight 초기화  
    ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20W%20%5Csim%20U%28-%5Csqrt%7B%5Cfrac%7B6%7D%7Bn_%7Bin%7D&plus;n_%7Bout%7D%7D%7D%2C&plus;%5Csqrt%7B%5Cfrac%7B6%7D%7Bn_%7Bin%7D&plus;n_%7Bout%7D%7D%7D%29)  
    
    cf) n in: layer의 input 수, n out: layer의 output 수  
    
  - PyTorch 패키지에 있는 공식 코드  
    
    ```  
    def xavier_uniform_(tensor, gain=1):  
    
      fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
      std = gain * math.sqrt(2.0 / (fan_in + fan_out))
      a = math.sqrt(3.0) * std
      with torch.no_grad():
        return tensor.uniform_(-a, a)
    ```  
  
  ##### He Initialization  
  - Xavier Initialization의 변형으로 보면 됨  
    
  - 방식1. Normal distribution으로 weight 초기화  
    ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20W%20%5Csim%20N%280%2C%20Var%28W%29%29)  
    ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20Var%28W%29%20%3D%20%5Csqrt%7B%5Cfrac%7B2%7D%7Bn_%7Bin%7D%7D%7D)  
    
  - 방식2. Uniform distribution으로 weight 초기화  
    ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20W%20%5Csim%20U%28-%5Csqrt%7B%5Cfrac%7B6%7D%7Bn_%7Bin%7D%7D%7D%2C%20&plus;-%5Csqrt%7B%5Cfrac%7B6%7D%7Bn_%7Bin%7D%7D%7D%29)  
    
    - `torch.nn.init.xavier_uniform_()`  
    
- Xavier 코드 적용 예시  

  ```  
  ## nn Layers 
  linear1 = torch.nn.Linear(784, 256, bias=True)
  linear2 = torch.nn.Linear(256, 256, bias=True)
  linear3 = torch.nn.Linear(256, 10, bias-True)
  relu = torch.nn.ReLU()  
  
  # xavier initialization   
  torch.nn.init.xavier_uniform_(linear1.weight)  
  torch.nn.init.xavier_uniform_(linear2.weight)
  torch.nn.init.xavier_uniform_(linear3.weight)  
  
  ...  
  ```  
 
---  
 
## Lab9-3. Dropout  
- `torch.nn.Dropout`  
- Dropout은 Overfitting을 피하는 데에 도움이 된다.  
  - Dropout을 사용하면 Network 앙상블의 효과를 얻을 수도 있다.  

- 예제 코드  
  
  ```  
  ## nn Layers  
  linear1 = torch.nn.Linear(784, 512, bias=True)
  linear2 = torch.nn.Linear(512, 512, bias=True)
  linear3 = torch.nn.Linear(512, 512, bias=True)
  linear4 = torch.nn.Linear(512, 512, bias=True)
  linear5 = torch.nn.LInear(512, 10, bias=True) 
  relu = torch.nn.ReLU()
  dropout = torch.nn.Dropout(p=drop_prob) 
  
  model = torch.nn.Sequential(linear1, relu, dropout, 
                              linear2, relu, dropout,
                              linear3, relu, dropout, 
                              linear4, relu, dropout, 
                              linear5).to(device)
  ```  

- 단, Dropout을 사용할 때 주의해야할 점은,  
  Train시에는 Dropout을 사용하지만, Test 시에는 모든 노드들을 사용해야 함.  

  - `model.train()`: Dropout을 사용  -> 학습을 할 때에는 반드시 `model.train()` 선언을 해주어야.  
  - `model.eval()`: Dropout을 사용x  -> Test를 할 때에는 반드시 `model.eval()` 선언을 해주어야.  
    
    [참고링크](https://pytorch.org/docs/stable/nn.html?highlight=eval#torch.nn.Module.eval)  
    
---     
    
## Lab9-4. Batch Normalization  
- Gradient Vanishing / Exploding  
  - Vanishing: Backpropagation시 Gradient가 너무 작아지면서 학습이 안 되는 현상  
  - Exploding: Backpropagation시 너무 큰 값들이 나오면서 발산해버리는 현상  
  
  - 이 두 문제가 있으면 학습이 어려워짐.  

- 해결방법  

  ```  
  1. Activation function을 바꾸어준다. 
    -> Sigmoid 대신 ReLU  
  2. Carful initialization  
    -> 초기화를 잘 해보자.  
  3. Small learning rate  
    -> Gradient Explosion 문제에 대한 해결책  
  ```  
  
- Batch Normalization  
  - Gradient Vanishing / Exploding 해결 뿐만 아니라 다른 이점들도 있다. (ex. 학습의 안정성)  
  
- Internal Covariate Shift  
  - Batch Normalization의 창시자들: "Vanishing 및 Exploding의 원인이다!"  
  
  - Cf. Covariate Shift  
    - Train Set과 Test Set간에 분포에 차이가 있다는 것  
  
  - Internal Covariate Shift?  
    - DL 상에서 Layer들을 통과하면서 분포가 변화하는 현상  
    - Layer가 많을수록 Covariate Shift는 더 많이 발생할 것!  
    - 이를 해결하기 위해 `Batch Normalization`  
      ![torch02-02](https://user-images.githubusercontent.com/43376853/91976076-66c1b880-ed5b-11ea-9f7e-5301a057d4ee.png)  
      (이름대로, Batch들마다 Normalization을 해주겠다는 것.)  
      - γ와 β 또한 학습 파라미터 (μ나 σㅎ는 학습 파라미터는 아님)  

  - 예시 코드: Batch Normalizatoin을 사용할 때와 하지 않을 때의 성능 차이 확인  
  
    ```  
    # nn Layers   
    linear1 = torch.nn.Linear(784, 32, bias=True)
    linear2 = torch.nn.Linear(32, 32, bias=True)
    linear3 = torch.nn.Linear(32, 10, bias=True)
    relu = torch.nn.ReLU()  
    bn1 = torch.nn.BatchNorm1d(32)
    bn2 = torch.nn.BatchNorm2d(32)
    
    nn_linear1 = torch.nn.linear(784, 32, bias=True)
    nn_linear2 = torch.nn_Linear(32, 32, bias=True)
    nn_linear3 = torch.nn.Linear(32, 10, bias=True)
    
    # model 
    # Bn은 ACtivation function 이전에 사용해주는 것이 일반적
    bn_model = torch.nn.Sequential(linear1, bn1, relu,  
                                  linear2, bn2, relu, 
                                  linear3).to(device)
    nn_model = torch.nn.Sequential(nn_linear, relu, 
                                  nn_linear2, relu, 
                                  nn_linear3).to(device)

    for epoch in range(training_epochs):
      bn_model.train() # Train mode 선언!  
      
      for X, Y in train_loader:
        X = X.view(-1, 28*28).to(device)
        Y = Y.to(device)
        
        bn_optimizer.zero_grad()
        bn_prediction = bn_model(X)
        bn_loss = criterion(bn_prediction, Y)
        bn_loss.backward()
        bn_optimizer.step()
        
        nn_optimzer.zero_grad()
        nn_prediction = nn_model(X)
        nn_loss = criterion(nn_prediction, Y)
        nn_loss.backward()
        nn_optimizer.step()
        
      ...  
      ```  
        
      - BatchNorm을 쓰면 더 결과가 잘 나온다.  
      
      

    
---  

## Reference  
[파이토치로 시작하는 딥러닝 기초 Part2](https://www.edwith.org/boostcourse-dl-pytorch/joinLectures/24016)  
