---
layout: post
title:  "파이토치로 시작하는 딥러닝 기초 - Part2.DNN"
subtitle:   "deep-learning-part2"
categories: data
tags: pytorch
comments: true
---

- Deep Neural Network에 대한 소개  
  - Perceptron   
  - Multi Layer Perceptron  
  - ReLU  
  - Weight Initialization  
  - Dropout  
  - Batch Normalization  

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

## Reference  
[파이토치로 시작하는 딥러닝 기초 Part2](https://www.edwith.org/boostcourse-dl-pytorch/joinLectures/24016)  
