---
layout: post
title:  "파이토치로 시작하는 딥러닝 기초 - Part4.RNN"
subtitle:   "deep-learning-part4"
categories: data
tags: pytorch
comments: true
---

- Recurrent Neural Network에 대한 소개  
  - RNN 기초  
  - RNN hihello & charseq  
  - Long sequence  
  - Timeseries  
  - seq2seq  
  - PackedSequence  

---  

# PART 4: RNN     
## Lab11-0. RNN intro  
- Sequential 데이터를 잘 다루기 위해 만들어진 모델  
  (데이터 값 뿐만 아니라 '순서'도 중요한 데이터)  
  - ex) 단어, 문장, 시계열 etc  
  
- RNN  
  ![pytorch04-1](https://user-images.githubusercontent.com/43376853/94040294-b861de00-fe03-11ea-863c-1173616da3a6.png)  

  - 모든 Cell이 같은 Parameter를 공유한다.  
  - A에서는  
    ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20h_t%20%3D%20f%28h_%7Bt-1%7D%2C%20x_t%29)  
    - ex  
      ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20h_t%20%3D%20tanh%28W_h%20h_%7Bt-1%7D%20&plus;%20W_x%20x_t%29)  
      
- RNN의 사용   
  - one to many  
    - ex) 입력: 하나의 이미지 -> 출력: 문장  
  - many to one  
    - ex) 입력: 문장 -> 츨략: 감정에 대한 label  
  - many to many  
    - ex) 입력: 문장 -> 출력: 문장  
  

## Lab11-1. RNN basics  
- PyTorch에서의 RNN 사용?  

  ```
  rnn = torch.nn.RNN(input_size, hidden_size) # 둘 다 integer
  outputs, _status = rnn(input_data) #input_data: 입력코자 하는 데이터  
  ```  
  
  - input 데이터의 shape: 3개의 차원을 가짐 (x, y, z)  
  
- 단어 'hello'를 RNN에 입력하는 과정?  

  ```  
  # One hot encoding  
  h = [1, 0, 0, 0]
  e = [0, 1, 0, 0]
  l = [0, 0, 1, 0]
  o = [0, 0, 0, 1]
  
  input_size = 4 # 4개의 차원을 input으로 받는다. -> input data의 shape는 (-,-,4) 형태 
  # word embedding을 사용할 경우에는 embedding vector의 dimension이 input size가 됨  
  # 이 input size는 model에 직접 알려주어야 함  
  
  hidden_size = 2 # 어떤 사이즈의 벡터를 출력하기를 원하는가?  -> output의 shape는 (-,-,2) 형태  
  # ex. 기쁨, 슬픔 중 하나 출력  
  # 역시 model에게 미리 이를 선언해주어야 함  
  
  # 결론적으로, 4차원 벡터를 받아서 2차원 벡터를 출력하는 꼴  
  
  ...   
  ```  

- 여기서 드는 의문  
  '어떻게 hidden state의 dimension이 output size와 동일한 차원을 갖게 되는가?'  
  - RNN cell 내부구조를 보면 이해할 수 있음  
  
    ![pytorch04-2](https://user-images.githubusercontent.com/43376853/94042187-34f5bc00-fe06-11ea-8df8-98a84f205175.png)  
  
    - 출력직전에 두 개의 가지로 갈라지는 설계!  
    
- Sequence Length?  
  - HELLO: Sequence Length=5  
    x0 = [1,0,0,0]  
    x1 = [0,1,0,0]  
    x2 = [0,0,1,0]  
    x3 = [0,0,1,0]  
    x4 = [0,0,0,1]   
  - Pytorch는 Sequence Length를 굳이 알려주지 않아도 알아서 파악함 
    -> input data의 shape는 (-,5,4)  
    - 따라서 input data만 잘 만들어주면 된다.  
    -> output data의 shape는 (-,5,2)  
    
- Batch Size?  
  - RNN 또한 여러 개의 데이터를 하나의 Batch로 묶어서 모델에게 학습시킬 수 있다.  
  - 예를 들어, h,e,l,o 이 4개의 문자로 만들어진 수많은 단어들 중에서  
    h,e,l,l,o / e,o,l,l,l / l,l,e,e,l 과 같이 주어졌을 때, 이 세 개를 단어를 묶어서 하나의 배치로 구성하고  
    이를 모델에게 전달.  
    
  - Batchsize 역시 모델이 자동으로 파악  
    -> Input data의 shape는 (3, 5, 4)  
    -> Output data의 shape는 (3, 5, 2)  
    
- (Batch size, Sequence length, embedding size)  
    
- 전체 코드  

  ```  
  import torch
  import numpy as np
  
  input_size = 4
  hidden_size = 2
  
  # One hot encoding  
  h = [1, 0, 0, 0]
  e = [0, 1, 0, 0]
  l = [0, 0, 1, 0]
  o = [0, 0, 0, 1]
    
  input_data_np = np.array([[h, e, l, l, o],
                            [e, o, l, l, l],
                            [l, l, e, e, l]], dtype=np.float32)
                            
  # Torch Tensor로 Transform  
  input_data = torch.Tensor(input_data_np)
  
  # RNN 모델 
  rnn = torch.nn.RNN(input_size, hidden_size)
  
  outputs, _status = rnn(input_data)
  ```  
  ---  
  
  ## Lab11-2. RNN hihello and charseq   
  - Character들을 어떻게 표현하는 것이 좋을까?  
    - 방법1. By index?  
      
      ---    
      'h' -> 0  
      'i' -> 1  
      'e' -> 2  
      'l' -> 3  
      'o' -> 4  
      ---  
      
      그닥 좋은 방법은 아님  
      ∵ 숫자의 크기에 따라 의미를 주는 셈.  
      
    - 방법2. One-hot Encoding  
      
      ```  
      char_set = ['h', 'i', 'e', 'l', 'o']
      x_data = [[0,1,0,2,3,3]]
      x_one_hot = [[[1,0,0,0,0],
                    [0,1,0,0,0],
                    [1,0,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,1,0],
                    [0,0,0,1,0]]]
      y_data = [[1,0,2,3,3,4]]
      ```  
      
- Cross Entropy Loss  
  - Categorical output을 예측하는 모델에서 주로 쓰임  
  
    ```  
    criterion = torch.nn.CrossEntropyLoss()
    ...
    # 첫번째 param: 모델의 output, 두번째 param: 정답 Label  
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))  
    ```  
        
- 'hihello' Code  

  ```  
  char_set = ['h', 'i', 'e', 'l', 'o']
  # hyper parameters
  input_size = len(char_set)
  hidden_size = len(char_set)
  learning_rate = 0.1
  
  # data setting  
  x_data = [[0,1,0,2,3,3]]
  x_one_hot = [[[1,0,0,0,0],
                [0,1,0,0,0],
                [1,0,0,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
                [0,0,0,1,0]]]
  y_data = [[1,0,2,3,3,4]]
    
  X = torch.FloatTensor(x_one_hot)  
  Y = torch.LongTensor(y_data)  
  ```  
      
- charseq Code  

  ```  
  import numpy as np
  
  sample = ' if you want you'  
  # 딕셔너리 만들기  
  char_set = list(set(sample))
  char_dic = {c: i for i, c in enumerate(char_set)}
  
  # hyper parameters
  dic_size = len(char_dic)
  hidden_size = len(char_dic)
  learning_rate = 0.1
  
  # data setting
  sample_idx = [char_dic[c] for c in sample]
  x_data = [sample_idx[:-1]]
  x_one_hot = [np.eye(dic_size)[x] for x in x_data] # np.eye: Identity Matrix를 만들어 줌  
  y_data = [sample_idx[1:]]
  
  X = torch.FloatTensor(x_one_hot)
  Y = torch.LongTensor(y_data)
  
  ```  
  
  - 이어서, RNN 모델 만들기  
  
  ```  
  # RNN 선언하기 
  # batch_first = True -> OUtput의 순서가 (B, S, F)임을 보장함  
  rnn = torch.nn.RNN(input_size, hidden_size, batch_first = True) 
  
  # loss & optimizer setting
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(rnn.parameters(), learning_rate)
  
  # 학습 시작  
  for i in range(100):
    optimizer.zero_grad()
    outputs, _status = rnn(X) # X: input, _status: 만약 다음 input이 있다면 다음 RNN 계산에 쓰이게 될 hidden state
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward()
    optimizer.step()
    result = outputs.data.numpy().argmax(axis=2) # Dim=2 -> 어떤 character인지를 나타냄 
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, 'loss: ', loss.item(), 'prediction: ', result, 'true Y: ', y_data, 
        'prediction str: ', result_str)
  ```  
  
---  

## Lab11-3. Long sequence  
- 11-2 보다 조금 더 긴 Character Sequence 모델링  

- 아주 긴 문장을 하나의 Input으로 사용할 수는 없기에, 특정 사이즈로 잘라서 써야 함.  

- 예시  

  ```  
  sentence = ("if you want to build a ship, don't drum up people together to ",
              "collect wood and don't assign them tasks and work, but rather ",
              "teach them to long for the endless immensity of the sea.")
  ```  
  
  - Size 10의 chunk를 생각해보자.  
    ![pytorch04-3](https://user-images.githubusercontent.com/43376853/94103691-352aa180-fe70-11ea-8fb1-35b5b35e6126.png)  
  
- Sequence dataset 만들기  

  ```  
  # 데이터 셋팅
  x_data = []
  y_data = []
  
  for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i: i+sequence_length]
    y_str = sentence[i+1: i+sequence_length+1]
    print(i, x_str, '->', y_str)
    x_data.append([char_dic[c] for c in x_str]) # x str to index
    y_data.append([char_dic[c] for c in y_str]) # y str to index
    
  x_one_hot = [np.eye(dic_size)[x] for x in x_data]
  
  # transform as torch tensor variable
  X = torch.FloatTensor(x_one_hot)
  Y = torch.LongTensor(y_data)
  ```  
  
- Vanilla RNN의 형태  
  ![pytorch04-4](https://user-images.githubusercontent.com/43376853/94104000-e5000f00-fe70-11ea-9efa-b1bf1bea4f6c.png)  
  - 하지만 모델이 Underfitting 된다든지, 그런 경우에, 좀 더 Complex한 모델을 만들고 싶을 수 있음  
  
- 예) Fully Connected Layer 추가 + RNN Stacking  
  ![pytorch04-5](https://user-images.githubusercontent.com/43376853/94104080-21cc0600-fe71-11ea-9516-704bf6889744.png)  
  
  ```  
  class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
      super(Net, self).__init__()
      self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers = layers, batch_First = True)
      self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
      
    def forward():
    x, _status = self.rnn(x)
    x = self.fc(x)
    return x
    
  net = Net(dic_size, hidden_size, 2) # RNN Layer 2개로 쌓겠다. 
  
  # Loss & Optimizer 설정   
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam((net.parameters(), learning_rate)
  
  # 훈련 시작
  for i in range(100):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()
    
    # 모델이 예측한 결과물 해석 
    results = outputs.argmax(dim=2)
    predict_str = ""
    for j, result in enumerate(results):
      for j, result in enumerate(results):
        print(i, j, ''.join([char_set for t in result]), loss.item())
        if j == 0:
          predict_str += ''.join([char_set[t] for t in result])
        else:
          predict_str += char_set[result[:-1]]
  ```  
  
  

---  

#### References  
[파이토치로 시작하는 딥러닝 기초 Part4](https://www.edwith.org/boostcourse-dl-pytorch/joinLectures/24018)      



  
  
  

