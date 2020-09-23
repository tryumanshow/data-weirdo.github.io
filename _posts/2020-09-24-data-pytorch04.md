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
  
  
  

