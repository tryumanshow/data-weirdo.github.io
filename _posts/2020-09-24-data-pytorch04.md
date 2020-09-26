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

## Lab11-4. RNN Timeseries  
- Timeseries Data?  
  - 시계열 데이터: 일정한 시간 간격으로 배치된 데이터  
  - ex. 주가 차트  
  
- 예  
  - 구글의 일별 주가 데이터 (일별 시가, 고가, 저가, 종가, 거래량에 대한 데이터)  
    ![pytorch04-6](https://user-images.githubusercontent.com/43376853/94340060-9f427280-0039-11eb-9c9c-11d900bdb45c.png)  
  - '7일 간의 데이터를 입력받아서, 8일차의 종가 예측해보자!'  
    (사실 주식 시장에서 7일 간의 데이터를 ㅂ고 데이터를 예측하겠다는 건 역설이기는 함)  
  - 문제 정의: Many-to-One Structure  
  
  - 8일차 Vector의 Dim=1일 것 (종가 예측이기 때문)  
    - 문제는 Hidden State들의 Dim도 1이 되어야 된다는 건데, 이는 모델로서 굉장히 부담스러운 일  
      (이는 곧, 시가, 고가, 저가, 종가, 거래량 다섯개 데이털르 합쳐서 하나의 값으로 압축시켜야 하는 일이기 때문에)  
      
  - 그래서, 8일차의 종가 예측을 위해서 Hidden State가 가진 Dimension 만큼 (이를 테면 10) 의 값을 최종 output으로 받고, 이 벡터에 Fully Connected Layer를 연결을 해서, 그 Layer의 Output이 종가를 맞추도록 하는 것이 일반적.  
    
- 코드  

  ```  
  import torch
  import torch.optim as optim 
  import numpy as np
  import matplotlib.pyplot as plt
  
  # Random seed 설정  
  torch.manual_seed(0)
  
  seq_length = 7
  data_dim = 5 # 시가, 고가, 저가, 종가, 거래량
  hidden_dim = 10 # 내가 임의로 설정  
  output_dim = 1 # Fully connected Layer가 맞추어야 할, 종가의 Dim  
  learning_rate = 0.01
  iterations = 500
  
  xy = np.loadtxt('data-02-stock_daily.csv', delimeter=',')
  xy = xy[::-1] # 순서를 역순으로 만든다.  
  
  train_size = int(len(xy) * 0.7)
  train_set = xy[0:train_size]
  test_set = xy[train_size - seq_length: ] 
  
  # Scaling! ∵ 주가는 800선, 거래량은 1000000선 -> 이격이 너무 크다. 
  # 모델은 1000000이 '숫자'로서 더 크다는 점을 인지를 함  
  # 따라서 모두 [0,1] range의 값으로 바꾸어준다.  
  train_set = minmax_scaler(train_set) 
  test_set = minmax_scaler(test_set)
  
  trainX, trainY = build_dataset(train_set, seq_length)
  testX, testY = build_dataset(test_set, seq_length)
  
  trainX_tensor = torch.FloatTensor(trainX)
  trainY_tensor = torch.FloatTensor(trainY)
  
  testX_tensor = torch.FloatTensor(testX)
  testY_tensor = torch.FloatTensor(testY)
  ```  
  
  - About minmax_scaler & build_dataset  
  
    ```  
    def minmax_scaler(data):
      numerator = data - np.min(data, 0)
      denominator = np.mad(data, 0) - np.min(data, 0)
      return numerator / (denominator + 1e-7)  
      
    def build_dataset(time_series, seq_length):
      dataX = []
      dataY = []
      for i in range(0, len(time_series) - seq_length):
        _x = time_series[i: i+seq_length, :]
        _y = time_series[i + seq_length, [-1]]
        print(_x, '->', _y)
        dataX.append(_x)
        dataY.append(_y)
      return np.array(dataX), np.array(dataY)
    ```  
    
  - NN 선언하기  
    
    ```  
    class Net(torch.nn.Module):
      def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        
      def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])
        return x
        
    net = Net(data_dim, hidden_dim, output_dim, 1)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    for i in range(iterations):
    
      optimizer.zero_grad()
      outputs = net(trainX_tensor)
      loss = criterion(outputs, trainY_tensor)
      loss.backward()
      optimizer.step()
      print(i, loss.item())
      
    plt.plot(testY)
    plt.plot(net(testX_tensor).data.numpy())
    plt.legend(['original', 'prediction'])
    plt.show()
    ```  
    
    ![pytorch04-7](https://user-images.githubusercontent.com/43376853/94340447-c9496400-003c-11eb-9f11-02f9a102cfd5.png)  
    
- 결과를 보면 굉장히 Prediction을 잘하는 것처럼 보이는데, 투자 모델 만들기 쉬워보이네?  
  - NONO, 주식시장은 변동을 일으키는 변수들이 정말 너무 많아서, 시가, 종가, 고가, 저가, 거래량 다섯 개의 피쳐만으로는 전체 주식시장의 예측이 정말 어려움  
  - 더많은 Feature를 고려할 것!  
  - 늘, 그렇게 성능이 좋지는 않아서, 심지어는 뉴스를 긁어와서, 감성분석을 한다거나 하는 등의 방법론을 통해 Feature들을 추가하는 사람들도 존재.  
  - Feature가 많아야 Robustness를 달성할 수 있을 (지도 모른다.)  
  
---  

## Lab11-5. RNN seq2seq  
- Sequence를 입력받아서 Sequence를 출력하는 모델  
  ![pytorch04-8](https://user-images.githubusercontent.com/43376853/94340658-88eae580-003e-11eb-838a-09d539c6caa3.png)  

- Seq2Seq  
  - 번역이나 Chatbot에서 잘 사용됨  
  - Encoder-Decoder 구조!  
    ![pytorch04-9](https://user-images.githubusercontent.com/43376853/94340743-f565e480-003e-11eb-9740-9932b8691fff.png)  
    - Encoder  
      - 입력된 Sequence를 벡터의 형태로 압축  
      - 이 압축된 형태를 Decoder에 전달  
    - Decoder  
      - Encoder에서 만들어서 넘겨준 벡터를 첫 Cell의 Hidden State로 넘겨줌  
      - 문장이 시작한다는 \<Start\> Flag와 함께 Decoder 활용을 시작  
      
- Code  
  
  ```  
  import random 
  import torch
  import torch.nn as nn
  import torch.optim as optim
  
  ...
  
  SOURCE_MAX_LENGTH = 10
  TARGET_MAX_LENGTH = 12
  # raw: 원문, Source text 및 Target text를 구성하는 문장의 최대 길이를 제한  
  load_pairs, load_source_vocab, load_target_vocab = preprocess(raw, SOURCE_MAX_LENGTH, TARGET_MAX_LENGTH)
  print(random.choice(load_pairs))
  
  enc_hidden_size = 16
  dec_hidden_size = enc_hidden_size
  enc = Encoder(load_source_vovab.n_covab, enc_hidden_size).to(device)
  dec = Decoder(dec_hidden_size, load_target_vocab.n_vocab).to(device)
  
  train(load_pairs, load_source_vocab, load_target_vocab, enc, dec, 5000, print_every=1000)
  evaluate(load_paris, load_source_vocab, load_target_vocab, enc, dec, TARGET_MAX_LENGTH)
  ```       

  ▲ 번역 Task를 시행하는 Seq2Seq 모델  
  - 번역: 번역의 원본이 되는 Source Text를 대상으로 함  
  
  
- Data Processing 단게  
  
  ```  
  import random
  import torch
  import torch.nn as nn
  import torch.optim as optim
  
  torch.manual_seed(0)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  raw = ["I feel hungry.   나는 배가 고프다.",
         "Pytorch is very easy.   파이토치는 매우 쉽다.",
         "Pytorch is a framework for deep learning.   파이토치는 딥러닝을 위한 프레임워크이다.",
         "Pytorch is very clear to use.   파이토치는 사용하기 매우 직관적이다."]
         
  SOS_token = 0 # Start of Sentence
  EOS_token = 1 # End of Sentence 
  ```  

  ```  
  # Source / Target Text를 나눠서 어떤 단어로 구성되어있고, 단어는 몇 개인지 측정  
  def preprocess(corpus, source_max_length, target_max_length):
    print('reading corpus...')
    pairs = []
    for line in corpus:
      pairs.append([s for s in line.strip().lower().split('\t')])
    print('Read {} sentence pairs'.format(len(pairs))
    
    pairs = [pair for pair in paris if filter_pair(pair, source_max_length, target_max_length)]
    print('Trimmed to {} sentence pairs'.format(len(pairs))

    source_vocab = Vocab() # Vocab이라는 클래스 선언 (이 클래스 내에 단어의 개수나 딕셔너리 같은 것들 넣어줌)   
    target_vocab = Vocab()
    
    print('Counting words...')
    for pair in pairs:
      source_vocab.add_vocab(pair[0])
      target_vocab.add_vocab(pair[1])
    print('source vocab size =', source_vocab.n_vocab)
    print('target vocab size =', target_vocab.n_vocab)
    
    return pairs, source_vocab, target_vocab
  ```  
  
- Encoder  
  
  ```  
  class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
      super(Encoder, self).__init__()
      self.hidden_size = hidden_size
      self.embedding = nn.Embedding(input_size, hidden_size)
      self.gru = nn.GRU(hidden_size, hidden_size)
      
    def forward(self, x, hidden):
      x = self.embedding(x).view(1, 1, -1)
      x, hidden = self.gru(x, hidden)
      return x, hidden
  ```  
  
- Decoder  

  ```  
  class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
      super(Decoder, self).__init__()
      self.hidden_size = hidden_size
      self.embedding = nn.Embedding(output_size, hidden_size)
      self.gru = nn.GRU(hidden_size, hidden_size)
      self.out = nn.Linear(hidden_size, output_size)
      self.softmax = nn.LogSoftmax(dim=1)
      
    def forward(self, x, hidden):
      x = self.embedding(x).view(1, 1, -1)
      x, hidden = self.gru(x, hidden)
      x = self.softmax(self.out(x[0]))
      return x, hidden
  ```  
  
- 학습 함수  
  
  ```  
  # Sentence를 입력으로 받아서, sentence를 one-hot encoding으로 바꾸고, 최종적으로 Tensor의 형태로!
  def tensorize(vocab, sentence):
    indexes = [vocab.vocab2index[word] for word in sentence.split(' ')]
    indexes.append(vocab.vocab2index['<EOS>'])
    return torch.Tensor(indexes).long().to(device).view(-1,1)
    
  def train(pairs, source_vocab, target_vocab, encoder, decoder, n_iter, print_every=1000, learning_rate=0.01):
    loss_total = 0
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    training_batch = [random.choise(pairs) for _ in range(n_iter)]
    training_source = [tensorize(source_vocab, pair[0]) for pair in training_batch]
    training_target = [tensorize(target_vocab, pair[1]) for pair in training_batch]
    
    criterion = nn.NLLLoss() # Negative Loglikelihood 
  ```  
  
- Training  

  ```  
  def train(pairs, source_vocab, target_vocab, encoder, decoder, n_iter, print_every=1000, learning_rate=0.01):
  
  for i in range(1, n_iter+1):
    source_tensor = training_source[i-1]
    target_tensor = training_target[i-1]
    
    encoder_hidden = torch.zeros([1, 1, encoder.hidden_size]).to(device)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    source_length = source_tensor.size(0)
    target_length = target_tensor.size(0)
    
    loss = 0
    
    for enc_input in range(source_length):
      _, encoder_hidden = encoder(source_tensor[enc_input], encoder_hidden)
      
      decoder_input = torch.Tensor([[SOS_token]]).long().to(device)
      decoder_hidden = encoder_hidden
      
      for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di] # teacher forcing  
        
      loss.backward()
      
      encoder_optimizer.step()
      decoder_optimizer.step()
      
      loss_iter = loss.item() / target_length
      loss_total += loss_iter
      
      if i % print_every == 0:
        loss_avg = loss_total / print_every
        loss_total = 0
        print("[{} - {}%] loss = {:05.4f}".format(i, i/n_iter*100, loss_avg))

  ```  
    
---  

## Lab11-6. PackedSequence  
- 길이가 각각 다른 Sequence data를 하나의 Batch로 묶는 두 가지 방법  


  

---  

#### References  
[파이토치로 시작하는 딥러닝 기초 Part4](https://www.edwith.org/boostcourse-dl-pytorch/joinLectures/24018)      



  
  
  

