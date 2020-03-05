---
layout: post
title:  "Sample-based Learning Methods - Week3"
subtitle:  "SBLM_Week3"
categories: ml
tags: rl
comments: true

---

- Temporal Difference Learning에 대한 소개, Monte-Carlo 및 Dynamic Programming 방법과의 비교에 관한 글입니다.  

---  


# 2. Sample-based Learning Methods - Week3  

## 3.1 Introduction to Temporal Difference Learning  
### 3.1.1 What is Temporal Difference (TD) Learning?  

- 교재의 내용을 빌리자면,  
  - 강화학습에 있어서 가장 중심이 되고 참신한 아이디어 중 하나를 꼽으라고 한다면 `Temporal Difference Learning`이다.  
  
- Incremental update in Monte-Carlo policy evaluation  
  ![](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cleftarrow%20V%28S_t%29%20&plus;%20%5Calpha%5BG_t%20-%20V%28S_t%29%5D)  
  - 상수 step size를 사용하고 있음에 유의  
  - incremental rule을 사용하면, 일련의 return 리스트를 저장하지 않고도 Monte-Carlo estimate를 만들어낼 수 있음.  
  - 믈론, return (G_t)의 계산을 위해서는 `Full Trajectory` 샘플들이 필요하기는 함.  
    - episode가 끝나기 전에는 학습을 할 수가 없다!!  
    
- __episode가 끝나지 않더라도 incremental하게 학습을 할 수는 없을까?__  
  ![](https://latex.codecogs.com/gif.latex?v_%7B%5Cpi%7D%28s%29%20%5Cdoteq%20E_%7B%5Cpi%7D%5BG_t%7CS_t%3Ds%5D)  
  ![](https://latex.codecogs.com/gif.latex?%3D%20E_%7B%5Cpi%7D%5BR_%7Bt&plus;1%7D&plus;%5Cgamma%20G_%7Bt&plus;1%7D%7CS_t%3Ds%5D)  
  ![](https://latex.codecogs.com/gif.latex?%3D%20R_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20v_%7B%5Cpi%7D%28S_%7Bt&plus;1%7D%29)  
  
  - 위 수식을 염두에두고, incremental update로 다시 돌아가면,  
  ![](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cleftarrow%20V%28S_t%29%20&plus;%20%5Calpha%5BG_t%20-%20V%28S_t%29%5D)  
  ![](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cleftarrow%20V%28S_t%29&plus;%20%5Calpha%5BR_%7Bt&plus;1%7D&plus;%5Cgamma%20V%28S_%7Bt&plus;1%7D%29%20-%20V%28S_t%29%5D)  
    - Return 대신에 next-state의 value값을 알면 되니까, episode가 끝날 때까지 기다려야할 필요는 없음.  
    - 물론 다음 step까지는 기다려야하지만.  
  
  - __notation__  
  ![](https://latex.codecogs.com/gif.latex?R_%7Bt&plus;1%7D&plus;%5Cgamma%20V%28S_%7Bt&plus;1%7D%29)를 `T-target`이라고 부를 것.  
  ![](https://latex.codecogs.com/gif.latex?R_%7Bt&plus;1%7D&plus;%5Cgamma%20V%28S_%7Bt&plus;1%7D%29%20-%20V%28S_t%29)를 `TD error`라고 부를 것  
    - ▼ TD error를 다음과 같이 표기하기도 함.  
      ![](https://latex.codecogs.com/gif.latex?%5Cdelta_t)  
      
  - `T target`이라는 것이 처음에는 조금 이상해 보일 수 있음  
    - 하지만, V(S_{t+1}) 이 값을 업데이터 해감에 따라,  
      ![](https://latex.codecogs.com/gif.latex?E_%7B%5Cpi%7D%5BG%7CS%3Ds_%7Bt&plus;1%7D%5D)로 분명 다가갈 것  
      
      ![](http://drive.google.com/uc?export=view&id=1_1RT8b_QqnTvHUKQTJ1BqIqYdPUeP8l1)  
      

### 3.1.2 Rich Sutton: The Importance of TD Learning  
- __TD learning은 prediction learning에 특화된 학습 방법__ 이다!  
  - 이 사실이 TD learning을 현 세기 AI에서 가장 중요한 것으로 만든다.  
  - 현 시대 연산 속도의 발전도 한 몫했다.  
    - 연산의 발전으로부터 가장 큰 수혜를 입은 부분: `prediction learning`  
  
- `TD learning is learning a prediction from another, later, learned prediction`  
  - learning a guess from a guess  
  - The TD error is the difference between the two predictors, the temporal difference  
  - Otherwise TD learning is the same as supervised learning backpropagating the error  
  
- 지도학습  
  : 정답이 뭔지를 말해주는 supervisor의 존재.  
  - 하지만 현실에서는 무엇을 해야될지 말해주는 supervisor나 instructor는 없다.  
  
- 혹자는 prediction learning을 지도학습이라고 혹은 비지도학습이라고 표현할 수 있음.   
  - 지도학습: 기다리면 결과가 나오고, 그 결과가 instructing하고 있는 것   
  - 비지도학습: supervisor 없이 기다리고 결과를 얻은 것이니까.  
  
- 계속 반복 중인 명제: `Temporal difference learning is a method that's speacialized for prediction learning for multi-step prediction learning`  
  - 지도학습에 대해 한 가지 더 지적  
    - 지도학습을 prediction learning인 것처럼 묘사하는 말들을 많이 들었을텐데, 그게 아니다.  
      - 정답이 무엇인지 주어지고 prediction을 할 때라야 prediction learning이 되는 것  
      - 지도학습이라고 하면, 종종 training set 그 자체를 의미하거나, 아니면 prediction learning을 사용하더라도 `한 단계`를 predict하는 것일 뿐.  
      - `하지만, 대부분의 prediction은 one-step prediction이 아니다.`  
      
- Temporal Difference Model  
  - 뇌의 reward system의 standard model.  
    - 뇌의 signal: TD error와 상응  
    - 도파민: TD error를 싣고 나르는 역할  
  - 뭐 결론은 TD Model은 AI와, 심리학 및 신경과학에 있어서 중요한 토픽  
    - 이와 관련한 모든 문제는 `prediction learning`  
    
- 한마디로 요약  
  ```  
  - Predition Learning은 AI 시대의 key  
  - TD Learning은 Prediction Learning에 특화된 학습 방법  
  ```  
  
## 3.2 Advantages of TD  
### 3.2.1 The advantages of temporal difference learning  
- TD learning: Dynamic Programming과 Monte Carlo의 combination  
  - Dynamic Programming의 측면  
    : Bootstraping을 한다.  
  - Monte Carlo의 측면  
    : Experience로 부터 직접 학습.  

  
- 한 예시로 시작  
  - 매일, 집으로 운전해서 가는 데에 얼마나 걸리는지 예측  
  - 시간, 요일, 날씨, 이외 다른 요인들에 근거 해 예측을 할 것  
  - 집으로 운전해서 간 적 많기 때문에, 이제 대충, 맞닥뜨릴 수 있는 다양한 상황들에 대해 꽤나 정학한 estimate를 할 수 있는 상태라고 하자.  
    
    ```  
    - 어느날 저녁, 사무실을 나와서, 집으로 가는 데에 보통 30분 소요  
    - 약 5분 만에 주차장을 나와서, 비가 온다는 걸 인지 → 비오는 날엔 차가 더 막힘 
      → 집 가는 데에 한 35분 정도 걸리겠구나 하는 생각 (estimate)   
    - 근데, 예상했던 것보다 고속도로를 더 빨리 빠져나옴 → 이제 집에 한 15분 정도 안에 집 가겠는데? 라는 생각 
    - 근데, 고속도로 빠져나오고 난 뒤에 앞에 느려터진 트럭이 있음. → 집 가는 데에 10분 더 걸리겠다 라는 생각  
    - 그 후, 보통 집까지 3분 정도 걸리는 home street로 접어들고, 실제로 3분 뒤에 집 도착!  
    ```  
    
    ▼ 위 상황 요약  
    ![](http://drive.google.com/uc?export=view&id=1PZ1IMKjSSg1FzKaEGHcYNvxYkZP41Z1Q)  
    - 원 안의 숫자는 remaining driving time에 대한 prediction  
    - __어떻게 이 prediction들을 개선시킬 수 있을까?__   
    
- __어떻게 이 prediction들을 개선시킬 수 있을까?__   
  - reward를 구체화하는 것이 필요.  
    ![](http://drive.google.com/uc?export=view&id=1TvU8h9yiEosrcwZR91A3MOyLDKTm5kxC)  
    
    - 이제 Monte Carlo 방법을 한 번 보자.  
      ![](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cleftarrow%20V%28S_t%29%20&plus;%20%5Calpha%5BG_t%20-%20V%28S_t%29%5D)  
      (α=1이고 γ=1인 Monte Carlo Method를 생각해보겠음.)  
      - (episode가 끝나야 업데이트가 가능했듯) 집에 도착할 때라야 각 state들을 추정할 수 있음.  
        - 사무실을 떠났을 때부터 생각해보자.  
          ![](https://latex.codecogs.com/gif.latex?V%28leave%29%20%5Cleftarrow%20V%28leave%29%20&plus;%20%5Calpha%5BG_0%20-%20V%28leave%29%5D)  
          ![](https://latex.codecogs.com/gif.latex?G_0%20%3D%205%20&plus;%2015%20&plus;%2010%20&plus;%2010%20&plus;%203%20%3D%2043)  
          - G0 = 43, V(leave) = 30  
          
        - 주차장을 나왔을 때  
          ![](https://latex.codecogs.com/gif.latex?V%28exit%29%20%5Cleftarrow%20V%28exit%29%20&plus;%20%5Calpha%5BG_t%20-%20V%28exit%29%5D)   
          ![](https://latex.codecogs.com/gif.latex?V%28exit%29%20%5Cleftarrow%20V%28exit%29%20&plus;%20%5Calpha%5BG_t%20-%20V%28exit%29%5D)  
          - V(exit) = 35, G_1 = 38  
          
        - In the same way  
        - 계속 강조했듯, 학습이 일어나기 위해선 episode가 끝날 때까지 기다려야 한다.  
        
  - __이 문제에 TD algo를 적용하면 어떨까?__  
    ![](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cleftarrow%20V%28S_t%29&plus;%20%5Calpha%5BR_%7Bt&plus;1%7D&plus;%5Cgamma%20V%28S_%7Bt&plus;1%7D%29%20-%20V%28S_t%29%5D)  
    - 사무실을 떠난 경우에 TD algo를 적용  
      ![](https://latex.codecogs.com/gif.latex?V%28leave%29%20%5Cleftarrow%20V%28leave%29%20&plus;%20%5Calpha%20%5BR_1&plus;%5Cgamma%20V%28exit%29%20-%20V%28leave%29%5D)  
      - V(leave)=30, V(exit)=35, R1=5  
      
    - 주차장을 나왔을 때 TD algo를 적용  
      ![](https://latex.codecogs.com/gif.latex?V%28exit%29%20%5Cleftarrow%20V%28exit%29%20&plus;%20%5Calpha%5BR_%7B2%7D%20&plus;%20%5Cgamma%20V%28S_%7Bexit%20%5C%3B%20highway%7D%29%20-%20V%28exit%29%5D)  
      - V(exit)=35, V(exit highway)=15, R2=15  
      - 연산 결과 V(exit) = 30이 됨 (Monte-Carlo 적용했을 때보다 estimated time이 줄었음.)   
        : 고속도로를 예상했던 것보다 빨리 빠져나왔기에 make sense!  
        
      - In the same way  
      - 집에 도착하기를 기다릴 필요가 없이 `online learning`이 가능해진다.  
      
- TD method의 이점 정리  
  - environment에 대한 model이 필요 없다.  
  - experience로 부터 직접 학습할 수 있다.  
  - (episode마다 value를 업데이트를 하는 Monte Carlo와는 다르게) 매 step마다 업데이트 가능  
  - `estimate에 근거하여 estimate를 업데이트 할 수 있음.` (개인적으로 가장 솔깃한 맥락)   
  - correct prediction에 `asymptotically converge`  
  - Monte Carlo보다 보통 더 빨리 수렴함.  
  
### 3.2.2 Comparing TD and Monte Carlo  
  
- 역시 예로부터 시작!  
  - Random Walk  
  ![](http://drive.google.com/uc?export=view&id=1Q7DMXIdVZIRGy0blQF_k3plTeC7UYyby)  
  
  - 5개의 state모두 non-terminal states  
  - 각 state마다 2 deterministic actions (왼쪽, 오른쪽)  
  - uniform random policy   
  - 매 episode는 C로부터 시작하고, 맨 왼쪽 혹은 맨 오른쪽에서 terminate.  
  - reward: 오른쪽에서 terminate 시: 1, 나머지는 다 0  
  - discount factor 0가정  
  
  - 이 경우, 각 state의 value는 intuitive한 의미를 지님 
    : 각 state로부터 시작해서 오른쪽에서 끝날 학률  
  
  ▼ True value labeled MDP  
    ![](http://drive.google.com/uc?export=view&id=1j44VHfR3q9BacHtHEjsfcbC219TCquSr)  
    ![](http://drive.google.com/uc?export=view&id=1UU3gwNvG_mG2OYMkT47Tu4D0EYZmZMBV)  
  
  ▼ TD Agent, Monte-Carlo agent로 하여금 각각 value function을 추정(`estimate`)하도록 해볼 것!  
  
  - 먼저 두 agent들 모두에게 (approximate) value function을 0.5로 초기화  
    ![](http://drive.google.com/uc?export=view&id=1zyef7CODETcwcWXqJndJIs2MetoSqvg5)  
    
  ▼ First Episode 이후의 결과  
    ![](http://drive.google.com/uc?export=view&id=18gisE8NOOyQxhLuhbJxLBkQf80P5xfsU)  
    - TD Learning에서는 state E`만` update   
    
      - C에서 D로의 transition을 보면 이해해볼 수 있음.  
        ![](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cleftarrow%20V%28S_t%29%20&plus;%20%5Calpha%20%5BR_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20V%28S_%7Bt&plus;1%7D%29%20-%20V%28S_t%29%5D)  
    - Monte Carlo에서는 C, D, E, 그러니까 agent가 episode 진행 동안 본 `모든` state를 업데이트  
      ![](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cleftarrow%20V%28S_t%29%20&plus;%20%5Calpha%20%5BG_t%20-%20V%28S_t%29%5D)  
      
  ▼ 2nd episode  
    - 좀 분잡한 episode 가정  
      - C → D → E → D → C → D → C → B → A → B → C → D ...  
        - TD Learning에서는 매 step마다 value에 대한 업데이트가 일어나지만,  
        - Monte-Carlo에서는 아직 episode가 끝나지 않았기 때문에 업데이트가 하나도 일어나지 않음.  
        
  - Plotting  
    ![](http://drive.google.com/uc?export=view&id=1ZFBGG-ADuF76VgchI5bnRo6i2yzxTLvX)  
    - episode의 수가 늘어날 수록 점점 더 true value에 가까워지고 있네~  
    
    - α = .1: 최근 episode의 결과에 따라 많이 요동친다는 말  
    - α 값을 더 줄이면 (smaller learning rate), 더 나은 estimate을 얻을 것  
    
- __TD가 Monte Carlo보다 빠른가?__  
  - 똑같은 예제에 대해  
    ![](http://drive.google.com/uc?export=view&id=1GbwgBRBtmrYXuNddOP412j08N-d4sL8S)  
  - 답: __그렇다__  
  
  - 단, TD에서, learning rate가 0.15면 0.05일 때보다 error는 더 빨리 줄어들지만, 최종 결과는 더 낮은 학습률일 때가 더 좋더라.  
       
### 3.2.3 Week 2 Summary  
- TD algo.  
  - Dynamic programming과 Monte Carlo의 좋은 점을 섞었다!  
  
  - 'TD는 guess로부터 guess를 update한다.'   
  - value estimate을 다음 state의 value를 향해 bootstrapping!  
    
- Tabular TD(0) algo  
  - 에피소드의 각 `step`들마다 value estimate을 update!  
  - 한 episode가 끝날 때까지 기다릴 필요가 없다.  단지 이전 state를 기억하고 있으면 될 뿐.  
 
- Monte Carlo보다 수렴 속도가 빠르다.  
- Dynamic Programming과는 다르게 model이 필요가 없다.  
- `online` & `fully incremental` (Monte Carlo와 Dynamic Programming은 그렇지 못함)  
