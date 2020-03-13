---
layout: post
title:  "Sample-based Learning Methods - Week5"
subtitle:  "SBLM_Week5"
categories: ml
tags: rl
comments: true

---

- 작성 중  

---  


# 2. Sample-based Learning Methods - Week5  
## 5.1 What is a Model?  
### 5.1.1 What is a Model?  
- 인생의 많은 결정들: 그것에 대해 너무 많은 생각을 하지 않고 결정을 하거나, 혹은 가능한 많은 시나리오들에 대한 엄청난 숙고를 하거나  
  - TD: 샘플링된 experience로부터만 학습  
  - DP: 완전한 정보를 사용하여 계획을 짬  
  
  - 위 양 극단을 적절히 Leverage 할 수 있는 방법은 없을까? (Intermediate)   
  
- Model: dynamics에 대한 지식을 저장하기 위해 사용  
  ![](http://drive.google.com/uc?export=view&id=1evietjpJQnIqUH3JviOpROceveMYf5FV)  
  - 특정 state와 action으로부터, 모델은 가능한 next state와 reward를 생산해내야.  
    - 실제로 행동을 취하지 않고도 action의 결과를 볼 수 있도록 함  
  - Model은 Planning을 고려한다(allow for)  
    - __Planning__  
      : policy를 improve 하기 위해 모델을 사용하는 과정을 이름  
      
      ```  
      * 모델로 plan하는 방법  
      - Model을 통해 Simulated Experience를 create한다.   
      - Simulated Experience로 Value function을 update한다.
      - Value function을 바탕으로 Policy를 modify 한다.  
      ```  
      
- 우리에게 유용할 수 있는 Model의 종류?  
  - a. Sample Model  
    - 어떠한 확률 아래에서 추출된 실제 결과를 생성  
    - 연산이 그다지 값비싸지 않음.  
    - ex) 동전 던지기 sample model → 앞, 뒤 면의 랜덤 시퀀스를 생성  
  - b. Distrbution Model  
    - 모든 결과의 likelihood나 확률값을 완전히(completely) 명시  
    - 더 많은 정보를 포함할 수는 있지만, 명시하기가 굉장히 어려울 수 있고, 매우 방대해질 수 있음.  
    - ex) 동전 던지기에서 동전 한 번 던지면 앞, 뒤 나올 확률이 각각 50%  
      - 어떤 sequence든지 간에 그 확률값을 명시할 수가 있음.  
      - 5개 coin이라면 32개의 각 시퀀스에 32개의 확률값을 부여할 수 있음  
  
  - Distribution Model은 각 결과의 명시적 확류랎에 따라 결과를 뽑는다면 Sample Model처럼 사용될 수 있음  

### 5.1.2 Comparing Sample and Distribution Models  
- 12개의 주사위를 굴리는 문제를 예로 들어보자.  
  - __sample model__  
    - 주사위 하나를 열 두번 굴리는 것: `sample model`의 예  
    - joint probability 필요x  
    - 한 주사위에 6개의 결과. 주사위가 2개면 결과 36개, 3개면 216개, 4개면 1296개... 던지는 수 늘어나면 노답  
  
  - __distribution model__  
    - 결과에 대한 정확한 롹률을 도출  
    - 기대 평균이나 분산을 수치화할 수 있음  
    

## 5.2 Planning  
### 5.2.1 Random Tabular Q-Planning  
- model experience와의 `process planning`?  
  : 세계와 상호작용 하지 않고 더 나은 의사결정을 하기 위해 모델을 어떻게 `leverage`할지에 대한 것  
  
- __Planning__  
  - 모델을 input으로 받아서 더 나은 policy를 구하는 과정!  
    ![](http://drive.google.com/uc?export=view&id=1N2CH_ToAMpwYRV9qbddV8tHl_1MIiMH7)  
    
  - Possible Planning Approach1  
    - 모델로부터 sample experience 샘플링  
      - 세계가 어떻게 돌아가는지에 대한 나의 이해를 바탕으로, 세상에서 일어날 법한 가능한 시나리오를 상상하는 것과 같음  
      - 이렇게 생성된 experience를 바탕으로 value function을 update (마치 그런 상호작용이 현실에서 일어난 것처럼)  
      - 위에서 얻어진 improved value를 바탕으로 greedy하게 행동하면 improved policy를 생성!  
    
      - (Recall) Q-Learning 또한 그러했다.  
        - environment로부터의 experience를 사용  
        - policy improvement를 위해 update 이행  
      
      - __Q-Planning__  
        - `model`로부터의 `experience`를 사용, policy improvement위해 위와 유사한 update 실행  
          ▲ `Random-sample one-step tabular Q-planning`가 위 아이디어 이용  
        
- __Random-sample one-step tabular Q-Planning__  
  - 가정1: 'Transition dynamics의 sample model을 갖고 있다.'  
  - 가정2: '관련된 state-action pair들을 샘플링하는 strategy를 갖고 있다.'  
  
  - Process  
    - Step1: Sampling    
      - 모든 state, action 집합에서 state-action pair를 랜덤하게 선택  
      - 이 state-pair pair로 sample model을 써서 next state, reward sample 생성  
    - Step2: Q-Learning update  
    - Step3: Greedy policy improvement   
      
      ![](http://drive.google.com/uc?export=view&id=17SuNi5MA_QqBaQY4VDFSsAAGSSBbkQ66)  
      
  - __Key Point__   
    - 이 planning method는 상상의(imagined) 혹은 시뮬레이트된(simulated) `experience`만을 사용  
    - 현실세계에서 실제 행동하지 않고, 혹은 interaction loop와 병행하여 사용 가능  
      - 현실에서는 어떤 action은 특정 시간에만 일어나는데, learning update는 꽤나 발리 일어난다고 해보자.  
      - 이 때는 Learning Update를 한 후, 다음 action이 취해질 때까지 기다려야 하는 상황을 초래  
      - 이 waiting time을 `Planning Update`를 통해 채울 수 있음.  
        ![](http://drive.google.com/uc?export=view&id=1TTIxR4ChtzeIXRP-N2tQiVT8rzHDjRKO)  
        
    - 예를 들어, cliff 부근에 서있는 로봇이 있다고 하자.  
    - 로봇이 가진 model이 cliff 아래로 떨어지는 것에 대한 결과를 알고 있다고 하더라도, 이는 value function이나 policy에 정확하게 반영되지는 않음.  
    - cliff 아래로 떨어지는 것에 대한 simulated experience를 형성하고, 이 transition에 대해 많은 planning step을 실행하면, cliff 아래로 떨어지는 것이 나쁘다는 것을 value function은 더 잘 반영하면서 policy는 cliff로부터 멀어지도록 할 것   
      
        
## 5.3 Dyna as a formalism for planning  
### 5.3.1 The Dyna Architecture  
- cf) Direct RL  

  ```  
  - Direct RL: Direct RL ﬁnds the optimal policy by directly maximizing the state value function for ∀s ∈S. 
  - Indirect RL:  Indirect RL ﬁnds the optimal policy by solving the Bellman’s optimality equation for ∀s ∈S.  
  ```  
  
- Q-Learning  
  : environment experience 사용 → update  
- Q-Planning  
  : model로부터 simulating 된 experience 사용 (model experience 사용) → update  
  
- __Dyna architecture__  
  : Q-Learning + Q-Planning  
  
    ![](http://drive.google.com/uc?export=view&id=1rDVAdJ5SjTp7Iot0I58qhMx9CxVcoylf)  
    - (평범한) environment와 policy → experience 흐름 형성   
    - 이 experience로부터 direct RL update 실행  
    
    
    - Q: Planning을 하려면 모델이 필요한데..? 이 모델: environment experience로부터 만듦  
    - 이 model로부터 model experience를 만들어낼 것  
      - 이 모델이 어떻게 simulated experience를 만드는지, agent가 어떤 state로부터 plan할지 control하고자 함  
        ▲ `Search Control`  
    - Planning update: model에 의해 생성된 experience를 이용하여 이루어짐.  
    
- ex)  
  ![](http://drive.google.com/uc?export=view&id=1Ppr51VkF_KOIZmJB6kUT955GufVYpf8P)  
  - reward: terminal state 도착시 +1, 나머지 +0  
  
  - 처음에는 진짜 랜덤하게 이곳저곳 막~ 돌아다니다가 결국에 한 번 Terminal state에 다다를 것  
    - One direct RL update, using `Q-Learning`  
      ![](https://latex.codecogs.com/gif.latex?Q%28s%2Ca%29%20%5Cleftarrow%20Q%28s%2Ca%29%20&plus;%20%5Calpha%20%28r%20&plus;%20%5Cgamma%20max_a%20Q%28s%27%2Ca%27%29%20-%20Q%28s%2Ca%29%29)  
      - 파란색으로 표시된 곳, 단 한 곳만 aciton value를 update  
        (0이 아닌 reward를 generate한 유일한 transition)  
        
    - `Dyna`는 environment의 model을 학습하기 위해 1st episode 동안의 모든 experience를 사용!  
      - 노란색으로 표시된 곳: 첫 episode 동안 방문한 모든 state  
      - 로봇이 모든 것을 방문한 것은 아니지만, 대부분은 방문  
      - __Dyna performs planning on every time step.__  
      - 한편, planning은 1st episode 동안에는 policy에 아무 영향도 끼치지 못함.  
        1st episode가 끝난 후 그 힘을 발휘. (로봇이 지나온 그 노란 패스가 로봇이 생각하는 grid world가 됨)  
        - 노란색: 모델이 다음에 어떤 일이 일어날지 아는 곳   
        
    - Model: 로봇이 첫 에피소드 동안에 방문했던 모든 state를 앎.  
      - 2nd episode 진입?  
        - `Dyna`는 1st episode 동안 방문했던 어떤 state-action pair에 대해서도 transition simulating이 가능  
        - 그 transition들을 마치 실제 세계에서 일어난 것처럼 자유롭게 replay 해봄  
        - 매 plannig step은 단순히 'simulated transition에 Q-learning update를 적용하고, value function을 업뎃'  
      - 충분한 planning step을 거치면, 꽤나 괜찮은 policy를 생성해냄. (완벽하진x)   
        ![](https://drive.google.com/open?id=1Ya73TB1OmMSiE_7NVqcf0SxM89uPSH2z)  
        
    - Q-Learning은 위와 비슷한 성능을 보이려면 다수의 episode를 필요로하지만,  
      Dyna를 썼을 때는 한 번의 episode만을 거치고도 goal에 빨리 도달하는 방법을 알게 됨  
      - `Dyna는 step마다 더 많은 연산을 하지만, 제한된 experience를 효율적으로 사용한다.`  
      
### 5.3.2 The Dyna Algorithm  
- 앞서서, Dyna: planning, learning, acting의 결합(unification)   
  - 이 결합은 추가적인 개념 몇 가지를 소개. (ex. model learing, search control)  

- `Tabular Dyna-Q` algorithm  
  - Dyna architecture의 구체적 예시 중 하나  
  - deterministic transition을 가정  
  
  ![](http://drive.google.com/uc?export=view&id=1jDyJZENM_xT6Pa_RrFV-o_AjSoP41xDd)  
  ▼ Q-learning update 먼저 수행 (direct-RL이라 부르는 것)  
  - 먼저, agent는 ε-greedy policy에 따라 action을 선택하고, 그에 따른 next state와 reward 관찰  
    - 이 까지만 한다면 단순히 Q-Learning. 추가적인 model-dependent method를 실행하면 Dyna-Q   
  
  ▼ Dyna-Q!  
    - 앞에서의 transition을 취하고, model-learning step 수행 (Q-Learning은 model-free였다는 것 감안)  
      - 이를 위해서는 algorithm이 주어진 state-action pair에 대해 next state와 reward를 외우고 있어야.  
        - environment가 deterministic하다고 가정했기에 가능한 얘기  
        
    - planning 단계 여러 번 수행  
      - 각 planning step의 구성  
      
        ```  
        1. search control
        2. model query
        3. value update
        ```  

        1. search control  
          - 랜덤하게 이전에 방문한 state-action pair를 선택  
          - 이 때의 state-action pair는 반드시 이전에 본 적이 있는 pair여야.  
            - 그렇지 않다면, 모델은 다음에 무엇이 일어날지 모를 것  
        2. model query  
          - 주어진 pair에 대해, 다음 state와 reward에 대한 모델 생성  
            - = model transition 생성한 것.  
        3. value update  
          - Dyna-Q는 simulated된 transition을 바탕으로 Q-learning update 수행  
          
        - Planning step은 여러 번 반복된다.  
        
    - `Dyna-Q는 각 environment transition에 대해 많은 planning update 수행`  
    = __Tabular Dyna-Q__  (a simple instance of the Dyna architecture)   
    
- 앞전의 로봇의 예를 통한 `Dyna-Q algorithm` 이해    
  ![](https://drive.google.com/open?id=1Ppr51VkF_KOIZmJB6kUT955GufVYpf8P)  
  - 로봇은 environment에 대해 아무 것도 모르는 데서 출발해서, 첫번째 episode에서 184 step을 거치며 점차 모델을 만듦  
  - agent는 파란색 화살표 부분만 state-action pair value 업데이트  
  - planning이 없다면 이다음의 episode들도 꽤나 긴 시간이 걸릴 것  

  - 매 time step마다 100 step의 planning을 한다고 해보자.  
    - 한 step 이후, planning은 두 개의 state value를 더 업데이트  
      - 이제 agent는 final corridor 에서의 올바른 action을 앎.  
    - 이를 계속하면, Dyna-Q는 거의 모든 state space에 걸쳐 reward 정보를 propagate  
    
  ![](http://drive.google.com/uc?export=view&id=1JQpsqpDwjNGpwIfmj6gR9bSvXOqVFqzr)  
  - planning에 의해 계산된 policy는 goal에 도착하는 데 18 step만이 필요했음.  
  - 두 개의 episode 이후에는, 첫 episode보다 10배이상 짧음.  
  
- Tabular Dyna-Q는 (planning이 없었을 경우에 비해) 훨씬 많은 value function update.  
- Dyna-Q는 environment와의 한정된 interaction을 효율적으로 더 잘 사용한다.  


### 5.3.3 Dyna & Q-learning in an Simple Maze  

  ![](http://drive.google.com/uc?export=view&id=1h4l02YufJjgvvH46QsxFz8tMZ8z3KJ4W)   
  - goal state로의 transition: reward +1, 나머지: 0  
  - episodic, discount: 0.95  
  - 세 개의 agent를 비교할 것  
    - 모두 α = 0.1, ε = 0.1  
    - 모두 action-value estimate: 0으로 초기화  
    - 50개의 episode에 대해 30 번 experiment &t take avg.  
    
- 매 에피소드를 완료하는 데에 평균적으로 취한 step 수 plotting  
  (agent가 잘 하고 있다면, 에피소드 수가 늘어날수록 # of step ↓일 것)  
  
  ![](http://drive.google.com/uc?export=view&id=139-QrsTuzZxii3iLVRS93D71nRf-JtSY)  
  - Dyna-Q with planning 0 = Q-learning algorithm  
    - 느리게 나아지면서, 14 steps per episode에서 평평해짐  
  - Dyna-Q with planning 5  
    - 0보다 훨씬 빨리 비슷한 성능에 도달  
  - Dyna-Q with planning 50  
    - 3 episode 정도면 괜찮은 policy 찾음  
    
  - `Dyna-Q is far more sample efficient.`  
