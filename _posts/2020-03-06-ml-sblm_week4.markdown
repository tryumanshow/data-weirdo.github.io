---
layout: post
title:  "Sample-based Learning Methods - Week4"
subtitle:  "SBLM_Week4"
categories: ml
tags: rl
comments: true

---

- Sarsa, Q-learning 및 Expected Sarsa algorithm에 관한 글입니다.  

---  


# 2. Sample-based Learning Methods - Week4  
## 4.1 TD for Control  

- `앞서 배운 TD algorithm을 GPI의 evaluation step에 쓰는 것은 어떨까?`라는 아이디어로 출발  
  
- GPI: policy evaluation + policy improvement  
  - Recall! Monte-Carlo 상에서의 GPI를 다시 상기해보자.  
    
    ![](http://drive.google.com/uc?export=view&id=1YIUDYtnH48nnl4ucvKB0LjnEqylKFBA1)  
    - 쥐는 아무 것도 모른채 random policy를 따라 행동할 것  
    - 그러다가 결국에는 치즈에 닿게 될 것  
    - 이 때, 쥐는 action value를 업데이트 하게 될 것  
    - 이 과정이 반복되면, 결국에는 `optimal policy`를 학습하게 될 것  
      - Monte Carlo 상에서의 GPI는 improvement 이전에 `full` policy evaluation step을 수행하지는 않는다는 점 상기할 것  
      - evaluate & improves after each `episode`  
    - policy evaluation step이 끝나고 policy improvement가 이루어지겠지.  
    
  - __위 과정을 TD에도 적용해보겠다.__  
    - GPI 상에서 TD를 사용하려면, `action value function을 학습해야!`  
    - (앞에서 state에서 state로 옮겨가면서 각 state의 value를 학습했다면,)  
      이번에는 `state-action pair에서 state-action pair로 옮겨가면서 각 pair의 value를 학습하는것`으로 관점을 바꾸어보자.  
      ▲ 이 관점이 바로 `Sarsa prediction`!!  
      
- __Sarsa algorithm__  
  - St At Rt+1 St+1 At+1의 acronym  
  - Sarsa는 state-action pair value에 대한 prediction 이행  
    - initial state에서 action을 택한다 (그럼 state-action pair가 형성됨)  
    - 그 후, action을 취하면, reward를 얻고 다음 state로 넘어감  
    
    - Sarsa algo: agent는 value estimat을 업데이트 하기 전에 다음 state-action pair를 알아야함.  
      - 이 말인 즉슨 업데이트를 위해서 다음 action을 알아야 함!  
      - agent는 특정 policy에 대한 action value를 배우고 있으므로 다음 action을 샘플링 하기 위해 그 policy를 사용  
      
  - Full update equation (Policy Evaluation)   
    (아, latex 페이지가 가끔 Invalid Equation을 뱉는 경우가 있는데, 오늘 또 그러네...)  
    ![](http://drive.google.com/uc?export=view&id=1C1aaZgvIHisl8Sj46QltdAZ6QFJu5msX)  
    - Sarsa의 update 알고리즘은 TD update algo.와 굉장히 닮았네 (state value를 action value로 대체한 것일 뿐)  
  - 위를 바탕으로 `control`  
    - 역시 episode 이후나 convergence 이후가 아닌, 매 time step마다 policy improvement 가능  
    
### 4.1.2 Sarsa in the Windy Grid World  
      
- Windy Gridworld 예시  
  ![](http://drive.google.com/uc?export=view&id=1nLKc2vv5qp2kHsEZD9AdE2v9ojxP0TDm)  
  - 하나의 start state와 terminal state  
  - 4개의 방향으로 움직일 수 있고, 각 step 마다 reward -1  
    - agent로 하여금 가능한한 빨리 탈출하려 하도록 만들 것  
  - episodic task → γ = 1  
    
  - 여기 twist situation 추가 ~_~  
    - 특정 state를 지날 때 바람이 agent를 위로 민다.  
    - state마다의 바람의 세기는 어떤 column에 있는가에 달려 있음.  
      (strength가 1인 컬럼에서 agent가 왼쪽으로 움직이면, 바람의 힘에 의해 왼쪽으로 한 칸, 위로 한 칸 움직이게 될 것)  
      ![](http://drive.google.com/uc?export=view&id=1XwAmLFXHUo99rsHyVUr7VyUUuVnvVu5N)  
    - 단, boudnary로 움직일 때는 아무일도 일어나지 않음.  
  
  ▼ Sarsa algo 적용!  
  - ε-greedy algo. 사용하겠다. (ε=0.1, α=0.5, 초기화값:0)  
    ![](http://drive.google.com/uc?export=view&id=1PoCXjigk16Vt0vY2vu6kCL5COoGH5T47)  
    - 우측 그림:  
      - 매 time step 이후에 완료된 episode들의 총 개수  
      - 100번의 run들에 대해 평균낸 값  
     
      - 처음 몇 몇 episode들은 완료하기 까지 수 천 번의 step을 필요로 했지만,  
        갈수록 곡선이 가팔라짐 (episode들이 더 빨리 완료된다는 것)  
        `Episode completion rate stops increasing`  
        = `Agent policy가 optimal policy 근처를 배회하기 시작했다` (exploration 때문에 정확히 optimal하지는 않을 것)  
        
      - 대략 7000 step 이후에는 greedy policy의 improving이 멈춤  
       
      - 이 상황에 Monte Carlo 방법은 잘 맞지 않을 것  
        - Monte Carlo는 episode가 끝나야만 학습이 일어나는데, terminate 되지 않는 많은 policy들이 있을 것이기 때문에  
          - 이런 deterministic policy는 'get trapped'되어 좋은 policy를 결코 학습하지 못할 것  
    
    - 좌측 그림  
      - Sarsa는 'get trapped'되지 않는다.  
        - 저런 policy들이 나쁘다는 걸 episode 동안에 학습할 것이기 때문에.  
        
## 4.2 Off-policy TD Control: Q-learning  
### 4.2.1 What is Q-Learning?  
- 강화학습의 최근 적용 사례  
  - Atari game 플레이 학습  
  - 교통 신호 컨트롤  
  - 웹 시스템 자동 configuring  
  
  ▲ 위 사례들: `단일` 알고리즘을 토대로 형성. (`Q-Learning`)  
  
- __Q-Learning__  
  - 1989에 만들어짐.  
  - online RL의 첫 주류  

  - pseudo-code  
    ![](http://drive.google.com/uc?export=view&id=1kuw6qJkOZX4ljbynz3vDBsGFYj4oPJMd)  
  
  ▼ Revisit  
  ![](http://drive.google.com/uc?export=view&id=1vjL1ZQzXzLL1pv7FxoWFBfVspur72tGr)  
  - Sarsa와 Q-Learning의 차이 → 왜?  
    - Bellman Equation 및 Bellman Optimality Equation을 다시 떠올려 보자.  
      - Sarsa는 Bellman Equation과, Q-Learning은 Bellman Optimality Equation과 suspiciously 비슷하다.  
      - Bellman Optimality Equation은 Q-Learning으로 하여금 policy improvement, policy evaluation을 반복하지 않고 q\*로부터 바로 학습하도록 함  
      
    - 정리 (1st row based on 2nd row)  
      
      |Sarsa|Q-Learning|  
      |-----|----------|  
      |Bellman Equation|Bellman Optimality Equation|  
      
    - 요컨대 Sarsa, Q-Learning 모두 Bellman Equation에 근거를 두고 있지만, 매우 다른 Bellman Equation에 근거를 두고 있는 것.  
  - 인용  
    - "Sarsa is sample-based version of policy iteration which uses Bellman equations for action values, that each depend on a fixed policy."  
    - "Q-Learning is a sample-based version of value iteration which iteratively applies the Bellman optimality equation."  

### 4.2.2 Q-learning in the Windy Grid World  
  
- 앞 전의 Windy Grid World 예시 
  - Sarsa와 Q learning 간의 비교  
  ![](http://drive.google.com/uc?export=view&id=1awbRs4yUAdnPLn73qGbFaXBo0mvHVzAz)  
    - 초반에는 두 알고리즘 성능 비슷  
    - 끝으로 갈수록 Q-Learning의 성능이 더 뛰어나다. (더 나은 final policy를 학습한 것처럼 보임) 
 
- __도대체 왜>>>>>__  
  - Sarsa는 agent가 exploratory action을 취할 때`마다` 업데이트 도지만  
    Q-Learning은 max값만을 취하기 때문에 agent가 '어떤 action이 다른 action보다 낫다'라고 평가할 때에만 업데이트  
  
- __그렇다면 SARSA가 더 잘하게 할 수는 없을까?__  
  - 0.5라는 큰 stepsize는 agent가 exploratory action을 취할 때 SARSA의 성능을 저하시킬 수 있다.  
  - α를 0.1로 낮추어 좀 더 오래 학습을 시켜보자.  
    - 속도는 느리겠지만 더 나은 policy를 찾을 것  
      ![](http://drive.google.com/uc?export=view&id=1ZJ72MAtioVWmuV1ml6NqwdOR-SfM_6H2)  
      - SARSA는 Q-Learning과 똑같은 final policy를 학습! (좀 더 느리게)  
      - 두 선의 기울기가 같기 때문에 두 알고리즘 모두 같은 policy로 수렴해야한다는 것을 알고 있음.  
      
        ```  
        똑같은 slope = agent가 똑같은 속도로 episode 끝낸다는 것  
        ```  
      - 한편, 이 실험은 RL이 파라미터의 영향을 받는다는 것을 역설하고 있는 것이기도.  
        - α, ε, 초기값, 실험횟수 등 모든 파라미터들이 최종 결과에 영향을 미칠 수 있다!!!  

### 4.2.3 How is Q-Learning off-policy?  
- Q-Learning  
  - off-policy algorithm!  
    - 앞에서, importance sampling을 이용한 off policy를 본 적이 있는데,  
      __Q-Learning은 어떻게 importance sampling을 사용하지 않고서도 off-policy가 될 수 있을까?__  
      
- Sarsa  
  - on-policy algorithm!  
  
  |-|Sarsa|Q-Learning|  
  |-|-----|----------|  
  |policy|on-policy|off-policy|  
  |Bootstrap|Agent는 다음에 취할 action에 대한 value를 bootstrap|Agent는 다음 state에서 취할 가장 큰 action value를 bootstrap|   
  
  - 요컨대, Sarsa는 behavior policy estimate로부터, Q-Learning은 optimal policy estimate로부터 action을 샘플링     ![](http://drive.google.com/uc?export=view&id=1y0xOlYlVeGzx9reVggzF6qvMXh-ztf1y)  
  
- RL에서 늘 등장하는 자연스러운 의문점  
  " Behavior Policy의 Target이 뭐지?"  
  - Q-Learning의 경우  
    - Target Policy:  
      ![](https://latex.codecogs.com/gif.latex?%5Cunderset%7Ba%7D%7Bargmax%7DQ%28s%2Ca%29)  
    - Behavior Policy:  
      학습 동안 모든 state action pair를 방문하는 그 어떤 것도 될 수 있음.  
      - ex) ε-greedy  
      
  - __그럼 Q-Learning이 off-policy라면 왜 important  sampling ratio가 등장하지 않는가?__  
    - `agent가 known policy를 갖고 action value를 추정하기 때문` (발음이.. 부정확.. the known인지 unknown인지)   
    - Q-Learning에서는 Importacne Sampling 없이 어떤 state 하에서라도 target policy의 expected return을 계산할 수 있다.  
      ![](https://latex.codecogs.com/gif.latex?%5Csum_%7Ba%27%7D%5Cpi%28a%27%7CS_%7Bt&plus;1%7D%29Q%28S_%7Bt&plus;1%7D%2Ca%27%29%20%3D%20E_%7B%5Cpi%7D%5BG_%7Bt&plus;1%7D%7CS_%7Bt&plus;1%7D%5D)  
      - agent의 target policy는 greedy하므로, maximum이 아닌 모든 action들은 확률이 0!  
      - `결과적으로, 한 state로부터의 expected return은 그 state로부터의 maximal action value와 같다`  
      
- 하지만, 앞에서 Q-Learning에 대해,  
  - policy evaluation과 policy improvement iteration을 하지 않고,  
  - optimal value로부터 바로 학습한다고 했었는데,  
  
  - 이는 물론 멋진 아이디어이지만, 이 때문에 특정 상황에서는 적용하기 쉽지 않은 경우가 있어..  
  
  - cliff 예시 재 사용  
    ![](http://drive.google.com/uc?export=view&id=1tQ6scoczrb0t8ODFHa_HtB3dfjejnlf1)  
    - Q-Learning은 optimal value function을 학습하기에 optimal policy를 빨리 학습  
      하지만 종종 cliff로 빠져버림  
    - 반면, SARSA는 현재 policy를 학습하기에 ε-greedy의 action selection의 영향을 고려 
      Q-Learning보다 시간은 오래 걸리겠으나 훨씬 `reliable` path를 낳음. (랜덤하게 cliff로 빠지는 action을 피함)   
        
 
- 각종 알고리즘의 이해를 도울 수 있는 그림.  
![](http://drive.google.com/uc?export=view&id=16asn9FohfLjhQtBvKJ-ImsQonGeMCzdo)  
  
## 4.3 Expected Sarsa  
### 4.3.1 Expected Sarsa  
- TD control methods: Sarsa, Q-learning에 대해 논의함.  
  - 다른 TD control method인 `Expected Sarsa`에 대해 논의해보기로.  
  
  ![](http://drive.google.com/uc?export=view&id=1IMOzFs4pUhiBEvpIxDiS1ldb8ESN11Tc)  
  - 1: action-value에 대한 Bellman Equation  
    = 가능한 다음 state-action pair들의 값들에 대한 `기대값(expectation)`  
  - 2: Sarsa는 위 기대값(q_π)를, 다음 state는 environment로부터, 다음 action은 policy로부터 샘플링 함으로써 추정(estimate)   
    - 그런데, agent는 이미 policy (`π(a'|s')`)를 아는데 왜 다음 action을 샘플링해야 하나?  
    - 3: '기대값을 직접 계산하면 되지 않나?'  
      - 모든 가능한 다음 action 값들의 가중치 합을 취할 수 있음. (weighted sum)  
        다음 action들에 대해 기대값을 계산하겠다는 것이 Sarsa algorithm의 메인 아이디어!  
        
        ![](https://latex.codecogs.com/gif.latex?Q%28S_t%2C%20A_t%29%20%5Cleftarrow%20Q%28S_t%2C%20A_t%29%20&plus;%20%5Calpha%28R_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20%5Cmathbf%7B%5Csum_%7Ba%27%7D%5Cpi%28a%27%7CS_%7Bt&plus;1%7D%29Q%28S_%7Bt&plus;1%7D%2C%20a%27%29%7D-Q%28S_t%2C%20A_t%29%29)  
        - 이처럼 기대값을 명시적으로 계산하는 건 썩괜찮음.  
          - Sarsa보다 더 stable한 update target을 지니고 있음.   
          
- 사례를 통한 Sarsa, Expected Sarsa 비교!  
  ![](http://drive.google.com/uc?export=view&id=1mL6qA_IEtrp0QXrpCE0eeCe9W5z2dfaw)  
  - Sarsa, Expected Sarsa 모두 다음 state에 대한 true action value로 시작  
    __But__  
    - Sarsa    
      : (심지어는 이상적인 경우라 하더라도) Sarsa의 다음 action에 대한 샘플링의 경우 value update가 잘못된 방향으로 일어날 수도 있다.  
      : 다수의 업데이트를 거친 기대값은 옳은 방향으로 갈 것이라는 사실에 기반   
    - Expected Sarsa  
      : 업데이트는 `Exactly Correct!`  
      : True value에서 벗어나게 추정 value값을 바꾸지 않는다.  
  - (일반적으로) Expected Sarsa가 Sarsas에 비해 훨씬 더 적은 분산을 갖고 타겟을 업데이트  
    - 다만, 분산이 더 작다는 건 그만큼의 downside도 있긴 함.  
      (action의 수가 증가할수록 다음 action의 평균 구하기 연산 비용 증가 - 매 time step마다 average를 게산해야되기 때문에)  
      

### 4.3.2 Expected Sarsa in the Cliff World   
- Cliff 예시 recap  
  - undiscounted episodic grid world  
  - cliff의 존재 (cliff로 가면 reward -100과 함께 agent가 start로 다시 돌아감)  
  - 그 외에는 매 time step마다 reward -1  
  
  - 적어도 이 문제에서는 Sarsa가 Q-learning보다 더 잘 수행  
  - __Expected Sarsa__ vs __Sarsa__ in cliff example  
    
    ![](http://drive.google.com/uc?export=view&id=1FpGsoedAxcF7Vv2_yEVVzVPm3_jbjxJp)  
    - ε=0.1 case  
    - step size 파라미터 α의 값을 agent마다 달리하여 테스트  
  - 100개의 에피소드 & 50000번의 independent run을 평균  
    
    - __Sarsa__  
      - step size 값이 커질수록 성능도 좋아짐. 단, 특정 값까지만.  
      - 최고의 α는 0.9  
      - 거의 모든 α값에 대해 Expected Sarsa가 Sarsa를 능가!  
      
    - __Expected Sarsa__  
      - 큰 α값을 더 효율적으로 사용  
        ∵ policy의 randomness를 명시적으로 평균내기 때문.  
        - Environment는 deterministic하고, 그래서 고려할 다른 randomness는 없다!  
          
          > Expected Sarsa's updates are deterministic for a given state and action.  
          > Sarsa's updates, on the other hand, can vary significantly depending on the next action.  
    
  - 이번엔 100000 개의 episode!  
    ![](http://drive.google.com/uc?export=view&id=1s1C76kS93m1YqN-Li_T0ttgrsohYlpsy)  
    - 이 때는, 두 알고리즘 모두 학습할 거리는 이미 다 해버림!  
      - 이 때 Expected Sarsa의 long-term behavior은 alpha 값에 의해 영향을 받지 않고, update도 deterministic  
        - 이 말인 즉슨, step size는 estimate가 target value로 얼마나 빨리 접근해가냐를 결정할 뿐.  
      - Sarsa: 개판!  
        - α값이 커지니까 수렴 실패.  
        - α값이 작아지니까, long-run performance가 expected Sarsa의 그것에 접근.  
 

### 4.3.3 Generality of Expected Sarsa    
- 이제껏 배운 TD algo for control  
  
  ```  
  1. Sarsa  
  2. Q-learning  
  3. Expected Sarsa  
  ```  
  
- `1. Sarsa`와 `3. Expected Sarsa`는 똑같은 Bellman Equation을 근사  
  & (target updating시 target policy의 기대값을 이용한다!)  

- 문제의식: __2. Q-Learning과 3. Expected Sarsa 간에도 관련이 있다???__  
  
- on-poliy case  
  - behavior policy = target policy  
  
  - Recall. Expected Sarsa Update  
    ![](https://latex.codecogs.com/gif.latex?Q%28S_t%2C%20A_t%29%20%5Cleftarrow%20Q%28S_t%2C%20A_t%29%20&plus;%20%5Calpha%28R_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20%5Csum_%7Ba%27%7D%5Cpi%28a%27%7CS_%7Bt&plus;1%7D%29Q%28S_%7Bt&plus;1%7D%2C%20a%27%29-Q%28S_t%2C%20A_t%29%29)  
    - 다음 action은 π로부터 샘플링 되는데, action들의 기대값은 실제로 다음 state에서 선택되는 action이랑은 독립적으로 (independently) 계산됨.  
    - 사실 π가 behavior policy랑 같아야할 필요가 없기도 함  
    
    > Expected Sarsa도 (Q-Learning처럼) importance sampling 없이 off-policy 학습이 가능하다  
    
    - 만약 target policy가 greedy 하다면!?  
      ![](http://drive.google.com/uc?export=view&id=1R0c_jZ5JmCpeP0oUc6JKVmjX0gv-aYWp)  
      ↓  
      ![](http://drive.google.com/uc?export=view&id=121ra_Xle2ytEXDq60UOBplovZBZ6ANK-)  
      
      > Q-Learning은 Expected Sarsa의 special case다!  


### 4.3.4 Week 3 Summary  
- TD control algorithm: Bellman Equation에 근거하고 있다.  
  - Sarsa  
    - Bellman Equation의 sample based version 사용  
    - q_π를 학습  
    - On-policy  
  - Q-Learning  
    - Bellman optimality equation을 사용  
    - q_*를 학습  
    - Off-policy  
  - Expected Sarsa  
    - Sarsa와 같은 Bellman Equation 사용  
    - 다만 샘플링 방식이 조금 다름  
      - 다음 action value의 기대값 사용  
    - Off-policy (사실 On-policy, Off-policy 둘다!)  
  ![](http://drive.google.com/uc?export=view&id=1hyin7a8YUKSXT6f_uI0k6EPe3M55csas)  

- Sarsa vs Q-Learning  
  - online-learning 시 Sarsa가 Q-Learning보다 더 나을 수 있다.  
    - on-policy control은 exploration까지 고려하기 때문.  
    - cliff 예시에서, Q-Learning은 exploratory action 때문에 종종 떨어짐  
    - Sarsa는 학습 시간이 길었지만 더 안전한 path를 거침으로써 cliff로 거의 떨어지지 않았고, 결과적으로 reward도 더 컸음.  
  
- Sarsa vs Expected Sarsa  
  - 모든 step size에 대해 Expected Sarsa가 Sarsa보다 낫더라!  
    - ∵ Expected Sarsa의 variance 완화 (policy 때문)  
      - 이름이 의미하듯, 다음 action들의 기대값!  
