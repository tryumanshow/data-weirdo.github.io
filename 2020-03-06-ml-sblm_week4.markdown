---
layout: post
title:  "Sample-based Learning Methods - Week4"
subtitle:  "SBLM_Week4"
categories: ml
tags: rl
comments: true

---

- Sarsa algorithm 및 Q-learning에 관한 글입니다.  

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
    - 역시 episode 이후나 convergence 이후가 아닌, 매 teim step마다 policy improvement 가능  
    
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
        
