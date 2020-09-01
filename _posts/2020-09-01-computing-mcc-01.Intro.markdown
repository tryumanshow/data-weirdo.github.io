---
layout: post
title:  "MCC01 - Parallel Computing 소개"  
subtitle:   "mcc01"
categories: computing
tags: multicore_computing
comments: true
---

- Parallel Cmoputing에 대한 Overview입니다.  

---  

## 병렬 컴퓨팅  
- 많은 계산을 `동시에`하는 연산의 한 방법  
- 현실에 대입해보기  
  - Call center: 수화원들 간에는 별다른 상호작용 없이 자신의 업무 처리  
  
- 병렬 컴퓨팅이 사용되는 사례들 흔히 볼 수 있음  
  - N-body Simulation (천체 입자들의 움직임 시뮬레이션)  
  - Google page-rank algorithm (웹 페이지들의 중요성 결정)  
  
- Instruction Level Parallelism (ILP)  
  = Hidden Parallelism  
  - ILP를 지원하는 CPU에서 application을 실행할 경우, instruction reordering 문제를 조심해야 함.  
    CPU가 Parallelism level 개선을 위해 기계어(machine instructions)를 자동으로 reordering 해버릴 수도 있음.  
    
- 병렬컴퓨팅이 흥하게 된 계기  
  - 무어의 법칙  
    : '매년마다 Transistors/inch^2 가 두 배 씩 증가한다.'  
    - 실제로 잘 들어맞았음. (정말 매년 트랜지스터의 밀도가 증가)   
  - 하지만 본질적인 문제에 부딪힘  
    -> 트랜지스터 밀도가 높아질수록, CPU의 Power 소모는 커지고, 이에 따라 온도도 높아지면서, Malfunction이 발생  
  - 해결책: `Parallelism`  
    : (트랜지스터의 밀도를 높이는 대신,) Core의 개수를 늘리자.  
    
- 하지만 단순히 Core의 개수를 늘린다고 반드시 좋은 건 아니다.  
  - Ex. 스마트폰 -> 코어가 8개까지 필요할까?  
    - No. 4개 정도면 충분할 것.  
      - Android OS 용  
      - Main Application 용  
      - Background Music 용  
        ...
    - 이런 경우는 Multi-core의 이점은 제대로 사용하지 못하면서, 배터리 파워만 많이 소모하는 경우.  
      (이런 경우 Heterogeneous CPU를 사용하기도)  
      - 다양한 크기의 CPU를 두고, 어떤 Task를 수행하는지에 따라 다양한 CPU에 할당)  
 
### Parallelism  

- `Paralleism` = concurrency + parallel hardware  
  - Concurrent: 한 CPU에서 多 프로그램을 동시에 돌릴 수 있다는 것  
  - Parallel: 한 프로그램을 多 core를 이용하여 돌릴 수 있다는 것  
  
  - 병렬처리에서는 이 두 개념 모두가 필요하다.  
  
  - 병렬 컴퓨팅 사용의 이유: 'High-performance'  
    - Serial Program을 Parallel하게 만들고 싶을 수 있음.  
      - 하지만 그렇게 만만한 일이 아님.  
      - 이를 위한 다양한 API들이 있는데 그 중 하나가 `OpenMP`  
        (OpenMP의 성공적인 활용을 위해서는 병렬 컴퓨팅의 이론에 대해서 반드시 잘 숙지하고 있어야만 함)  
        (이것이 '병렬컴퓨팅' 학습의 이유)  
    
#### Parallelism의 종류  
└ 1. Task Parallelism  
└ 2. Data Parallelism  
  
  Ex) 
  특정 강의 수강자 300명의 중간고사 시험지 채점  
  문항 수 3개  
  세 명의 TA  
  
  1. Task Parallelism  
  - 문항 1번은 TA1이, 2번은 TA2가, 3번은 TA3가 채점하는 경우  
  - 3개의 코어가 '다른' 문제를 할당받음  
  
  2. Data Parallelism  
  - 세 명의 TA가 100개씩 채점하는 경우  
  - 3개의 코어가 '똑같은' 문제를 할당받음  
  
  - Parellelism에서는 Core들 간의 `Coordinate`가 중요  
    
    ```  
    1. Communication 
    2. Load Balancing 
    3. Synchronization  
    ```  
    위 세가지를 모두 만족하는 방향으로 처리되어야.  
    
    
#### Parallel Systems의 종류  
└ 1. Shared-memory system  
└ 2. Distributed-memory system  

  1. Shared-memory system  
  - 1 Physical computer with 多 CPU cores  
  - Data를 Share할 수 있다.  
  
  2. Distributed-memory system  
  - Data를 Share할 수 없다.  
  - Network 송수신을 통해 데이터를 공유해야 함  
  
  ![mcc01-1](https://user-images.githubusercontent.com/43376853/91813825-36502080-ec6e-11ea-8400-8aa32ea93fa3.png)  

