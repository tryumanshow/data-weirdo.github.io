---
layout: post
title:  "Numpy 유용 함수 정리 (ing)"  
subtitle:   "numpy"
categories: computing
tags: python
comments: true
---

- numpy를 사용하면서 한 번쯤은 다시 찾아볼 것 같은 것들을 꾸준히 기록합니다.

---  

- 가장 먼저 R로 코딩을 시작했고, 또 학부 강의를 수강하면서 직접 교수님께 배웠던 언어가 R이기 때문에, 제 머릿속은 온통 R의 명령어로 셋팅되어 있습니다. 사소한 것일지라도, 다시금 찾아볼 것 같은 numpy의 메소드들을 이곳에다 정리합니다. 

### repeat  
- `np.repeat(a, repeats)`  a array를 repeats 번 만큼 반복  

### nan  
- `np.nan`: 그 자체로 NA값을 의미! (R의 NA에 해당)  
  - `np.float('inf')`: 그 자체로 ∞를 의미! (`-np.float('inf'): -∞)  
- `np.nansum(x)`: nan값이 있더라도 nan값을 무시하고 합계를 계산  
- `np.nan_to_num(x)`: nan값을 0으로 변환  
 
### random  
- `np.random.choice(a, p)`: Given 1-d array인 a로부터 p의 확률에 따라 랜덤 샘플링  
- `np.random.rand()`: \[0,1)의 Uniform Distribution에서 랜덤샘플링 한 후 주어진 형태의 array 생성    
- `np.random.randn()`: 일변수 표준정규분포로부터 랜덤샘플링 한 후 주어진 형태의 array 생성   
- 

### 비교   
- `np.array_equal`: 두 array object가 같은지를 비교 (Elementwise한 비교가 아님) -> True 아니면 False 둘 중 하나의 단일 boolean 값만을 반환  
