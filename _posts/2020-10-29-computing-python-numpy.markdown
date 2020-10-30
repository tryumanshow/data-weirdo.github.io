---
layout: post
title:  "Numpy!"  
subtitle:   "numpy"
categories: computing
tags: python
comments: true
---

- numpy를 사용하면서 한 번쯤은 다시 찾아볼 것 같은 것들을 기록합니다.

---  

- 가장 먼저 R로 코딩을 시작했고, 또 학부 강의를 수강하면서 직접 교수님께 배웠던 언어가 R이기 때문에, 제 머릿속은 온통 R의 명령어로 셋팅되어 있습니다. 사소한 것일지라도, 다시금 찾아볼 것 같은 numpy의 메소드들을 이곳에다 정리합니다. 

### repeat  
- `np.repeat(a, repeats)`  a array를 repeats 번 만큼 반복  


### nan  

- `np.nan`: 그 자체로 NA값을 의미! (R의 NA에 해당)  
- `np.nansum(x)`: nan값이 있더라도 nan값을 무시하고 합계를 계산  
- `np.nan_to_num(x)`: nan값을 0으로 변환  
 
### random  

- `np.random.choice(a, p)`: p의 확률에 따라 a를 선택  
  - `np.random.choice(direction, p=np.repeat(1/theta_0.shape[1], 4))` 와 같은 응용 가능!  
- `np.random.randn(a,b)`: \[0,1)의 범위에서 a x b dimension의 numpy array를 만듦 (input으로 scalar값만 넣으면, 벡터를 출력)  
