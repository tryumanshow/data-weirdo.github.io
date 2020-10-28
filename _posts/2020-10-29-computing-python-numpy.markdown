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

### nan  

- `np.nan`: 그 자체로 NA값을 의미! (R의 NA에 해당)  
- `np.nansum(x)`: nan값이 있더라도 nan값을 무시하고 합계를 계산  
- `np.nan_to_num(x)`: nan값을 0으로 변환  
 
### random  

- `np.random.choice(a, p)`: p의 확률에 따라 a를 선택  
