---
layout: post
title:  "Outlier Detection-Z-Score"
subtitle:   "outlier_detection-z-score"
categories: ml
tags: detection
comments: true

---

이상치 탐지의 방법으로 Z-Score에 대해 공부해보겠습니다.

---

# Z-Score
: `Z-score`는 가우시안 분포에서 관찰값들이 표본평균에서 얼마나 떨어져있나(표준편차 이용)를 측정하는 방법입니다.
분포를 가정하고 있기 때문에 Z-Score 방법은 parametric한 방법으로 분류할 수 있습니다. 
현실의 많은 데이터들은 가우시안 분포에 의해 설명되지 않지만, 이 문제는 데이터에 변환(Transformation)을 가함
으로써 해결할 수 있습니다. ex. scaling

참고로 파이썬 라이브러리들은 Pandas, Numpy 등과 더불어 이런 계산들을 쉽게 해줄 수 있는 Scipy, Scikit Learn
라이브러리들을 제공하고 있습니다 :)

데이터셋에서 우리가 선택한 피처 공간에 대해 적절한 변환을 가해준 후 모든 데이터는 z-score로 다음과 같이 표현됩니다.
![](https://miro.medium.com/max/85/0*TwXvmgI5j7ArPPq4.)  
Z-score로부터 outlier를 판단하기 위해서는 미리 threshold를 정해주어야 하는데, 경험적으로
표준화된 정규분포로부터 2.5, 3, 3.5 표준편차 정도의 거리에 있는 점들을 outlier로 판단하게 됩니다. 
해당 threshold 밖의 점들은 outlier, 그렇지 않은 점들은 outlier가 아닌 점들로 분류할 수 있습니다. 

![](https://miro.medium.com/max/560/0*i5Moxki9Pe2noYN2.)

Z-score 방법은 매우 단순하지만, 이상치들을 제거하기에 여전히 강력한 방법입니다. parametric한 분포를
가정하고 있다면요. Z-score는 `Parametric + Low dimensional`에서 효과적이라 하겠습니다. 

이후 소개할 Dbscan과 ISOLATION Forest는 nonparametric 문제에 대한 좋은 대안이 될 수 있습니다. 

---






#### Reference
[A Brief Overview of Outlier Detection Techniques](https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561)  
[Novelty와 Outlier Detection](https://flonelin.wordpress.com/2017/03/29/novelty%EC%99%80-outlier-detection/)  
