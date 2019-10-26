---
layout: post
title:  "Outlier Detection에 대한 소개"
subtitle:   "outlier_detection-intro"
categories: ml
tags: detection
comments: true

---

- 통계학에서와 마찬가지로, 기계학습에서도 outlier를 어떻게 처리할 것인가는 큰 관심사입니다.
Outlier가 무엇이고, 종류로는 어떤 것들이 있고, outlier를 탐지할 때 어떤 것들을 유념해 두어야 하는지
에 대해 정리해보았습니다. 또, 몇 가지 outlier detection 방법론에 대해서도 열거해보았습니다. 

---

# Outlier란 무엇일까?

이상치(Outlier)란 주어진 데이터 상의 다른 관찰값들로부터 떨어져있는 극단적인 값을 의미합니다.
(`Outliers are extreme values that deviate from other observations on data.`)
전반적인 샘플의 패턴과 다른 모습을 보이는 관찰값이라고 할 수 있겠습니다.

![](https://miro.medium.com/max/480/0*R9u16eEcsZHpjH4O.)

---

# Outlier의 종류

아웃라이어에는 두 가지 종류의 아웃라이어가 있습니다.  
```
구분1
1. Univariate outlier  
2. Multivariate outlier
```
하나는 일변량 아웃라이어(Univariate outlier)이고, 다른 하나는 다변량(Multivariate outlier)입니다. 이 둘은 몇 차원의 공간을
탐색하고 있냐에 따라 쉽게 구분될 수 있습니다. 하지만 다차원의 공간을 사람의 뇌로 처리하기는 굉장히 어려운 
일이라 이 부분이 바로 모델이 인간 대신 해주길 바라는 부분이기도 합니다. 

한편, 아웃라이어는 다음과 같이 구분될 수도 있습니다.  
```
구분2
1. Point outliers  
2. Contextual outliers 
3. Collective outliers 
```

* Point outliers (점 이상치) 
: Point outlier는 분포에서 나머지 점들과 떨어져있는 데이터 포인트 하나하나를 일컫습니다.  
* Contextual outliers (상황적 이상치)
: 상황에 맞지 않는 데이터를 의미합니다. 한 여름의 25도는 정상일 수 있으나 겨울에 25도를 기록하고 있다면 이는 분명 
상황에 맞지 않는 데이터입니다.
* Collective outliers (집단적 이상치)
: 데이터 내에서 관측치 하나하나는 이상치가 아닐지라도, 일부 모인 데이터가 이상한 양상을 띈다면 이것이 곧
collective outlier라 할 수 있습니다. 아래의 그림에서, 분명 데이터 내의 점 하나하나들은 그리 이상해
보이지 않지만, 중간의 점들이 모이니 새로운 경향성을 띠고 있습니다. 

![](https://miro.medium.com/max/736/0*oj1v0rhvPt0u-jYH.)  

---

# Outlier Detection을 할 때 유념해야 할 것?  

이상치를 탐지할 때 두 가지를 유념할 것을 주문합니다.
```
1. 이상치 탐지를 위해 어떤, 그리고 얼마나 많은 피쳐들을 고려할 것인가? `(univariate / multivariate)`
2. 내가 선택한 피쳐들에 대해 분포를 가정할 수 있는가? `(parametric / non-parametric)`
```

# Outlier Detection 방법론  
- Z-Score or Extreme Value Analysis
- Probabilistic and Statistical Modeling
- Linear Regression Models
- Proximity Based Models
- Information Theory Models
- High Dimensional Outlier Detection Methods (high dimensional sparse data)

차차 이 방법론들을 학습해볼 것이고, 다음 포스팅은 Z-score, Dbscan, Isolation Forest에 대해 정리해볼 
것입니다. 

  
#### Reference
[A Brief Overview of Outlier Detection Techniques](https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561)  
[이상감지(Anomaly Detection)과 아웃라이어 감지(Outlier Detection)의 차이](http://intothedata.com/02.scholar_category/anomaly_detection/)  
[Novelty Detection(이상치 탐지) - Overview](https://jayhey.github.io/novelty%20detection/2017/10/18/Novelty_detection_overview/)  
