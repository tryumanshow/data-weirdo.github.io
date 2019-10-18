---
layout: post
title:  "Outlier Detection-Dbscan"
subtitle:   "outlier_detection-dbscan"
categories: ml
tags: detection
comments: true

---

지난 포스팅에 이어 이번에는 Dbscan(Density Based Spatial Clustering of Applications with Noise)
을 사용한 Outlier Detection에 대하여 정리해보았습니다.

---

기계학습에서 클러스터링 방법들은 데이터를 시각화하거나 데이터를 더 잘 이해하는 데에 도움을 줘 왔습니다.
하지만 클러스터링 방법들은 다차원의 비모수적인 분포에서(`multivariate & nonparametric`) 
이상치들을 탐지하는 데에도 사용된다고 합니다. 그 중 오늘은 Dbscan을 통한 이상치 탐지에 대해 
공부해보았습니다.

# Dbscan이란?  

`Dbscan`은 밀도 기반의 클러스터링 알고리즘입니다. `K-Means`나 `Hierarchical` 클러스터링의 경우 군집간의 거리를 이용해서 
클러스터링을 했으나 Dbscan은 그 이름이 말해주듯 밀도 기반으로 클러스터를 만듭니다. n 차원의 구 공간에서 반지름 $$\varepsilon$$ 안에
클러스터로 인정하기 위한 최소한의 점(MinPts)이 있는지 없는지를 통해 밀도를 판단하여 클러스터를 만드는 그런 알고리즘입니다.  

Dbscan 군집화 알고리즘에서, 각각의 점들은 이 셋 중의 하나에 해당 됩니다.
```
1. Core point
2. Border point
3. Outlier
```  
- Core point  
  : 어떤 한 점으로부터 $$\varepsilon$$의 거리 안에 최소한 MinPts 만큼의 점이 있다면, 그 어떤 한 점은 Core point가 됩니다.  
  
- Border point  
  : 어떤 한 점이 클러스터 안에는 있지만 그 점으로부터 $$\varepsilon$$ 안의 거리 안에 MinPts보다 적은 수의 점이 있고
  그렇지만 'density reachable'하다면 그 어떤 한 점은 Border Point가 됩니다.  
  
- Outlier
  : 어떤 한 점이 클러스터 안에 있지도 않고 'density reachable'하지도 않으며 다른 점들에 대해 'density connected'하지도 않다면
  그 어떤 한 점은 Outlier가 됩니다. Outlier는 단지 자기 자신의 클러스터를 가지게 될 뿐입니다.  
  
이 그림은 세 종류의 데이터 포인트에 대한 이해에 도움이 될 것입니다. 아울러 바로 아래에서 reachable 함과 connected 함에 대해서 설명하는 데에도
도움이 될 것입니다.  

![](https://miro.medium.com/max/400/0*3A8VdnNSC2d32Q_I.)  

어떤 점 p가 $$\varepsilon$$ 만큼의 거리 내에 최소 MinPts 만큼의 점을 갖고 있다면 점 p는 `Core point`입니다.  
이 점 p로부터 구 모양의 클러스터를 만들었을 때 이 클러스터 내의 모든 점들은 p로부터 `directly reachable`한 점이 됩니다.  
어떤 점 q가 있고, $$p_1$$, $$p_2$$, $$p_3$$, ... $$p_n$$라는 점들이 있는데, 이 때 $$p_1$$가 p이고 $$p_n$$가 q라고 하겠습니다.
각각의 $$p_{i+1}$$는 $$p_{i}$$와 `directly reachable`할 때, q는 p와 `reachable`한 관계에 있습니다. 단, 이 때 $$p_1$$부터  
$$p_{n-1}$$까지의 점은 모두 각자가 `Core point`이어야 합니다.  
어떤 점으로부터도 `reachable`하지 않다면 그 점은 outlier 혹은 noise point가 됩니다.  
  
이 때, 어떤 o라는 점이 P와 Q라는 점 모두로 부터 `reachable`하다면, 결국 P와 Q가 공유하게 되는 o라는 점이 존재한다면
P와 Q는 `density connected` 한 점들이 됩니다.  (p, q와 동일한 점으로 오해될 소지가 있는 것 같아 일부러 대문자로 적었습니다.)  

클러스터는 다음과 같은 성격을 지닙니다.  
- 같은 클러스터 내의 모든 점들은 `density-connected 되어 있다.`
- 어떤 한 점이 클러스터 내의 임의의 한 점으로 부터 `density-reachable`하다면 그 어떤 점 또한 클러스터에 속한다.

## 파라미터에 대한 설명과 그에 대한 조정

다음은 사이킷런에서 제공하는 Dbscan 함수입니다. 
```
class sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric=’euclidean’, metric_params=None, algorithm=’auto’, leaf_size=30, p=None, n_jobs=None)
```
여기서 eps가 $$\varepsilon$$, min_samples가 MinPts에 해당할 것입니다.  

### Scaling  
$$\varepsilon$$이 클러스터를 만드는 데 있어서 중요한 역할을 하기 때문에, 이를 잘 측정하기 위해 
데이터를 잘 스케일링 하는 것이 필요합니다. 이 문제에 대해 괜찮은 스케일러로 `Robust Scaler`를 추천하네요.  

### Choosing Spatial Metric  
Feature space를 스케일링 해주고 난 뒤에는 클러스터링을 실행할 Spatial Metric을 선택해야 한다고 합니다.
2·3차원이라면 유클리디안 공간에서 잘 작동하지만, 더 차원이 많아지면 manhattan metric도 유용할 수 있다고 하네요.

### Choosing epsilon value  
이제 피쳐 스케일링과 Spatial Metric 선택을 마쳤으니 $$\varepsilon$$을 정해주어야 하는데, 
0.25와 0.75 사이의 값들로 시도해보는 걸 추천하고 있었습니다.  

### Choosing MinPts  
한편, MinPts는 문제에 따라 다르니 알아서 잘 조정하라고 합니다.  


## 부가적인 설명  
1. Dbscan의 시간복잡도는 O(nlogn)으로 medium 사이즈의 데이터 셋에서 효과적이라 합니다.  
2. 사이키런을 사용해서 dbscan을 fitting 하고 나면 클러스터가 나뉘고 각 샘플은 각자의 클러스터에 할당됩니다.
Outlier들은 -1 클러스터에 할당되고, 차후 이들을 제거한 뒤에 분석을 진행할 수 있습니다.  


다음은 housing price 데이터셋에 Dbscan을 사용해서 이상치들을 골라낸 그림이라고 합니다. 
![](https://miro.medium.com/max/1400/0*A1Wupu3hKsJMvUdH.)  

다음에는 Isolation Forest에 대하여 공부해보겠습니다. 



#### Reference
[A Brief Overview of Outlier Detection Techniques](https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561)
[DBSCAN (밀도 기반 클러스터링)](https://bcho.tistory.com/1205)
[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)
[sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
