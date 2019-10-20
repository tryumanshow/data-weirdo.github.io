---
layout: post
title:  "Outlier Detection-Isolation Forest"
subtitle:   "outlier_detection-isolation_forest"
categories: ml
tags: detection
comments: true

---

- Outlier Detection의 한 방법인 Isolation Forest에 대해 정리해보았습니다.

---

Isolation Forest는 데이터의 outlier와 novelty를 탐지하는 데에 
있어서 유용한 방법입니다. 이 모델은 이진 결정 트리에 입각한 모델입니다. 

![](https://miro.medium.com/max/981/0*0GuMixLdSZo3V3Nh.)  

Isolation forest의 기본 원칙은 `Outlier는 그 수가 적고, 나머지 다른 관찰값들과 멀리 떨어져있다.` 라는 것입니다.    
트리를 만들기 위해 feature space에서 랜덤하게 feature 하나를 뽑고 그 feature를 이용하여 최대값과 최소값 사이의 범위에서 
랜덤하게 공간을 분할해 갑니다. 모든 관찰값들이 마지막 노드에 위치할 때까지 feature를 뽑고 분할하는 과정을 반복합니다. 
이 때 정상치인가 이상치인가를 판단해내기 위해 사용되는 개념이 `Path Length`라는 것인데 그 이름이 의미하듯
'어떤 관찰값이 마지막 노드가 되기 위해 몇 번을 분핟되어야 하는가' 정도로 이해할 수 있습니다. 이상치라면 
적은 분할로도 마지막 노드에 위치할 것이고, 정상치들은 많은 수의 분할을 거친 뒤에라야 마지막 노드에 위치하게 될 것 
입니다. 이 과정을 거치면 '한'개의 트리가 만들어질 것이고, 여러 트리 모델들을 만들어 앙상블합니다. 물론 이 
모든 과정은 훈련 세트 (training set)을 대상으로 한 것입니다. 

앙상블 모델은 `이상지수(Outlier score)`라는 지표를 계산해내게 됩니다.  
![](https://miro.medium.com/max/206/0*uVVSUfptaeFzqRZW.)  
 
```
h(x): 샘플 x의 path length  
n: external node의 개수  
c(n): 이진 트리의 'unsuccessful length search'
```  

모든 관찰값들에 대해 0과 1 사이의 값으로 스케일링 될 것이며, 1에 가까울수록 이상치라고 판단하게 됩니다. 
보통의 threshold는 0.55, 0.66 정도라고 합니다. (단, 사이킷런이 제공하는 isolation forest 라이브러리는 
0.5만큼 shift하고 revers까지 했기 때문에 -0.5와 0.5 사이의 값이 도출되며, 값이 작을수록 더 이상치에 
가깝다고 합니다.  

이상 Isolation Forest에 대한 내용이었습니다. 


#### References  
[2.7. Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)  
[A Brief Overview of Outlier Detection Techniques](https://miro.medium.com/max/981/0*0GuMixLdSZo3V3Nh.)  
[의사결정나무를 이용한 이상탐지](https://ko.logpresso.com/documents/anomaly-detection)  
[Spotting Outliers With Isolation Forest Using sklearn](https://dzone.com/articles/spotting-outliers-with-isolation-forest-using-skle)  
[3.Tree에 대해서 알아보자.](https://gnujoow.github.io/ds/2016/08/27/DS3-Tree/)  
[Data Structures and Algorithms
with Object-Oriented Design Patterns in
C#](http://programming.etherealspheres.com/backup/ebooks%20-%20Other%20Programming%20Languages/Data%20Structures%20And%20Algorithms%20With%20Object-Oriented%20Design%20Patterns%20In%20C%20Sharp.pdf)
