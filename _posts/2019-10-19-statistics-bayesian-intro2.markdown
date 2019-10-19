---
layout: post
title:  "Bayesian-Intro2"
subtitle:   "Bayesian-intro2"
categories: statistics
tags: bayesian
comments: true

---

- 지난 Intro에 이어 estimation에 대한 빈도주의와 베이지안의 차이, 빈도주의적인 접근이 잘 작동하지 않는 경우, 
그리고 주어진 데이터들에 대해 미래의 관찰값이 갖게 될 확률에 대해 추가적으로 정리해보았습니다. 

---

# Estimation에 대한 빈도주의와 베이지안의 차이  

$$Y_1$$, $$Y_2$$, $$Y_3$$, ... $$Y_N$$, N개의 데이터가 있고 이 데이터들은 분산이 1인 정규분포를 따르고 모분산에 대해서는 모른다고 가정하겠습니다. 그럼 이 상황을 다음과 같이 표시해 볼 수 있습니다.  

$$Y_1$$, $$Y_2$$, $$Y_3$$, ... $$Y_N$$ ~ N($$\mu$$, 1)  

모평균에 대해 모르기 때문에 샘플을 이용하여 모평균을 추정해야 하는 상황입니다. Frequentist와 Bayesian은 어떻게 이에 대해 접근할까요?  

먼저 `빈도주의자`들의 접근입니다. 빈도주의자들에게 있어 모평균은 `unknown` 값이지만 `fixed`되어 있는 값입니다. MLE를 사용하여
$$hat{\mu}$$ = $$bar{y}$$라고 해버리면 그만입니다. 데이터가 주어지면 해당 데이터들을 모두 평균낸 값을 모평균의 추정치로 생각을 할 것입니다.  

하지만 `베이지안`적인 접근은 이와는 조금 다릅니다. 베이지안에게 있어서 모평균은 `fixed`된 어떤 값이 아니라 `Random Variable`, 즉 `확률변수`입니다.
절대 fixed된 값이 아니며 변화 무쌍한 값입니다. 확률변수는 확률분포를 가질 것입니다. 데이터가 주어졌을 때 모평균을 추정하기 위해, 즉 
P($$\mu$$|$$y_1$$, ..., $$y_n$$)의 분포를 구하기 위해 사전확률분포와 샘플링 데이터의 분포를 이용합니다. 즉, 데이터와 사전확률분포를 이용해서 
사후확률분포를 추정해낼 것이고 말 그대로 '분포'를 갖게 될 것입니다. 따라서 사후확률분포에 대해 평균, 분산, 심지어는 중간값까지도 추정이 가능해집니다. 
좋은 사전확률분포를 참고했다면, 베이지안적인 접근은 분명 빈도주의자들의 접근에 비해 좋은 결과를 낳았을 것입니다. 

# 빈도주의적 접근이 잘 통하지 않는 경우?  
빈도주의적인 접근이 잘 통하지 않는 경우는 한마디로 `Likelihood Principle`을 위반할 때 입니다. 
[Likelihood Principle](https://en.wikipedia.org/wiki/Likelihood_principle)이란 
어떤 unknown 파라미터에 대해 추정할 때, 특정 분포를 따르는 확률변수로부터 관찰값들이 뽑혔다면, 모델 파라미터와 관련된 모든
증거 및 정보들이 어떤 관찰값들이냐에 상관없이 Likelihood function에 담겨있다는 것입니다. 특정 관찰값을 $$x^{*}$$라고 한다면 
수식으로 이렇게 표현해볼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?L_%7BX%5E*%7D%28%5Ctheta%29%20%5Cpropto%20P%28X%5E*%7C%5Ctheta%29%2C%20%5Ctheta%20%5Cin%20%5CTheta)  

LP는 다음 두 가지를 요구합니다. 
1. ![](https://latex.codecogs.com/gif.latex?L_1%28%5Ctheta%29%20%5Cpropto%20L_2%28%7B%5Ctheta%7D%29)  
2. $$\theta$$에 대한 결론이 우리가 사용하는 관찰값들에 의존하지 않아야 한다. (즉 결론이 같아야 한다.)  

하지만 빈도주의적 관점에서는 문제를 해결하기 위해 어떤 분포를 사용하는지에 따라 결과 자체가 달라지는 경우가 생깁니다. p-value가 0.05인 점을 
기준으로 어떤 분포를 사용했을 때는 그보다 커지고, 분포를 조금 바꿨을 뿐인데 그보다 작아질 수도 있다는 점이죠. 이런 점에서도 빈도주의적 관점에 
비해 베이지안이 썩 괜찮은 영역 같다는 생각이 드는군요.

# 주어진 데이터들에 대해 미래의 관측값이 갖게 될 확률  
새로이 주어질 관찰값을 다음과 같이 표시하겠습니다. $$\tilde{y}$$. 그리고 우리가 찾아야 할 확률 분포는 이것입니다. P($$\tilde{y}$$|y)  
이를 다음과 같이 나타내볼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Cwidetilde%7By%7D%7Cy%29%20%3D%20%5Cfrac%7BP%28%5Cwidetilde%7By%7D%2C%20y%29%7D%7BP%28y%29%7D%20%3D%20%5Cfrac%7B%5Cint%7BP%28%5Cwidetilde%7By%7D%20%2C%20y%2C%20%5Ctheta%29%7Dd%5Ctheta%7D%7BP%28y%29%7D%20%3D%20%5Cfrac%7B%5Cint%7BP%28%5Cwidetilde%7By%7D%7C%20y%2C%20%5Ctheta%29%7DP%28y%2C%20%5Ctheta%29d%5Ctheta%7D%7BP%28y%29%7D%20%3D%20%5Cint%7BP%28%5Cwidetilde%7By%7D%7Cy%2C%5Ctheta%29P%28%5Ctheta%7Cy%29%7Dd%5Ctheta)

만약 $$\tilde{y}$$와 y가 given $$\theta$$에 대해 `conditionally independent하다고 가정한다면 다음과 같이 더 간단하게 나타낼 수도 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Cwidetilde%7By%7D%7Cy%29%20%3D%20%5Cint%7BP%28%5Cwidetilde%7By%7D%7C%5Ctheta%29P%28%5Ctheta%7Cy%29%7Dd%5Ctheta)  
다음 식은 추후의 많은 증명들에서 자주 사용되게 됩니다.  

ps. 통계는 모수에 대한 추정에, 그리고 주어진 데이터를 바탕으로 미래의 값을 예측하는 것에 관심이 있습니다. 이 글에서 이 둘에 대한 개략적인 
베이지안적 접근을 다루어 보았습니다.



#### Reference
[Wikipedia Likelihood Principle](https://en.wikipedia.org/wiki/Likelihood_principle)  
[Duke University 강의 자료](http://www2.stat.duke.edu/~st118/sta732/PrincHO.pdf)
