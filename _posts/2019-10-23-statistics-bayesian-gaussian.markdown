---
layout: post
title:  "Bayesian-정규분포 샘플"
subtitle:   "bayesian-gaussian"
categories: statistics
tags: bayesian
comments: true

---

- 정규분포를 따르는 샘플에 대한 베이지안 통계 분석입니다. 

---  

정규분포를 따르는 샘플에 대해 베이지안 사후확률을 추정하려 합니다. 크게 세 가지 경우로 나눠 볼 수 있겠습니다. 
```  
1. 정규분포를 따르는 샘플의 모분산은 알지만 모평균을 모르는 경우  
2. 정규분포를 따르는 샘플의 모평균은 알지만 모분산을 모르는 경우  
3. 정규분포를 따르는 샘플의 모평균, 모분산 모두를 모르는 경우  
```  

- Case1.  
$$y_1, y_2, ... , y_n | \theta $$ ~iid $$N(\theta, \sigma^2)$$  
$$\theta$$ ~ N($$\mu_o$$, $$\tau_0^2$$)  

![](https://latex.codecogs.com/gif.latex?P%28%24%24%5Ctheta%7Cy_1%2C%20y_2%2C%20...%2C%20y_n%24%24%29%20%24%24%5Cpropto%24%24%20%24%24e%5E%7B-%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Ctheta%29%5E2%7D%7B2%5Csigma%5E2%7D%7De%5E%7B-%5Cfrac%7B%28%5Ctheta-%5Cmu_0%29%5E2%7D%7B2%5Ctau_0%5E2%7D%7D)  


$$\theta|y_0, y_1, ..., y_n$$ ~ N($$\frac{\frac{n\bar{y}}{\sigma^2} + \frac{\mu_0}{\tau_0^2}}{\frac{n}{\sigma^2} + \frac{1}{\tau_0^2}}$$,$$\frac{1}{\frac{n}{\sigma^2} + \frac{1}{\tau_0^2}}$$)  
편의상 사후확률분포의 평균을 $$\mu_1$$, 분산을 $$\tau_1^2$$라 하겠습니다. 
사후확률분포의 평균 역시 샘플의 평균과 사전확률분포의 평균의 가중평균입니다. `Precision`은 분산의 역수를 의미하는데, 샘플의 precision이 사전 precision보다 
훨씬 크다면 사후평균은 샘플의 평균과 가까워질 것이고, 그 반대라면 사전 평균과 가까워질 것입니다. 참고로, 
일반적으로는 `Posterior precision = sample precision + prior`의 관계가 성립하지 않으나, 가우시안 분포에서는 이 관계가 성립합니다.  

아울러 샘플의 크기가 매우 크거나, $$\tau_0^2$$의 크기가 매우 크다면(이 경우 Gaussian 사전 분포는 거의 non-informative가 되버림) 사후평균은 
샘플의 평균과 흡사해 집니다.  

그렇다면 주어진 샘플 데이터에 대해 미래에 관찰될 관찰값의 분포는 어떻게 될까요?  
결론적으로는, 다음과 같은 분포를 따르게 됩니다. 

$$\tilde{y}|y_1, y_2, ..., y_n$$ ~ N($$\mu_1$$, $$\tau_1^2$$ + $$\sigma^2$$)  

역시 샘플의 크기가 굉장히 커지면, 사후확률분포는 $$N(\bar{y}, 0)$$ 분포에 가까워지며 샘플 데이터의 평균을 기준으로 거의 `degenerate distribution`이 되어버릴 것이고, $$\tilde{y}|y_1, y_2, ..., y_n$$는  $$N(\bar{y}, \sigma^2)$$ 분포에 굉장히 가까워질 것입니다. 
또한 사후확률분포가 근사적으로 $$N(\bar{y}, \sigma^2/n)$$을 따른다고 볼 수도 있습니다.  

