---
layout: post
title:  "사전분포(Prior Distribution)의 종류와 이항분포를 따르는 샘플"
subtitle:   "bayesian-prior_and_binom"
categories: statistics
tags: bayesian
comments: true

---


- 사전분포를 설정하는 몇 가지 방법과, 이항분포를 따르는 샘플에 대한 베이지안 분석에 관한 글입니다.

---  

앞서 사후확률분포와 사전확률분포, 그리고 샘플 간의 관계를 다음과 같이 정의해보았습니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%7Cy%29%20%5Cpropto%20P%28y%7C%5Ctheta%29P%28%5Ctheta%29)  

  만약 $$P(y|\theta)$$가 학생들의 신장에 대한 데이터라면, transformation을 거쳐서 정규분포를 고려해보는 등의 액션을 취해볼 수 있습니다. 
하지만 사전확률분포를 선택하는 일은 샘플의 분포에 대해 생각해보는 일에 비해 쉽지 않습니다. 이번 글에서는 '사전확률분포를 어떻게 설정해야 할 
것인가?'에 대해 포스팅해보려고 합니다.  

# 사전확률분포를 어떻게 설정할 것인가?  
  사전확률분포를 설정하는 방법에는 여러가지가 있습니다. Conjugate Prior, Noninformative Prior(Laplace Prior, Jeffrey Prior), 
Refence Prior, Matching Prior 등을 고려해보는 것입니다. 이번 글에서는 `Conjugate Prior`와 `Laplace Prior`에 대해서만  
설명해보도록 하고, 차후에 `Jefferey Prior`에 대해서도 설명 및 적용을 해보도록 하겠습니다.

## Conjugate Prior가 무엇인가?  
베이지안 이론에서 사후확률분포가 사전확률분포와 같은 확률분포족(same probability distribution family)에 속한다면, 
이 때 사전확률분포와 사후확률분포는 `Conjugate Distribution`이 됩니다. 또 이 때의 사전 확률 분포를 `Conjugate Prior`라고 합니다. 
한글로는 켤레사전분포 라고도 부릅니다.  

## Laplace Prior는 또 무엇이지?  
우선 라플라스 사전확률분포를 설명하기 전에, Noninformative Distribution(이하 ND)에 대해 먼저 설명하겠습니다. ND란 정보가 극도로 불충분 할 때 
사용하는 사전분포입니다. 참고할만한 자료가 거의 없는 상황에서 사용되는 분포이죠. 라플라스 분포는 ND의 한 케이스입니다. 어떤 분포를 따른다고 
상정하지 않고 일단은 공평하게 모든 가능성을 열어둔 분포입니다. 라플라스 사전확률분포는 균등분포(Uniform Distribution)을 분포로 사용합니다.  

# 샘플이 이항분포를 따를 때, Conjugate Prior와 Laplace Prior는 어떻게 적용될 수 있을 것인가?  
예를 들어, n을 2019년 10월에 태어난 신생아의 수, y를 2019년 10월에 태어난 남자 아이의 수, 그리고 $$\theta$$를 남자아이가 태어날 
확률이라고 해보겠습니다.  

그렇다면, 우리는 $$y|\theta$$가 이항분포를 따른다고 생각할 수 있고, 그렇다면 다음과 같이 식을 나타낼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28y%7C%5Ctheta%29%20%3D%20%5Cbinom%7Bn%7D%7By%7D%5Ctheta%5Ey%281-%5Ctheta%29%5E%7Bn-y%7D%2C%20y%3D0%2C%201%2C%20...%2C%20n)  

이로부터 사후확률분포를 얻어내기 위해, Conjugate Prior와 Laplace Prior를 고려해보겠습니다. 단, 증명은 생략했습니다. 
차후에 필요하신 분이 계시다면 추가로 증명까지 업데이트 하도록 하겠습니다.  

## Laplace Prior  
먼저 라플라스 분포입니다. 0과 1 사이의 값인 $$\theta$$에 대한 사전확률분포로 0과 1 사이의 값을 취하는 균등분포를 생각해볼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%29%20%5Csim%20Uniform%5B0%2C1%5D)  
이를 이항분포를 따르는 샘플데이터와 함께 고려하면 베이즈정리에 의해 다음과 같은 사후확률분포를 얻을 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%7Cy%29%20%5Cpropto%20%5Ctheta%5Ey%281-%5Ctheta%29%5E%7Bn-y%7D)  
Proportional한 형태로 나타낼 것이 아니라 실제로 베이즈 정리에 따라 계산을 해보면 $$P(\theta|y)$$는 Beta(y+1, n-y+1)분포를 따르게 됩니다. 

## Conjugate Prior  
켤레사전분포 입니다. 사전확률분포가 $$Beta(\alpha, \beta)$$ 분포를 따른다고 하겠습니다. 사전확률분포의 
[커널](https://en.wikipedia.org/wiki/Kernel_(statistics))로 다음과 같이 표현하기도 합니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%29%20%5Cpropto%20%5Ctheta%5E%7B%5Calpha-1%7D%281-%5Ctheta%29%5E%7B%5Cbeta-1%7D) 
이 역시 이항분포를 따르는 샘플데이터와 함께 고려하면 베이즈정리에 의해 다음과 같은 사후확률분포를 얻을 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%7Cy%29%20%5Cpropto%20%5Ctheta%5E%7By&plus;%5Calpha-1%7D%281-%5Ctheta%29%5E%7Bn-y&plus;%5Cbeta-1%7D%29)  
이 역시 실제로 베이즈 정리에 따라 계산을 해보면 $$P(\theta|y)$$는 $$Beta(y+\alpha, n-y+\beta)$$ 분포를 따릅니다. 사전확률분포도, 사후확률분포도 
모두 베타분포를 따르고 있으니 이 때의 $$P(\theta)$$는 켤레사전분포입니다.  

이항분포 샘플과 Conjugate Prior인 $$Beta(\alpha, \beta)$$ 사전분포에 대하여 사후확률분포의 평균을 다음과 같이 나타낼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?E%28%5Ctheta%7Cy%29%20%3D%20%5Cfrac%7By&plus;%5Calpha%7D%7Bn&plus;%5Calpha&plus;%5Cbeta%7D)
이 식은 다음과 같이 분리해볼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?%3D%20%5Cfrac%7Bn%7D%7Bn&plus;%5Calpha&plus;%5Cbeta%7D%5Cfrac%7By%7D%7Bn%7D%20&plus;%20%5Cfrac%7B%5Calpha&plus;%5Cbeta%7D%7Bn&plus;%5Calpha&plus;%5Cbeta%7D%5Cfrac%7B%5Calpha%7D%7B%5Calpha&plus;%5Cbeta%7D)  
$$\frac{y}{n}$$이 표본평균(sample mean), $$\frac{\alpha}{\alpha+\beta}$$이 prior mean임을 감안하면, posterior mean을 
표본 평균과 사전확률분포의 평균의 가중평균(weighted average)이라고 볼 수 있겠습니다. 추가적으로 n이 $$\alpha+\beta$$보다 상당히 크다면, 
사후확률분포의 평균은 표본평균 쪽으로 기울게 될 것이고, 그 반대의 경우라면 사후확률분포의 평균이 사전확률분포의 평균쪽으로 기울게 될 것입니다. 
요컨대, 샘플의 수가 많다면 Posterior mean이 MLE쪽으로 기울게 되겠군요.  

이번에는 분산입니다. 사후확률분포의 분산은 다음과 같이 나타낼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?var%28%5Ctheta%7Cy%29%20%3D%20%5Cfrac%7B%28y&plus;%5Calpha%29%28n-y&plus;%5Cbeta%29%7D%7B%28n&plus;%5Calpha&plus;%5Cbeta%29%5E2%28n&plus;%5Calpha&plus;%5Cbeta&plus;1%29%7D%20%3D%20%5Cfrac%7BE%28%5Ctheta%7Cy%29%281-E%28%5Ctheta%7Cy%29%29%7D%7Bn&plus;%5Calpha&plus;%5Cbeta&plus;1%7D)  
y와 n-y 모두가 굉장히 크다면, 사후확률분포의 분산은 MLE의 분산의 추정값이 됩니다. 이해를 돕기 위해 수식을 추가해 둡니다.  
![](https://latex.codecogs.com/gif.latex?Var%28%5Cwidehat%7B%5Ctheta%7D%29%20%3D%20Var%28%5Cbar%7By%7D%29%3DVar%28%5Cfrac%7B1%7D%7Bn%7D%5Csum%20%7By_i%7D%29%3D%5Cfrac%7B1%7D%7Bn%5E2%7Dn%5Ctheta%281-%5Ctheta%29%3D%5Cfrac%7B%5Ctheta%281-%5Ctheta%29%7D%7Bn%7D)
![](https://latex.codecogs.com/gif.latex?%5Cwidehat%7BVar%28%5Cwidehat%7B%5Ctheta%7D%29%7D%20%3D%20%5Cfrac%7B%5Cwidehat%7B%5Ctheta%7D%281-%5Cwidehat%7B%5Ctheta%7D%29%7D%7Bn%7D%20%3D%20%5Cfrac%7B%5Cfrac%7By%7D%7Bn%7D%281-%5Cfrac%7By%7D%7Bn%7D%29%7D%7Bn%7D)  

그렇다면 예측값에 대해서는 어떨까요? 이는 수식으로 다음과 같이 표현해 볼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Ctilde%7By%7D%3D1%7Cy%29%20%3D%20%5Cint_%7B0%7D%5E%7B1%7DP%28%5Ctilde%7By%7D%3D1%7C%5Ctheta%2C%20y%29P%28%5Ctheta%7Cy%29d%5Ctheta%20%3D%20%5Cint_%7B0%7D%5E%7B1%7D%5Ctheta%20P%28%5Ctheta%7Cy%29d%5Ctheta)  
![](https://latex.codecogs.com/gif.latex?%3D%20E%28%5Ctheta%7Cy%29%20%3D%20%5Cfrac%7By&plus;%5Calpha%7D%7Bn&plus;%5Calpha&plus;%5Cbeta%7D)  

Conjugate prior을 사용함으로써 Posterior distribution의 분포를 알기 때문에 percentile 값들을 쉽게 찾을 수 있습니다. 자고로,  
뒤이어 나중에 소개할 MCMC 방법을 사용해서도 percentile을 찾을 수 있습니다.  

다음에는 정규분포를 따르는 샘플에 대해 똑같은 과정을 적용해보도록 하겠습니다. 

