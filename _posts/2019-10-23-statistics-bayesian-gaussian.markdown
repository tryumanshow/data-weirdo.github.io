---
layout: post
title:  "Bayesian-정규분포 및 포아송 샘플"
subtitle:   "bayesian-gaussian&poisson"
categories: statistics
tags: bayesian
comments: true

---

- 정규분포 및 포아송분포를 따르는 샘플에 대한 분포 입니다. 

---  

# 정규분포  

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

![](https://latex.codecogs.com/gif.latex?%24%24%5Ctilde%7By%7D%24%24%7C%24%24y_1%2C%20y_2%2C%20...%2C%20y_n%24%24%20%7E%20N%28%24%24%5Cmu_1%24%24%2C%20%24%24%5Ctau_1%5E2%24%24%20&plus;%20%24%24%5Csigma%5E2%24%24%29)

역시 샘플의 크기가 굉장히 커지면, 사후확률분포는 $$N(\bar{y}, 0)$$ 분포에 가까워지며 샘플 데이터의 평균을 기준으로 거의 `degenerate distribution`이 되어버릴 것이고, $$\tilde{y}|y_1, y_2, ..., y_n$$는  $$N(\bar{y}, \sigma^2)$$ 분포에 굉장히 가까워질 것입니다. 
또한 사후확률분포가 근사적으로 $$N(\bar{y}, \sigma^2/n)$$을 따른다고 볼 수도 있습니다.  

결과적으로는 Conjugate Distribution을 사용했습니다.  

---  

- Case2.  
Case2에서 역시 Conjugate Prior을 고려해보겠습니다. 모평균은 알지만 모분산을 모르는 경우, Conjugate Prior로 `Inverse Gamma density`를 고려하겠습니다. 다음과 같이 써볼 수 있겠습니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Csigma%5E2%29%20%5Cpropto%20%28%5Csigma%5E2%29%5E%7B-%5Cfrac%7Ba%7D%7B2%7D-1%7De%5E%7B-%5Cfrac%7Bb%7D%7B2%5Csigma%5E2%7D%7D)  
$$P(\sigma^2)$$ ~ IG($$\frac{a}{2}$$, $$\frac{b}{2}$$)  

그렇다면 다음 관계에 의해,  
![](https://latex.codecogs.com/gif.latex?P%28%5Csigma%5E2%7Cy_1%2C%20y_2%2C%20...%2C%20y_n%29%20%5Cpropto%20%28%5Csigma%5E2%29%5E%7B-%5Cfrac%7Bn%7D%7B2%7D%7De%5E%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5Csum_%7Bi%7D%28y_i-%5Ctheta%29%5E2%7D%28%5Csigma%5E2%29%5E%7B-%5Cfrac%7Ba%7D%7B2%7D-1%7De%5E%7B-%5Cfrac%7Bb%7D%7B2%5Csigma%5E2%7D%7D)  
![](https://latex.codecogs.com/gif.latex?%3D%28%5Csigma%5E2%29%5E%7B-%5Cfrac%7B1%7D%7B2%7D%28n&plus;a%29-1%7De%5E%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5B%5Csum_i%28y_i-%5Ctheta%29%5E2&plus;b%5D%7D)  
사후분포는 다음 분포를 따르게 됩니다.  
![](https://latex.codecogs.com/gif.latex?IG%28%5Cfrac%7B1%7D%7B2%7D%28n&plus;a%29%2C%20%5Cfrac%7B1%7D%7B2%7D%5B%5Csum_i%28y_i-%5Ctheta%29%5E2&plus;b%5D%29)  

Inverse Gamma 분포에 대해 도움이 되는 설명이 있어 정리해보았습니다.  
```
The inverse gamma distribution is a two-parameter family of continuous probability distributions on a positive line  
which is teh distribution of the reciprocal of a variable distributed according to the Gamma distribution.  ...  
Perhaps the chief use of the inverse gamma distribution is in Bayesian statistics, where the distribution arises as  
the marginal posterior distribution for the unknown variance of a normal distribution, if an uninformative prior is used  
and as an analytically tractable conjugate prior, if an informative prior is required.  
```  

- Case3.  
이제 Case3에서는 두 개의 파라미터에 대해 모두 추정을 해야하는 상황입니다. 이 경우에는 사후확률분포를 구하기 위해 사전 확률 분포 두 개가 필요합니다.  
우선 샘플 데이터에 대한 분포는 다음과 같이 나타낼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28y_1%2C%20...%2C%20y_n%7C%5Ctheta%2C%20%5Csigma%5E2%29%20%3D%20%5Cfrac%7B1%7D%7B%282%5Cpi%5Csigma%5E2%29%5Cfrac%7Bn%7D%7B2%7D%7De%5E%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5Csum_i%28y_i-%5Ctheta%29%5E2%7D)  

이 때, $$\sigma^{-2}$$를 r로 나타내면 위의 식을 이렇게 나타낼 수도 있겠습니다.  
![](https://latex.codecogs.com/gif.latex?P%28y%7C%5Ctheta%2C%20r%29%20%5Cpropto%20r%5E%7B%5Cfrac%7Bn%7D%7B2%7D%7Dexp%5E%7B-%5Cfrac%7Br%7D%7B2%7D%5Csum_i%7B%28y_i-%5Ctheta%29%7D%5E2%7D)  

구해야 할 사후확률분포는 P($$\theta$$, r | y)인데 이 분포는 다음과 같이 정리해볼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%2Cr%7Cy%29%20%5Cpropto%20P%28y%7C%5Ctheta%2C%20r%29P%28%5Ctheta%2C%20r%29)  
![](https://latex.codecogs.com/gif.latex?%3DP%28y%7C%5Ctheta%2C%20r%29P%28%5Ctheta%7Cr%29P%28r%29)  

따라서 사후확률분포를 얻어내기 위해 $$P(\theta|r)$$와 P(r)에 해당하는 분포가 필요합니다. 각 분포는 다음과 같이 나타낼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28r%29%20%5Cpropto%20r%5E%7B%5Cfrac%7Ba%7D%7B2%7D-1%7Dexp%5E%7B-%5Cfrac%7Bb%7D%7B2%7Dr%7D%2C%20r%5Csim%20Gamma%28%5Cfrac%7Ba%7D%7B2%7D%2C%20%5Cfrac%7Bb%7D%7B2%7D%29)  
![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%7Cr%29%20%5Cpropto%20r%5E%7B%5Cfrac%7B1%7D%7B2%7D%7Dexp%5E%7B-%5Cfrac%7B%5Clambda%20r%7D%7B2%7D%28%5Ctheta-%5Cmu%29%5E2%7D%2C%20%5Ctheta%7Cr%20%5Csim%20N%28%5Cmu%2C%20%5Cfrac%7B1%7D%7B%5Clambda%20r%7D%29)  

이제 사후확률분포를 계산하기 위해 필요한 사전분포와 샘플의 분포를 갖게 되었으니, 이 식을 잘 계산하면 다음과 같은 
사후확률분포 두 가지를 얻게 됩니다.  

![](https://latex.codecogs.com/gif.latex?%5Ctheta%7Cr%2Cy%20%5Csim%20N%28%5Cfrac%7Bn%5Cbar%7By%7D&plus;%5Clambda%5Cmu%7D%7Bn&plus;%5Clambda%7D%2C%20%5Cfrac%7B1%7D%7Br%28n&plus;%5Clambda%29%7D%29)  
![](https://latex.codecogs.com/gif.latex?r%7Cy%20%5Csim%20Gamma%28%5Cfrac%7Bn&plus;a%7D%7B2%7D%2C%20%5Cfrac%7B%5B%5Csum_i%28y_i-%5Cbar%7By%7D%29%5E2%20&plus;%20%5Cfrac%7Bn%5Clambda%7D%7Bn&plus;%5Clambda%7D%28%5Cbar%7By%7D-%5Cmu%29%5E2&plus;b%5D%7D%7B2%7D%29)  

굉장히 복잡하지만 정규분포를 따르는 샘플에 대해 사후확률분포를 추정해보았습니다.  

---  

# 포아송분포  

이번엔 포아송분포를 따르는 샘플에 대한 베이지안 추정을 해보겠습니다.  
$$y_1, y_2, ..., y_n$$ 이 Poisson($$\theta$$) 분포를 따른다고 해보겠습니다. (단, $$\theta$$ > 0)  

그렇다면 샘플의 분포는 다음과 같이 나타낼 수 있습니다.  
![](https://latex.codecogs.com/gif.latex?P%28y_1%2C%20...%2C%20y_n%20%7C%20%5Ctheta%29%20%3D%20%5Cfrac%7Be%5E%7B-n%5Ctheta%7D%5Ctheta%5E%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7By_i%7D%7D%7D%7B%5Cprod_i%5En%20y_i%21%7D)  

이 때 사전확률분포를 Gamma 분포로 잡아준다면, 사후확률분포 또한 Gamma 분포가 되어 Conjugate 분포가 됩니다.  

사전확률분포는 다음과 같이 나타낼 수 있습니다. 
![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%29%20%3D%20%5Cfrac%7B%5Ctheta%5E%7Ba-1%7Db%5Ea%7D%7B%5CGamma%20%28a%29%7De%5E%7B-b%5Ctheta%7D%20%5Csim%20Gamma%28a%2C%20b%29)    

이제 사후확률분포를 계산하면, 다음과 같이 proportional한 형태로 나타낼 수 있고,  
![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%7Cy_1%2C%20...%2C%20y_n%29%20%5Cpropto%20e%5E%7B-%28n&plus;b%29%5Ctheta%7D%5Ctheta%5E%7B%5Csum_%7Bi%3D1%7D%5En%7By_i&plus;a-1%7D%7D)  
따라서 사후확률분포는 Gamma($$\sum_{i=1}^{n}{y_i}$$ + a, n+b) 분포를 따르게 됩니다.  
마찬가지로 포아송분포를 따르는 데이터로부터 얻어낸, 감마분포를 따르는 사후확률분포는 샘플의 평균과 사전확률 평균의 가중평균이 됩니다.  

이상 정규분포 및 포아송분포를 따르는 샘플에 대한 사후분포추정이었습니다. 
