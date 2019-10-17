---
layout: post
title:  "Bayesian-Intro"
subtitle:   "bayesian-intro"
categories: statistics
tags: bayesian
comments: true

---

베이즈 통계학이 무엇인지, 빈도주의자들과의 관점이 어떻게 다른지에 대해 정리한 글입니다. 

---

# 베이즈 통계학이란?  
`베이즈 통계학`이란(Bayesian statistics)은 `하나의 사건에서의 믿음의 정도를 확률로 나타내는 베이즈 확률론에 기반한 통계학 이론`입니다. 
'베이즈 통계학'은 '빈도주의'와 함께 통계학의 양대 산맥을 이루고 있습니다. 베이지안에 대한 설명은 항상 빈도주의적 접근과의 비교로 시작합니다.

# 베이지안(Bayesian) vs 빈도주의자(Frequentist)  

베이지안과 빈도주의자들은 확률에 대한 다른 정의를 갖고 있습니다. 먼저 빈도주의자들은, 확률을 '많은 시행이 반복되었을 때 해당 사건이 일어나는
`빈도수`'로 이해합니다. 주사위를 많이 던지면, 1이 나올 확률이 1/6이 된다는 우리의 상식과 상통합니다. 반면, 베이지안은 확률을 '해당 사건이 일어날 확률에 대한
주관적인 믿음'의 관점에서 접근합니다. 주사위를 두 번 던졌더니 두 번 다 1이 나왔다면, 우리는 1이 나올 확률을 1/6이 아니라 1로 판단할 것입니다.

베이지안을 이해하기 위해서는 `사전확률분포(Prior Distribution)`와 `사후확률분포(Posterior Distribution)`에 대한 이해가 필요합니다. 
사전확률분포란 현재의 상황에 대해 결론을 내리기 '전'의 믿음의 정도를 의미합니다. 현재의 문제에 대해 판단할 수 있는 근거 및 과거의 자료가 없는
경우도 물론 존재하지만(`non-informative prior` - 차후에 설명하겠습니다.), 해당 자료가 존재한다면 이 정보는 현재의 문제에 대해 판단하는 데에
큰 도움이 될 것입니다. 요컨대, 사전확률분포는 관측자가 관측을 하기 전에 갖고 있는 선험적 확률 분포를 의미한다고 할 수 있겠습니다.  
현재 우리가 갖고 있는 데이터와, 사전에 갖고있던 확률분포 (사전확률분포)를 동시에 고려하여 현재의 상황에 대한 분포를 얻어내면
그것이 곧 사후확률분포라고 할 수 있습니다. 이를 수식으로 표기하면 다음과 같습니다. 

![](https://latex.codecogs.com/gif.latex?P%28%5Ctheta%7Cy%29%3D%20%5Cfrac%7BP%28y_%5Ctheta%7C%5Ctheta%29P%28%5Ctheta%29%7D%7B%5Cint_%5Ctheta%7BP%28y_%5Ctheta%7C%5Ctheta%29P%28%5Ctheta%29%7Dd%5Ctheta%7D)
![][https://latex.codecogs.com/gif.latex?P%28%5Ctheta%7Cy%29%20%5Cpropto%20P%28y_%5Ctheta%7C%5Ctheta%29P%28%5Ctheta%29)

---
단,
y: 데이터  
$$\theta$$: 타겟 모수  
P(y|$$\theta$$): 우도 함수 (Likelihood function)  
P($$\theta$$): 사전확률분포  
P($$\theta$$|y): 사후확률분포

---

이 수식을 요약하자면, 베이지안 추론 (Bayesian Inference)는 곧 우도 함수와 사전확률분포에 전적으로 의존합니다. 
베이지안에서는 기존의 빈도주의자들은 고려하지 않았던 '과거의 정보'에 대해서까지 고려를 하고, 
따라서 이 과거의 정보가 좋은 정보라고 한다면 베이지안적인 접근은 빈도주의적 접근에 비해 더 좋은 결과를 낼 것입니다. 

다음 Intro2에서는, 추정에 대한 관점에서의 빈도주의와 베이지안의 차이, 빈도주의적 접근이 잘 작동하지 않는 경우, 그리고
주어진 데이터들에 대해 미래의 관찰값이 갖게 될 확률에 대해 부가적으로 설명해보도록 하겠습니다.

# Reference
[베이즈 통계학](https://ko.wikipedia.org/wiki/%EB%B2%A0%EC%9D%B4%EC%A6%88_%ED%86%B5%EA%B3%84%ED%95%99)  
[Duke 대학교 통계학 강의 교안 - Notes 9. The Likelihood Principle](http://www2.stat.duke.edu/~st118/sta732/PrincHO.pdf)  
[What exactly does it mean to and why must one update prior?](https://stats.stackexchange.com/questions/166321/what-exactly-does-it-mean-to-and-why-must-one-update-prior)
