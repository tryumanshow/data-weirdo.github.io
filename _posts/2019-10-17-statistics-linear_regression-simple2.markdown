---
layout: post
title:  "Linear Regression-Simple Linear Regression-2"
subtitle:   "linear regression-simple linear regression-2"
categories: statistics
tags: linear_regression
comments: true
---
오차의 분산($$σ^2$$)을 추정하는 방법에 대한 글입니다.

---

앞서 [Simple Linear Regression-1](https://data-weirdo.github.io/statistics/2019/10/15/statistics-linear_regression-simple/)에서는
$$β_0$$과 $$β_1$$에 대하여 추정해보았습니다. 이와 마찬가지로 $$σ^2$$ 값에 대해서도 추정해보도록 하겠습니다. 

우선, 결론적으로 말하자면 아래의 수식이 $$σ^2$$ 에 대한 추정치가 됩니다.  
![](https://latex.codecogs.com/gif.latex?%5Cwidehat%5Csigma%5E2%20%3D%20%5Cfrac%7BSS_%7BRes%7D%7D%7Bn-2%7D%20%3D%20MS_%7BRes%7D)  
   
사고의 과정은 다음과 같습니다. 
1. $$σ^2$$에 대한 추정은 `오차 제곱합(Error sum of squares)`, 또는 `잔차 제곱합(Residual sum of squares)`으로 부터 얻어집니다.

![](https://latex.codecogs.com/gif.latex?SS_%7BRes%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7De_i%5E2%20%3D%20%5Csum_%7Bi%3D1%7D%5En%28y_i-%5Cwidehat%7By_i%7D%29%5E2)
![](https://latex.codecogs.com/gif.latex?SS_%7BRes%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5En%28y_i-%28%5Cwidehat%7B%5Cbeta_0%7D%20&plus;%20%5Cwidehat%7B%5Cbeta_1%7Dx_i%29%29%5E2)
![](https://latex.codecogs.com/gif.latex?SS_%7BRes%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5Eny_i%5E2%20-%20n%5Cbar%7By%7D%5E2%20-%20%5Cwidehat%7B%5Cbeta_1%7DS_%7Bxy%7D_)
![](https://latex.codecogs.com/gif.latex?SS_%7BRes%7D%20%3D%20%5CSS_%7BT%7D%20-%20%5Cwidehat%7B%5Cbeta_1%7DS_%7Bxy%7D)

이 때, $$β_0$$과 $$β_1$$을 추정했기 때문에 결국 오차제곱합의 자유도는 전체 관찰값 개수에서 모수의 개수를 뺀 `n-2`가 됩니다.


![](https://latex.codecogs.com/gif.latex?%5Cwidehat%7B%5Csigma%5E2%7D%20%3D%20%5Cfrac%7BSS_%7BRes%7D%7D%7Bn-2%7D%20%3D%20MS_%7BRes%7D)  
$$MS_{Res}$$는 `Residual Mean Square`라 불리고, 이 값에 root를 씌운 값인 $$\sqrt{MS_{Res}}$$ `standard error of regression`이라고도 불립니다. 
$$MS_{Res}$$ 값은 $$σ^2$$의 `불편추정량`됩니다. 

---

$$MS_{Res}$$ 값은 $$SS_{Res}$$ 값에 의존하고, (왜냐하면, $$SS_{Res}$$ = $$MS_{Res}$$ * (n-2)이기 때문에) 따라서 회귀 모델의 가정을 위반할 시, 
$$MS_{Res}$$는 결코 $$σ^2$$에 대한 좋은 추정량이 되지 못할 것입니다. 그래서, $$MS_{Res}$$는 $$σ^2$$에 대한 `model-dependent`한 추정량이 됩니다. 
