---
layout: post
title:  "선형회귀분석에 대한 간단한 소개"
subtitle:   "Linear Regression-Intro"
categories: statistics
tags: linear_regression
comments: true

---

- DOUGLAS C. MONTGOMERY 저 Introduction to Linear Regression Analysis의 Intro에 해당하는 부분을 일부 요약하였습니다.  

---
한 개의 regressor variable만을 포함한 단순회귀모델의 방정식은 이러합니다.

![](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cbeta_%7B0%7D%20&plus;%20%5Cbeta_%7B1%7Dx%20&plus;%20%5Cvarepsilon)  

이 때, regressor variable인 x의 값들을 '고정'하고, 그에 상응하는 response y를 관찰한다라고 한다면
이 단순 회귀식의 평균 및 분산을 다음과 같이 표현할 수 있습니다.

![평균](https://latex.codecogs.com/gif.latex?E%28y%7Cx%29%20%3D%20%5Cmu_%7By%7Cx%7D%20%3D%20E%28%5Cbeta_%7B0%7D%20&plus;%20%5Cbeta_%7B1%7Dx%20&plus;%20%5Cvarepsilon%29%20%3D%20%5Cbeta_%7B0%7D%20&plus;%20%5Cbeta_%7B1%7Dx)  

![분산](https://latex.codecogs.com/gif.latex?Var%28y%7Cx%29%20%3D%20%5Csigma%5E2_%7By%7Cx%7D%20%3D%20Var%28%5Cbeta_%7B0%7D%20&plus;%20%5Cbeta_%7B1%7Dx%20&plus;%20%5Cvarepsilon%29%20%3D%20%5Csigma%5E2)  

이 때, True regression model 식은 아래와 같은데,  
  
![](https://latex.codecogs.com/gif.latex?%5Cmu_%28y%7Cx%29%20%3D%20%5Cbeta_%7B0%7D%20&plus;%20%5Cbeta_%7B1%7Dx)  

이것은 곧 평균값들의 선이고, 이 말인 즉슨 어떤 x 값에서든 회귀 직선의 높이는 그 x값에서의 기대(expected) y값이 됩니다.

뿐만 아니라, 특정 x값에 대한 y의 분산은 결국 ε의 분산에 의해서 결정되게 되는데
이 말인 즉슨 각각의 x값들에서 y의 값들은 분포를 가지게 되며, 이 분포의 분산은 모든 x 값들에 대해서 동일합니다.
이 때, 분산이 커지면 y값들은 회귀 직선들로부터 많이 떨어져서 분포하게 되고, 작아지면 회귀직선에 가까이 붙는 양상을 띠게 될 것입니다.

---

일반적으로 회귀식은 그 회귀식을 도출하게 된 영역 한에서만 valid합니다. 
extrapolation이 잘 안된다고 할 수 있겠군요. 

--- 

위의 단순회귀식과 달리 regresson variable들이 2개 이상이 되면 곧 Multiple linear regression model이 됩니다.
다음과 같은 형태가 되겠네요.

![](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cbeta_0%20&plus;%20%5Cbeta_1x_%7B1%7D%20&plus;%20%5Cbeta_2x_%7B2%7D%20&plus;%20...%20&plus;%20%5Cbeta_kx_k%20&plus;%20%5Cvarepsilon)  

한 가지 중요한 것은, 이 모델을 두고 linear이라고 표현하는 것은 이 회귀식이 파라미터인 Beta 값들에 대해 linear하다는 것입니다. 
따라서, x 변수들이 nonlinear한 형태로 주어지더라도, 회귀식은 여전히 Beta 값들에 대해 linear하기 때문에 여전히 'linear' regression으로 칭해질 수 있게 되는 겁니다.

---

회귀 분석은 ```Iterative procedure``` 입니다. 
모델을 만들어서 이 모델이 적합한지를 판단하고, 그렇지 못하다면 다시 개선된 모델을 만들고 판단하고를 반복하는 과정입니다. 

--- 

회귀 모델은 결코 Cause-and-effect 관계를 나타내지 않습니다. 인과관계가 성립되기 위해서는, regressor와 response 간의 해당 관계가 샘플 데이터 외부에서 정의될 수 있어야 합니다. 예를 들어, 'A는 B를 초래한다'는 명제 자체가 하나의 이론으로 이미 정립이 되어있어야겠지요. 

---

회귀 분석은 더 고차원적인 분석을 위한 한 부분입니다. 회귀 방정식 그 자체가 목적이 아니라, 문제 해결을 위한 인사이트를 얻는 등의 용도로 사용될 수 있습니다. 
