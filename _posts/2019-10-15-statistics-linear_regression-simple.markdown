---
layout: post
title:  "Linear Regression-Simple Linear Regression-1"
subtitle:   "Linear Regression- Simple Linear Regression-1"
categories: statistics
tags: linear_regression
comments: true

---

- 단순선형회귀모델에 대한 간단한 설명, 단순선형회귀에서의 파라미터에 대한 최소 제곱 추정, 추정된 파라미터들의 성질 이 세 가지에 대해서 살펴보습니다. 

---

# 단순선형회귀모델에 대한 설명

앞선 [Intro](https://data-weirdo.github.io/statistics/2019/10/14/statistics-linear_regression-intro/)에서 에서 단순선형회귀모델은 다음과 같이 표현될 수 있다고 했습니다.

![](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cbeta_0%20&plus;%20%5Cbeta_1x%20&plus;%20%5Cvarepsilon)

이 식에서 표현된 문자들에 대해 살펴보도록 하겠습니다.

- 절편값(intercept)인 $$β_0$$, 기울기값(slope)인 $$β_1$$: `unknown` `constants`
- Error값인 ε: `random` error component
- y: `random` variable

ε은 random variable이고 이에 따라 y 또한 random variable이 되기 때문에, y는 각각의 x값들에 대해 그에 상응하는 확률분포를 갖게 됩니다.

---

추가적으로 회귀분석을 위해서는 error들이 서로 `uncorrelated`되어 있다는 가정이 필요합니다. error들이 서로 연관되어 있지 않다는 말은 곧 y값들 또한 연관되어있지 않음을 의미합니다. 

---

앞서 언급했던 $$β_0$$, $$β_1$$는 종종 `회귀계수(Regression coefficients)`라고 불리웁니다.
이 계수들은 아주 단순하지만 꽤나 유용한 해석을 제공하게 됩니다.  

$$β_1$$는 x가 한 단위 증가할 때 y의 확률분포의 평균이 얼마나 증가하는지를 의미하는 지표가 됩니다.
한편, $$β_0$$은 x가 0일 때의, y의 확률분포의 평균을 의미합니다.
만약 분석 범위에서 x가 0인 경우가 포함되지 않는다면, $$β_0$$는 사실상 해석에서 크게 신경쓸 부분 아닙니다. 

---

# 단순선형회귀에서의 파라미터에 대한 최소 제곱 추정
### Least-squares estimtion of the parameters

앞서 단순회귀식의 파라미터인 $$β_0$$, $$β_1$$는 unknown constant였습니다.
이 값들은 알려지지 않았기 때문에 샘플 데이터를 이용하여 추정하여 추정치를 얻어내어야 합니다. 그렇다면 이 두 파라미터는 어떻게 추정하여야 할까요?

## 최소 제곱 추정법

최소 제곱 추정법은 오차 제곱의 합을 최소화하는 $$β_0$$, $$β_1$$를 찾는 방법입니다.  

각 샘플들에 대하여 회귀식을 다음과 같이 적어볼 수 있습니다. 

![](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cbeta_0%20&plus;%20%5Cbeta_1x_%7Bi%7D%20&plus;%20%5Cvarepsilon_%7Bi%7D%2C%20i%3D1%2C%202%2C%20...%2C%20n)  

이 식을 오차항에 대해서 정리하면 아래와 같이 나타낼 수 있습니다. 

![](https://latex.codecogs.com/gif.latex?%5Cvarepsilon_i%20%3D%20y_i-%5Cbeta_0-%5Cbeta_1x_i)

이제 오차 제곱의 합은 아래와 같은 형태가 되는데

![](https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Cbeta_0%20-%20%5Cbeta_1x_i%29%5E2)

이 식을 최소화시킬 수 있는 $$β_0$$, $$β_1$$을 찾는 것이 곧 `최소제곱법(Method of least squares)`이라고 할 수 있겠습니다.  

두 파라미터를 찾는 방법은 간단합니다. 최고차항이 양수인 이차 방정식을 미분한 식의 x절편이 이차 방정식의 값을 가장 작게 만드는 점이라는 사실과 미분, 그리고 연립방정식을 푸는 방법 정도만 알면 됩니다. 

![](https://latex.codecogs.com/gif.latex?%5Cfrac%5Cpartial%7B%5Cpartial%5Cbeta_0%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Cbeta_0%20-%20%5Cbeta_1x_i%29%20%3D%200)
![](https://latex.codecogs.com/gif.latex?%5Cfrac%5Cpartial%7B%5Cpartial%5Cbeta_1%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Cbeta_0%20-%20%5Cbeta_1x_i%29%20%3D%200)  
위 식을 정리하면 다음과 같아집니다. 이제부터는, 파라미터들에 대해 `추정`을 하고 있기 때문에, 지금부터는 $$β_0$$과 $$β_1$$ 모두에 hat을 씌우겠습니다.  
![](https://latex.codecogs.com/gif.latex?n%5Cwidehat%7B%5Cbeta_0%7D%20&plus;%20%5Cwidehat%7B%5Cbeta_1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dx_i%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dy_i)
![](https://latex.codecogs.com/gif.latex?%5Cwidehat%7B%5Cbeta_0%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dx_i&plus;%20%5Cwidehat%7B%5Cbeta_1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dx_i%5E2%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dy_ix_i)
위의 정리된 방정식은 ```정규방정식(Least-squares normal equations)```이라고도 불립니다.  
이제 이 연립 방정식을 풀면, 결과적으로 
![](https://latex.codecogs.com/gif.latex?%5Cwidehat%5Cbeta_1%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5En%28x_i-%5Cbar%7Bx%7D%29%28y_i-%5Cbar%7By%7D%29%7D%7B%5Csum_%7Bi%3D1%7D%5En%28x_i-%5Cbar%7Bx%7D%29%5E2%7D)
![](https://latex.codecogs.com/gif.latex?%5Cwidehat%5Cbeta_0%20%3D%20%5Cbar%7By%7D%20-%20%5Cwidehat%5Cbeta_1%5Cbar%7Bx%7D)이 됩니다.

 $$\hat{β_1}$$은 편의상 다음과 같이 표기되기도 합니다.  
![](https://latex.codecogs.com/gif.latex?%5Cwidehat%5Cbeta_1%20%3D%20%5Cfrac%7BS_%7Bxy%7D%7D%7BS_%7Bxx%7D%7D)

이제, 관찰값과 추정된 직선(fitted line) 간의 차이가 바로 잔차(residual) (≠ error)가 되며, 
잔차는 모델의 적합도를 판정하거나 회귀분석의 가정이 들어맞는지 등을 판단할 때 쓰이는 중요한 역할을 하게 됩니다. 

다음은 잔차에 대한 수학적 표기입니다. 
![](https://latex.codecogs.com/gif.latex?e_i%20%3D%20y_i%20-%20%5Cwidehat%7By_i%7D%20%3D%20y_i%20-%20%28%5Cwidehat%7B%5Cbeta_0%7D&plus;%5Cwidehat%7B%5Cbeta_1%7Dx_i%29%2C%20i%3D1%2C%202%2C%20...%2C%20n)

```
오차(error)는 참 회귀선에 대한 noise이고, 잔차(residual)은 추정 회귀선에 대한 noise입니다. 오차와 잔차는 엄밀히 구분되어야 합니다. 

```
---
  우리는 최소제곱법을 사용해서 추정 단순회귀선을 얻을 수 있었습니다. 
  
![](https://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D%20%3D%20%5Cwidehat%5Cbeta_0%20&plus;%20%5Cwidehat%5Cbeta_1x)  
이것이 곧 추정 회귀식입니다. 

그런데, 이렇듯 회귀 모델을 만들고 나니, 다음과 같은 의문점이 생깁니다. 

1. 이 추정 회귀식이 데이터에 얼마나 잘 들어맞을까?
2. 이 모델이 예측을 함에 있어서 유용할까?
3. 회귀 분석의 기본 가정들 중 위배된 것이 있을까? 있다면 얼마나 큰 영향을 미칠까?  

회귀식을 추정한 후 그 모델을 실제로 사용하기 전에 이런 이슈들은 반드시 조사되어야만 합니다. 앞서 언급했듯, 잔차는 모델의 적합도를 판단하는 데에
있어서 큰 역할을 합니다. 이에 대해서는 추후에 업데이트 하도록 하겠습니다. 


---
# 추정된 파라미터들의 성질  

  단순회귀식으로부터 두 개의 파라미터 $$β_0$$과 $$β_1$$에 대해 추정하였습니다. 과연 이 추정량들은 좋은 추정량일까요? 결과적으로 말하자면 'Yes'입니다.
  
두 추정량의 평균 분산은 다음과 같습니다. (증명 생략)  
![](https://latex.codecogs.com/gif.latex?E%28%5Cwidehat%5Cbeta_1%29%20%3D%20%5Cbeta_1%20%2C%20Var%28%5Cwidehat%5Cbeta_1%29%20%3D%20%5Cfrac%7B%5Csigma%5E2%7D%7BS_%7Bxx%7D%7D)
![](https://latex.codecogs.com/gif.latex?E%28%5Cwidehat%5Cbeta_0%29%20%3D%20%5Cbeta_0%20%2C%20Var%28%5Cwidehat%5Cbeta_0%29%20%3D%20%5Csigma%5E2%28%7B%5Cfrac%7B1%7D%7Bn%7D&plus;%5Cfrac%7B%5Cbar%7Bx%7D%5E2%7D%7BS_%7Bxx%7D%7D%7D%29)  
<center> 조건 </center>
![](https://latex.codecogs.com/gif.latex?E%28%5Cvarepsilon%29%3D0%2C%20Var%28%5Cvarepsilon%29%20%3D%20%5Csigma%5E2%2C%20uncorrelated%3A%20errors)
라는 조건 하에서 두 추정량은 `Gauss-Markov Theorem`에 따라 상당히 좋은 추정량이 됩니다. $$\hat{β_0}$$은 $$β_0$$에 대한 불편추정량이 되며, 
$$\hat{β_0}$$이 아닌 다른 어떤 불편추정량을 가져오더라도 $$\hat{β_0}$$의 분산보다 작은 추정량을 찾을 수는 없습니다. 마찬가지로 
$$\hat{β_1}$$ 또한 $$β_1$$에 대한 불편추정량이 되며, $$\hat{β_1}$$이 아닌 다른 어떤 불편추정량을 가져오더라도 $$\hat{β_1}$$의 분산보다 작은 추정량을 찾을 수는 없습니다.
따라서 두 최소 제곱 추정량은 `Best Linear Unbiased Estimators` 또는 `BLUE`라고 일컬어질 수 있습니다. 최소제곱추정량이 다른 어떤 
불편추정량과 비교했을 때도 불편추정량이면서 최소분산을 가지게 됨을 설명하는 Gauss Markov Theorem은 차후에 설명하게 될 다중회귀에 또한 적용될 수 있습니다. 

### Least-Square fit의 추가적인 성질 몇 가지
![](https://latex.codecogs.com/gif.latex?1.%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_i%20-%20%5Cwidehat%7By_i%7D%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7De_i%20%3D%200)
![](https://latex.codecogs.com/gif.latex?2.%20%5Csum_%7Bi%3D1%7D%5En%20y_i%20%3D%20%5Csum_%7Bi%3D1%7D%5En%20%5Cwidehat_%7By_i%7D)
![](https://latex.codecogs.com/gif.latex?3.%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dx_ie_i%20%3D%200)
![](https://latex.codecogs.com/gif.latex?4.%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cwidehat%7By_i%7De_i%20%3D%200)
+ 최소제곱 회귀직선은 항상 centroid를 지난다. ($$\bar{x}$$, $$\bar{y}$$)


