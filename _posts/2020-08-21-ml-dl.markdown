---
layout: post
title:  "범주형 변수(Categorical Variable) 인코딩하기 #1"
subtitle:   "preprocessing-범주형 변수 다루기"
categories: ml
tags: preprocessing
comments: true
---

- 범주형 데이터들을 전처리하는 방법과, 변수의 종류에 따른 전처리 방법이 차이에 대해 간단하게 정리해 보았습니다.

## 범주형 자료

범주형 자료(Categorical Data)는 수치형 자료(Numerical Data)와 대비되는 개념으로 `수치로서 측정할 수 없는 데이터`입니다.

범주형 자료는 범주 간의 순서를 따질 수 있는지의 여부에 따라 명목형(nominal) 데이터와 순서형 데이터(ordinal)로 나누어집니다.
예를 들어, 사람의 혈액형은 A, B, O, AB와 같이 나누어지지만 혈액형 간에 순서가 없기 때문에 명목형 변수입니다.
반면, 학점은 A, B, C, D, F(+는 제외하겠습니다.)와 같이 나누어지지만, F에서 A로 갈수록 이는 곧 학점이 높아지는 것을 의미하기 때문에, 범주들 간에는 분명 순서가 존재합니다. 따라서 학점은 순서형 변수라고 볼 수 있겠네요.

### 범주형 자료에 대한 전처리가 왜 필요한가?

그렇다면, 기계학습을 진행함에 있어서 왜 범주형 데이터에 대한 전처리가 필요한 것일까요?
이것은 대부분의 머신러닝 모델이 전처리 과정을 거치지 않고서는 바로 범주형 데이터를 처리할 수 없기 때문입니다. 

### 범주형 자료 전처리 방법?

범주형 자료를 전처리하는 방법으로는 대표적으로 두 가지가 있습니다. Label Encoding과 One-hot Encoding이 바로 그것입니다. 
사이킷런에서는 각각을 구현하는 라이브러리로 LabelEncoder와 OneHotEncoder를 지원합니다. 

두 라이브러리를 설명하기 위해 간단한(말도 안되는?) 상황을 설정해보겠습니다.
네 개의 데이터를 이용하여 성별, 나이, 고향, 학점을 통해 해당 학생들의 취업 여부를 예측하려는 상황입니다.

독립변수인가 종속변수인가에 따라서 Categorical Data를 처리하는 방식도 조금은 차이가 있기 때문에 우선은 독립변수만을 갖고 전처리를 해보겠습니다. 

![](http://drive.google.com/uc?export=view&id=1xLf28q9SSoDtFj00gmxzm991G8bTHQzF)

* LabelEncoder

LabelEncoder를 사용하면 각 변수별로 0부터 (Unique한 값들의 개수 - 1) 까지의 숫자로 값들이 변형됩니다.
예를 들어, Female가 Male은 각각 0과 1로 변형되었습니다. 

![](http://drive.google.com/uc?export=view&id=1wieDjYuUUbq_3YTsTG39GcaTrmmAZMmX)

하지만, LabelEncoder로 변형된 결과를 그대로 기계학습의 모델링에 사용하기에는 문제가 있습니다.
각 변수 내에서 변형된 숫자의 값이 클수록 그 값을 실제로 큰 값으로 인식해버린다는 점입니다.
예를 들어, Female은 0이고 Male은 1이었기에, 실제로 모델은 Male을 더 큰 값으로 인식하게 되어 모델은 Male에 더 많은 가중치를 주게 될 것입니다.
하지만 이것은 우리가 원하는 상황이 아닐 것이고 Female과 Male이 동등하게 평가받기를 원할 것입니다.

다행히도, OneHotEncoder는 이 문제를 보완해줍니다. 

* OneHotEncoder

OneHotEncoder는 더미변수의 개념을 차용합니다. 각 변수마다 unique한 값들의 개수만큼 1과 0으로 구성된 컬럼들을 새로만듭니다.

이전에는 OneHotEncoder 내의 categorical_features 파라미터를 이용하면 되었지만, 버전이 업데이트 되면서 deprecated 되어 더이상은 사용할 수 없는 기능이 되었습니다. 대신 ColumnTransformer 를 사용하면 같은 기능을 구현할 수 있다고는 합니다.

```
Deprecated since version 0.20: The categorical_features keyword was deprecated in version 0.20 and will be removed in 0.22. You can use the ColumnTransformer instead.
```
하지만 pandas에서 제공하는 get_dummies 함수를 사용하면 보다 편하게 onehot encoding을 할 수 있습니다. 

![](http://drive.google.com/uc?export=view&id=1D05jDTbv_GSuCfJBRqxdUxRe6zRYR5H2)

단, OneHotEncoding도 변수의 수, 범주의 개수가 많아질수록 학습 속도가 저하된다고 하니 주의할 필요가 있겠습니다.

### 종속 변수는 어떻게 encoding을 해주어야 할까?

 대부분의 경우 종속 변수에는 LabelEncoding을 해주는 것으로 충분합니다.
종속 변수에까지 OneHotEncoding을 해주면 출력값의 dimension이 single-dimension에서 multi-dimension으로 바뀌기 때문입니다. 
이는 곧 binary-classification 문제를 multi-class classification 문제로 바꾸어 버리는, 의도치 않은 상황을 초래하게 될 것입니다.

--- 

> 요컨대, 원활한 기계학습의 진행을 위하여 범주형 변수는 목적에 맞게 인코딩 되어야 하며, 독립변수가 범주형 데이터인 경우에는 LabelEncoding, OneHotEncoding 등을 생각해볼 수 있겠습니다. 반면, 종속변수가 범주형 데이터인 경우에는 많은 경우에 Label Encoding을 해주는 것으로 충분하다고 할 수 있겠습니다. 물론, 범주형 데이터를 처리하기 위한 인코딩 방식에는 이 외에도 많은 것들이 있습니다. 추후에 다른 방법들도 포스팅할 계획입니다.

#### Reference
[Handling Categorical Data in Machine Learning Models](https://www.pluralsight.com/guides/handling-categorical-data-in-machine-learning-models)  
[Using “one hot” encoded dependent variable in random forest](https://stackoverflow.com/questions/53589993/using-one-hot-encoded-dependent-variable-in-random-forest)

#### 추가로 학습할 것
[All about Categorical Variable Encoding](https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02)
