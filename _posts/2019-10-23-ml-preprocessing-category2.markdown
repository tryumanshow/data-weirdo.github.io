---
layout: post
title:  "Preprocessing-범주형 변수 다루기 2"
subtitle:   "preprocessing-categorical2"
categories: ml
tags: preprocessing
comments: true

---

- Mean encoding에 대한 글입니다. 

---  

이제껏 범주형 변수를 다룰 때에 기껏해야 label encoding, one-hot encoding 정도에 대해서만 알고 있었습니다. 하지만, 분명 
스스로가 느끼기에도 이 둘은 한계를 가지고 있었고, 다른 방법이 없을까를 고민하다가 mean encoding이라는 범주형 변수 처리방법이 있다는 것을 
알게 되어 공부해보았습니다. mean encoding 뿐만 아니라 scikit-learn은 범주형 변수를 처리할 수 있는 많은 방법들에 대한 API를 제공하고 있었습니다.  

Label Encoder와 One-hot encoder는 분명 해당 범주형 feature와 출력값 간의 관계는 고려하지 않는 방법이었습니다.  
이와 대비되게, Mean encoding은 범주형 변수와 출력값 간의 관계를 지어주려는 노력입니다.  
변수들을 숫자로 표현하면서도, 그 숫자들이 숫자로서의 의미를 가지게 되는 것입니다.  

Kaggle에서 돌아다니는 [타이타닉 데이터]9https://www.kaggle.com/c/titanic/data?)를 이용하였습니다.  
여기에서 Pclass는 카테고리가 3개인 범주형 변수입니다. 이 변수를 `mean encoding` 해보겠습니다.  

```  
data = pd.read_csv('data\train.csv', usecols=['Survived, 'Pclass'])  
data.head(5)  
```  
[그림1]()  
```  
data['Pclass_mean']=data['Pclass'].map(data.groupby('Pclass')['Survived'].mean())  
data.head(10)  
```  
[그림2]()  
index 0의 0.242363은 Pclass가 3인 애들의 평균을 의미하는 것인데 실제 다음과 같이 평균을 구해보면 그 값이 
동일함을 확인할 수 있습니다.   
```  
len(data.loc[(data.Survived==1) & (data.Pclass==3) ]) / sum(data['Pclass']==3)  
0.24236252545824846  
```  

```  
data.pivot_table(columns=df.Survived, index=df.index, values='Pclass_mean').iplot(kind='histogram', bins=100, xrange=(0,1))  
```
[그림3]()

그런데 다음과 같은 접근은 비록 범주형 변수와 출력값의 관계를 포함해준 것이기는 하지만, 다음과 같은 장·단이 존재합니다.  
```  
장점  
1. One-hot encoding처럼 범주의 개수에 따라 컬럼을 많이 만들어내지 않는다. (차원의 저주가 없다.)  
2. 그렇기 때문에 학습 속도가 빠르다.  
3. 아무래도 출력 레이블과의 관계를 고려했기 때문에 bias가 적다.  

단점  
* Overfitting  
1. Data Leakage 문제 - Mean encoding의 과정에서 이미 출력값과의 관계를 다 고려해버렸기 때문에  
Training set에만 오버피팅 되는 결과가 생긴다.  
2. 하나의 mean을 사용 → 테스트셋의 분포가 이와 같을 거라는 보장이 없고, 실제로 다르다면 오버피팅 발생  
3. Training set에서 특정 범주의 데이터가 적을 경우, 과연 이 값이 실제 테스트 셋의 경향성을 대표할 수 있을까?  
```  

요컨대, Label encoder와 one-hot encoder가 지니는 문제를 일부 해결했지만 여전히 오버피팅 문제가 남아있습니다.  
이 문제를 해결하려는 노력 또한 존재합니다. (`Smoothing`,  `CV loop`, `Expanding mean` 등)  

---  

- Smoothing  
단점 3번을 상쇄하기 위한 해결책입니다. 만약 트레이닝 셋의 소수가 테스트 셋의 전체 평균을 대표하기 힘들다면, 
소수의 평균을 전체 평균에 치우치도록 만들어주면 됩니다.  

Smoothing 방법은 다음과 같습니다.  
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FbU3ZZc%2Fbtqyt43239w%2F8B8jyGqemj62p9pKQoNKjk%2Fimg.png)  
alpha는 유저가 결정하는 하이퍼파라미터이고 이 값이 커질수록 전체 평균 쪽으로 더 치우치게 될 것입니다.  
  
$$p_c$$ : mean target value for category c  

```  
data['Pclass_n_rows'] = data['Pclass'].map(df.groupby('Pclass').size())  
global_mean = data[target].mean()  
alpha = 0.7  

def smoothing(n_rows, target_mean):  
    return (target_mean*n_rows + global_mean*alpha) / (n_rows + alpha)  

data['Pclass_mean_smoothing'] = data.apply(lambda x:smoothing(x['Pclass_n_rows'], x['Pclass_mean']), axis=1)  
data[['Pclass_mean', 'Pclass_mean_smoothing']]  
```  
![그림4]()  
결론적으로 이전에 비해 값들이 평균 쪽으로 더 기울게 됩니다.  
```  
data.pivot_table(columns=df.Survived, index=df.index, values='Pclass_mean').iplot(kind='histogram', bins=100, xrange=(0,1))  
```  
![그림5]()  

---  

- CV Loop  
CV Loop는 훈련 세트에 Cross Validation과 Mean Encoding을 섞어 Data Leakage를 줄이고, 이전 과는 다르게 
모든 인코딩된 값들이 다양한 값을 가지도록 만듦으로써 후에 트리모델을 적용할 때 더 좋은 훈련 효과를 볼 수 있도록 합니다.  

```  
from sklearn.model_selection import StratifiedKFold # Kfold 만들어 주기.  

data_new = data.copy()  
data_new[:] = np.nan  
data_new['Pclass_mean'] = np.nan  


X_train = data.drop('Survived', axis=1)  
Y_train = data['Survived']  
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 각 Fold iteration.  

for tr_idx, val_idx in skf.split(X_train, Y_train):  
    
    X_train, X_val = data.iloc[tr_idx], data.iloc[val_idx] # train set 에서 구한 mean encoded 값을 validation set 에 매핑해줌.   
    
    
    means = X_val['Pclass'].map(X_train.groupby('Pclass')['Survived'].mean())  
    X_val['Pclass_mean'] = means  
    data_new.iloc[val_idx] = X_val # 폴드에 속하지못한 데이터들은 글로벌 평균으로 채워주기.  
    
global_mean = data['Survived'].mean()  
data_new['Pclass'] = data_new['Pclass'].fillna(global_mean); data_new  
```  
실제로 data_new의 Pclass_mean 값들이 다양해졌음을 알 수 있다.  
![그림6]()  

그래프를 그려보면 다음과 같다.  
```  
data_new.pivot_table(columns=data_new.Survived, index=data_new.index, values='Pclass_mean').iplot(kind='histogram', bins=100, xrange=(0,1))  
```  
![그림7]()







#### References  
[Why you should try Mean Encoding](https://towardsdatascience.com/why-you-should-try-mean-encoding-17057262cd0)  
[Categorical Value Encoding 과 Mean Encoding](https://dailyheumsi.tistory.com/120#1.-one-hot-encoding)
