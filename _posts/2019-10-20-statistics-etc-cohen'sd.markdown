---

layout: post
title:  "Cohen's D란?"
subtitle:   "statistics-cohen's_d"
categories: statistics
tags: etc
comments: true

---  

- Cohen's D에 대해 정리한 글입니다. Cohen's D는 무엇인지, 해당 지표의 형태와, 해당 지표에 대해 어떻게 해석해야 되는지에 대해 조사해보았습니다.  

---  

통계학을 공부하다가 Cohen's D에 대해 한 번쯤은 들어봤을 법도 한데, 기억을 못하는 것인지 모르겠지만 "통계학을 공부한 사람이지 
않냐?"는 말에 깜짝 놀랐습니다. 부리나케 찾아보았습니다. 찾아보니 두 모평균 차이 검정에서 본 적이 있는 형식이었습니다.  

## Cohen's D가 무엇인가?  
Cohen's D는 효과의 크기(`effect size`)를 측정하는 가장 보편적인 방법 중의 하나라고 합니다. 
두 약물 A, B가 있을 때 약물 A의 효과가 약물 B의 효과보다 크다고 하는 것처럼요.  

## Cohen's D의 형태(formula)  
$$ d = \frac{(M_1 - M_2)}{S_{pooled}} $$,  

$$M_1$$: 그룹 1의 평균  
$$M_2$$: 그룹 2의 평균
![](https://latex.codecogs.com/gif.latex?%24%24S_%7Bpooled%7D%24%24%3A%20%24%24%5Csqrt%7B%5Cfrac%7Bs_1%5E2&plus;s_2%5E2%7D%7B2%7D%7D%24%24)  

Cohen's D는 큰 크기의 샘플에서 잘 작동하며, 샘플의 크기가 작을 때는 효과를 과대포장하는 경향이 있기 때문에, 샘플의 개수가 적을 경우 
correction factor를 사용하여 효과의 과대포장을 잡아준다고 합니다. 형태는 다음과 같습니다.  
![](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2016/10/small-samples-formula.png)  

## Cohen's D 해석  
Cohen's D의 크기 즉 d는 두 집단 간에 d의 표준편차만큼 차이가 난다는 의미라고 합니다.  

|d의 크기|Effect|
|------|------|
|.2|small|
|.5|medium|
|.8|large|  

## 유의할 점  

small effect는 육안으로 확인하기 힘든 정도, medium effect는 육안으로 확인될 수 있을 정도로 충분히 큰, 
large effect는 명백하게 육안으로 확인 가능한 효과의 정도라고 합니다. 15세 학생과 16세 학생의 신장 차이를 small effect로, 
13세 학생과 18세 학생의 신장 차이를 large effect의 예로 들 수 있습니다.  

하지만 large effect가 small effect보다 반드시 낫다는 것은 아닙니다. 특히 작은 차이가 큰 영향을 만들어 낼 때 말입니다. 
예를 들어 시험 성적이나 건강 지수가 조금만 올라도 현실 세계에서는 그 조그마한 차이가 굉장히 유의미할 수 있다는 것입니다. 
한편, effect가 0.2보다 작을 때에도 지수는 낮아서 효과는 낮지만 통계적으로 유의할 수는 있습니다. Cohen's d의 판단이 
곧 통계적 유의성을 판단하는 지표는 아니라는 점에 주의해야 할 것 같습니다.  


cf. effect size를 판단하는 기준으로 [Hedge's G](https://www.statisticshowto.datasciencecentral.com/hedges-g/)라는 개념도 있었습니다. 
요약하자면, Hedge's G는 샘플 사이즈가 20보다 작을 때는 Cohen's D보다 더 정확하고 샘플 사이즈가 20보다 크면 Cohen's d와 거의 비슷하다고 합니다. 
그래서 Hedge's G를 `corrected Cohen's D`라고 부르기도 한답니다. 또, Hedge's G는 실험군과 대조군 사이의 차이를 볼 때 보통 쓰이며, 만약 
두 집단 간의 표준편차가 유의하게 다르다면, 이 때는 Hedge's G 대신 `Glass's delta`라는 개념을 쓴다고 합니다. Glass's delta는 대조군의 
표준편차만을 사용한다고 합니다.  


#### Reference  
[Cohen’s D: Definition, Examples, Formulas](https://www.statisticshowto.datasciencecentral.com/cohens-d/)  
[Hedges’ g: Definition, Formula](https://www.statisticshowto.datasciencecentral.com/hedges-g/)
