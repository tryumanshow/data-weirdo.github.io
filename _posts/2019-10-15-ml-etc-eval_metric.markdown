---

layout: post
title:  "Et cetera-Evaluation metric"
subtitle:   "et cetera-evaluation_metric"
categories: ml
tags: etc
comments: true

---

- 분류문제의 평가 지표(Classification Metric)에 대한 글입니다.

---  

어떤 상황이 있습니다. 공장장은 한 직원에게 "불량 여부를 예측하는 모델을 만들어달라"라는 요청을 했습니다. 
이에 그 직원은 모델을 만들었고 그 모델의 정확도는 96.2%에 달했습니다. 이에 공장장은 기뻐하며 그 모델을 테스팅 
과정 없이 그대로 사용하기로 결정했습니다. 하지만 몇 주 뒤 공장장은 돌아와서 얘기했습니다. "이 모델은 쓰레기야!" 
이에 직원은 물품들을 조사해보았고, 실제로 불량률은 3.8% 정도라는 결론을 내렸습니다. 무엇이 잘못된 것일까요?  

그렇습니다. 불량률이 3.8%인데 모델의 정확도가 96.2%라면, 모든 물품이 불량품이 아니라고 예측을 해버리면 정확도는 
96.2%가 나올 수 있는 것입니다. 이런 문제는 현재 갖고 있는 데이터의 불균형에 의해 발생한 것입니다. Imbalanced 
Dataset에 대한 얘기가 나오면, 항상 Evaluation metric으로 accuracy를 선택하는 것의 문제점에 대한 고찰로 시작합니다.  

---  

# Classifier의 성능을 어떻게 평가할 것인가?  
정확도(Accuracy)는 분류기의 성능을 평가하는 좋은 지표가 맞지만, 때때로는 성능을 평가하기에 좋은 지표가 아닐 수 있으므로 
다른 metric들도 함께 알아두는 것이 중요합니다. 

## Confusion matrix  
분류 모델의 성능 평가를 시작하기에 좋은 시작점입니다. Confusion Matrix는 모델이 분류를 어떻게 하고 있는지에 대한 
오버뷰를 제공해줍니다. 다음 2x2의 직사각형 그림이 바로 `Confusion Matrix`입니다.  
![](https://miro.medium.com/max/2104/1*Yslau43QN1pEU4jkGiq-pw.png)  
이 경우를 참/거짓의 문제로 바꾸어 보면 다음과 같이 표현할 수 있습니다. 행은 관찰값, 열은 예측값을 의미합니다.   
![](https://t1.daumcdn.net/cfile/tistory/995C7A3359E629C812)  

## Accruacy, Precision, Recall, F1-Score  

- 정확도(Accuracy)  
  : 정확히 예측한 수 / 전체 샘플 수  
  : $$\frac{TP + TN}{TP + FN + FP + TN}$$
- 정밀도(Precision)  
  : 진짜 양성 / 양성으로 예측된 샘플 수  
  : $$\frac{TP}{TP+FP}$$  
- 재현율(Recall)  
  : 통계학의 민감도(sensitivity)와 같은 개념  
  : 전체 양성 샘플 중 실제로 양성 클래스로 분류되는 수  
  : $$\frac{TP}{TP+FN}$$  
- F1-score  
  : 정밀도, 재현율의 조화평균  
  : $$\frac{2*Precision*Recall}{Precision + Recall}$$  

Recall과 Precision의 조합은 다음과 같은 해석을 제공합니다.   
`High recall + High precision`: 모델이 클래스 분류를 잘 컨트롤하고 있다.  
`High recall + Low precision` : 쿨래스가 잘 탐지되고는 있지만 모델이 다른 클래스의 점들 또한 포함시키고 있다.  
`Low recall + High precision` : 모델이 클래스를 잘 분류하지는 못하지만 분류를 한다면 이건 믿을만하다.  
`Low recall + Low precision` : 모델이 클래스 분류를 잘 컨트롤하지 못하고 있다.  

## ROC(Receiver Operating Characteristic) Curve  
ROC 곡선 또한 metric으로 사용될 수 있습니다. 통계학에서는 ROC 커브의 x축을 (1-Specificity), y축을 Sensitivity로 
놓습니다. 다르게 표현하면 x축이 1-TN 즉 FP, y축이 TP가 되겠네요. 결국 FP와 TP의 관계를 나타낸 곡선입니다. 
가장 좌측 하단이 (0,0), 가장 우측 상단이 (1,1)의 점이고, 각각은 그래프의 시작점과 끝점이 됩니다. 곡선이 가파를수록 
좋은 모델이 되며, 곡선 아래의 면적을 AUROC라고 합니다. AUROC는 0.5와 1 사이의 값입니다. ROC Curve는 다음과 같은 형태로 그려집니다.  
![](https://miro.medium.com/max/4885/1*thHBCWlaKWIkouryKBh6Wg.jpeg)  
요컨대 높은 AUROC score란 좋은 Recall을 얻기 위해 Precision을 많이 희생하지 않아도 되는 모델로부터 얻어질 수 있습니다.  




#### References  
[Handling imbalanced datasets in machine learning](https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28)  
