---  

layout: post
title:  "Samsung SDS Techtonic 2019"
subtitle:   "daily_life-Techtonic 2019"
categories: small_talk
tags: daily_life
comments: true

---  

삼성SDS Techtonic 2019행사에 참여하였습니다.  관심사에 대한 정리입니다. 

---  

# Track1-A. 자동 레이블링과 분산 학습을 통해 딥러닝을 쉽고 빠르게!  
→ `개인적으로 Brightics Tool을 홍보하는 세션처럼 느껴졌습니다.`  

- 딥러닝은 반드시 플랫폼이 필요하다. 왜?  
	
  - a. 데이터가 많다고 하더라도, 이 데이터들 모두를 학습에 적용하기는 힘듦.
    - 웹 크롤링 후 → 쓸 만한 데이터(사진)가 별로 없더라. (레이블링 손수 했어야 됐다.. 힘들다)
		
		→ `Auto Labeling`이라는 모듈 탄생
	
	- b. 모델 개발 
		- 모델 시작하면 하루 이틀 ~ 일주일 정도까지 걸리는 케이스들도 있다. 
		  - 하지만 학습은 한 번으로 끝나지는 않는다.  
		- 일일이 다 해봐야 하는가? 의 문제  
    
- Brightics DL  
  - `Auto Labeling` → 학습 데이터 라벨링 자동화  
  - `Distributed Model Training`  → 대용량 데이터 수집/전처리/학습과정 분산처리  
  
  - 데이터수집 → 데이터전처리 → 데이터라벨링 → 모델선정 → 모델평가 → 모델추론`   
  
## Auto-Labeling  
- 기존: 레이블링은 수작업  
- 원리  
```  
  - 1. 약 5% 정도의 unlabeled 데이터를 사람이 손수 레이블링 해준 후 학습  
  - 2. 나머지 95%의 unlabeld 데이터: 모델이 추론  
    - 가령 추론 이후 90% 데이터는 확실, 5%는 불확실  
    → 3. 이 5% 데이터에 사람이 라벨링을 해준 후 학습  
  - 일정 수준의 목표 정확도 달성시까지 2, 3 반복  
```  
  → STL10 데이터, 20%의 레이블 정보로, Auto Label 정확도 80% 달성  
  
## Distributed Model Training  
```  
estimator = Estimator.NewClassificationEstimator(mode1_fn=my_model_function_with_mnist)
name = 'BestModelEstimator' + str(time.time()).replcae('.','')  
estimator = Estimator.create(name, 'Hello', estimator)  

hyper_parameters = HParams(iterations=50000, batch_size=10)
rc = RunConfig(no_of_ps=5, no_of_workers = 20, summary_save_frequency=5000, run_eval=True, 
                use_gpu=False, checkpoint_frequency_in_steps = 500)
                
exper = Experiment.run(experiment_name='BestModelEstimator' + str(time.time()).replace('.',''),
                       description='Really first model', 
                       estimator=estimator,
                       hyper_parameters=hyper_parameters,
                       run_config=rc,
                       dataset_version_split=None,
                       input_fuction=input_function)  
job = exper.get_single_job()
print(job.__dict__)
print('')
print('tensorboard url')
print(job.get_tensorboard_url())

job.has_finishied()

job.wait_until_finish()
```  
`엔지니어링 및 분산 처리 쪽에도 관심이 있지만, 아직까지는 이해가 많이 짧습니다.`  

---  

# Track2-A. 회로 설계 자동화를 위한 강화학습 적용기 - 제조현장 강화학습 적용 가이드  

## 시작하기에 앞서  
- '반도체 회로 (PCB) 설계 프로세스 자동화 기술' 이라는 프로젝트  
  → 난관 봉착  
```  
  1. 데이터가 적다 
    - 일반적인 딥러닝 기법 적용 불가  
    - 현재 소수의 전문가들이 직접 '손'으로 설계  
  2. 회로 간의 교차가 없어야 한다는 제약 조건  
    - 회로 설계에서는 회로 간의 간섭이 없어야  
    - 교차가 생기면 설계도로서의 기능을 못함  
  3. 시작점과 끝점이 고정되어 있다는 제약 조건
    - 시작점과 끝점은 이어져야  
    - 시작점과 끝점은 쌍으로 정해져 있음.  
```  
  
  → 그래서, `강화학습`으로 눈을 돌림  
```  
  1. 매우 일반적인 방법론으로 모든 문제에 적용 가능(하다고 생각을 했다.)  
  2. 하지만 기타 AI에 비해 성능이 떨어짐  
  
  - 각 상태마다 최적의 행동을 도출  
```  
## 산업과 강화학습

- 강화학습: 특정 상태에서 최적의 행동을 학습하는 일반적 방법론  
  - 하지만 산업체에서는 적용이 잘 안되고 있다. 왜?  
  
  - 산업에서 강화학습이 적용되기 위한 4요소  
```  
  1. 문제를 나타내는 판 (Environment)  
  - State와 Action을 이용하여 문제 표현  
  - 문제를 명확히 정의  
  2. 데이터 생성 방향성 (Reward)  
  - 탐색과 선택을 도울 수 있도록 보상 설정  
  - 학습 방향 가이드 역할  
  3. 현재 상태를 해석 (Feature Extraction)
  - State를 feature로 표현  
  - 신경망 이용  
  4. 학습의 주출이 되는 RL 알고리즘 (RL Algorithm)  
  - Policy Gradient 
  - Value Function  
```  
  `강화학습은 자동차의 엔진일 뿐, 이 네 가지중 한 가지로 갖추어져 있지 않다면 강화학습 적용 어렵다.`  
  - 하지만 이 4개 충족되기 굉장히 어려움  
```  
  1. Environment 설정 자체가 힘듦  
  2. Reward 설정에는 현업 전문가 필요  
  3. 적절한 Reward를 통해 Agent가 움직이더라도 현 상태를 분석하지 못하면 학습 불가  
  4. 현재 논문들: RL 알고리즘은 Environment나 Reward가 확정된 후 평가될 수 있다고 서술  
```  
## 강화학습 적용기  
  - 해결하기 위한 노력들 (위상수학 개념까지 도입)  

## 강화학습이 성공하려면?  
- 위 네 가지 요소 모두 필요  
- 이 네가지 요소 설정을 위해 개발자와 현업 전문가 간의 협업 필요  
- 개발자는 딥러닝에 대한 폭넓은 이해력과 구현력 필요  
- 강화학습은 일반적인 딥러닝보다 더 많은 학습 데이터 요구  
  (강화학습은 데이터가 없어도 된다? → 이건 착각)  
- 강화학습은 일반적인 supervised learning보다 성능이 떨어짐  
  

---  

# Track3-A. 딥러닝 실전 활용! 이미지 데이터에서 유용한 정보를 자동 추출해보자!  

- 다양한 성공 사례  
```  
  1. 자동차 결함 검출  
  2. 설계 도면 인식  
  3. 암 발병 위험 예측  
  4. 안질환, 부정맥 예측  
```  
  
- 하지만 딥러닝의 Feature Extraction을 100% 자동화한다는 것은 불가능  
- No single prescribed recipe  

- 딥러닝 개발 과정에서 겪는 어려움  
  
```  
  * 개발의 관점  
  1. 데이터  
    - 데이터는 넘쳐나는데, 어떻게 정리?  
    - 데이터가 부족한데, 어떻게 확보?  
  2. 모델  
    - 다양한 모델 중 어떤 모델을 써야하지? 선택 어려움  
    - 선택 모델이 문제/데이터에 적합?  
  3. 학습  
    - 지정한 파라미터가 최선? 
    - 더 좋은 파라미터 어떻게 찾지?     
    
  * 실행의 관점  
  1. 개발환경  
    - 실험 준비 기간이 너무 길다.  
    - 개발환경/라이브러리 설치의 어려움  
  2. 자원관리  
    - 한정된 자원으로는 많은 실험이 어려움  
    - 효율적 자원 사용 방법?  
  3. 분산수행  
    - 과정이 복잡, 수작업 많음  
    - 설정만으로 쉽게 분산 수행이 될까?  
```  
   
## 정확도 향상을 위한 작업들  

- cf) OCR (Image to Text)  
- `AICR` (AI-based OCR)  
  - OCR + 자동 (이미지 전처리 + 양식/표 추출 + 신뢰도 + ...)  

  - 다양한 해결 기법 설명 
    - 학습 데이터 Augmentation  
    - 워터마크(Watermark) 개선  
      - 명도 기준으로 인식에 주는 영향을 약화  
    - 회전 보정  
      - 90도 이하의 작은 회전은 영상처리 기법 할용 (컴퓨터 비전)  
    - 직인 제거  
      - 색상 분할(Color Clustering)을 통한 분할  
    - 스탬프 추출  
      - 색상 분할(Color Clustering)을 통한 분할  
    - 필기 영역 추출  
      - 색상 분할(Color Clustering)을 통한 분할  
    - 변형된 심볼 검출  
      - P&ID 도면 인식 시 Pain Point  
    - 접힌 혹은 구겨진 이미지 처리  
      - (SUNY-Stony Brook Univ.와의 공동연구)  
   
## 결과의 효과적 활용을 위한 작업들에 대한 설명  
- 인식 결과에 대한 신뢰도 제공 (Adjustable Confidence Thresholds)  
- 모델 성능 검증 자동화  

## 결론  
`공짜 점심은 없다.`  
![](http://drive.google.com/uc?export=view&id=16F_mb-UurhG1p1xBJT8oNcxgx_V2jC_G)  
---  

# Track4-A. DefogGAN: GAN을 활용한 스타크래프트 게임의 상대 정보 유추  

- Game AI  
  - MDP (Markov Decision Process)  
  - Agent가 지속적으로 게임을 하면서 스스로 학습  

- StarCraft AI: fog-of-war의 문제를 가짐  
  - Partially Observable Markov Decision Process (PDMDP)  
  - 상대방이 무엇을 하고 있는지 알 수 없다는 것 (fog-of-war)  

|Inpainting|Defog|  
|----------|-----|  
|가려진 부분|전체 부분|  
|주변의 그림을 기반|Partial observation을 기반|  
|진짜 같은 그림을 그린다|공간상 유닛이 존재할 위치를 찾고 개수를 추론|  

- Defog를 통한 에이전트 성능 개선  
  - Defogger는 전투 가치 판단을 도와줌  

- GAN  
  - `잠재공간` → `데이터 공간`  
  - Interpolation in Latent Space  
  
- Defog GAN  
  - 손실함수: (실제상황과 픽셀 별로 비교하여 그 차이만큼)  
  - 데이터 구성 방법  
    - 이미지를 사용하지 않고, 유닛 맵을 만들어 사용  
  - 관찰기억 사용  
  - `부분적 관찰 공간(Fog state)` → 재현되는 공간(Real State)`  
  
- 결과 비교  
```  
  1. Convolutional Encoder-Decoder (SOTA)  
  2. DCGAN  
  3. BEGAN  
  4. WGAN-GP  
  5. cWGAN
  6. DefogGAN  
```  
- AAAI 2020 학회 발표 예정  
    
---  

# Track6-B. Optimizing 엔진을 이용한 물류운영 개선 사례  

## Route Optimizer  
- 트럭 배송계획 수립 솔루션  
- 루트 생성 방법  
```  
  1. 수작업  
  2. 컴퓨터 알고리즘 이용 → 논문의 알고리즘을 현장에 즉시 적용하기는 어려움   
```  
  - ex) 수납장 하나 설치 20분, 그럼 3개 설치는? 60분!  
    No. "기사는 몸이 재산.. 현장을 몰라도 너무 모르네" → 한 20분 쉬고 80분!  
    
  - 현장에는 제약사항이 너무 많다.  

- 제약의 예  
  - Max Stop (배송 가능한 Stop 수 제약)  
    - 이 제약이 없으면, 트럭의 가용공간을 모두 사용할 때까지 Route 생성  
    - 제약을 사용하면, 트럭의 가용공간에 여유가 있더라도 다른 트럭 사용  
  - 창고복귀  
    - 창고에 꼭 복귀 후 퇴근할 것인가, 바로 집으로 퇴근할 것인가.  
  - 2인 설치  
    - 두 명이 만나서 같이 출발할 것인가, 설치 위치에서 두 기사가 만날 것인가.  
    
  - 고려해야할 제약 사항이 엄청나게 많음  
  ![](http://drive.google.com/uc?export=view&id=1jldNFt2cHH5Cvxum5jrbrK-IKiz6wVAt)  
  - 이런 수많은 제약들을 만족할 수 있어야 `Route Optimizer`를 쓸 수 있음  
  
- 이런 것들을 잘 만족하면서 Route Optimizer를 만드려면?  
```  
1. 초기해를 잘 만든다. 
2. Route 개선 위해 Move, Swap 연산 수행 
3. Tabu Search  
```   

- Warehouse Network Optimizer (클러스터링 알고리즘)  
  - K-Medoids 
  
---  

크게는 인공지능 및 데이터, 개발 및 컴퓨팅(분산처리) 세션으로 나누어져 있었습니다.  
전자가 주 관심사이며, 전자의 영역을 공부한 뒤 후자의 영역을 공부할 것입니다.  

### 자료  
[삼성SDS Techtonic 2019](https://www.samsungsds.com/global/ko/about/event/techtonic2019.html)  
