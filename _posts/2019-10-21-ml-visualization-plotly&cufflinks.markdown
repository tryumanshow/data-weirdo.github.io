---

layout: post
title:  "Plotly&Cufflinks에 대한 간단한 소개"
subtitle:   "visualization-plotly&cufflinks"
categories: ml
tags: visualization
comments: true

---  

- Plotly와 Cufflinks에 대한 간단한 소개글 입니다.

---  

# Plotly

![](https://miro.medium.com/max/272/1*X4YrdhT1SPLkTPDiX_jUGA.png)  

Plotly는 `Interactive Data Visualization`을 실현할 수 있는 오픈 소스입니다. 여기서 Interactive라 함은
대화식 데이터 시각화를 통해 플롯에서 직접 작업하여 요소를 변경하고 여러 플롯들을 연결할 수 있음을 의미합니다. 
Plotly를 통해 실현할 수 있는 그림들에 대한 한 클립을 보고 놀라지 않을 수 없었습니다.  
![](https://miro.medium.com/max/600/1*A8muRMkAljwW8PKWa_OFpg.gif)  
이제부터 Plotly에 대해 설명해보고자 합니다.  

### 설치  
```
import plotly.offline as py  
py.init_notebook_mode(connected=False)  
```  
Plotly 라이브러리를 만든 회사의 이름은 Plotly이며 이 회사는 많은 프로덕트들에 대해 향상된 기능들을 제공함으로써 돈을 법니다. 
`connected = False`로 설정해줌으로써 Plotly 사용을 위해 계정을 생성할 필요없이 오프라인에서 사용할 수 있게 됩니다.  

### 간단한 api 설명  
- 선그래프 생성: `.iplot()` 사용, 
  범례의 요소들을 클릭함으로써 선들을 숨기고 보이게 조절하는 것이 가능,  
  특정 영역의 피쳐들을 확대해볼 수도 있음.  
  
  ![](https://miro.medium.com/max/1590/1*zCxTcb7Bzgrb0Y0gY_DcrA.gif)  
  
- 스캐터플롯 생성: '.iplot(kind='scatter', x=X변수, y=Y변수, mode='markers')`  

  ![](https://miro.medium.com/max/1794/1*c2FpqWRUwPIs77jaQjPREg.png)  
  
- 막대그래프 생성: `.iplot(kind='bar')`  

  ![](https://miro.medium.com/max/1780/1*w4qTiHEF_Drn8LaviaGY_Q.png)

이 외에도 Plotly를 이용하여 박스 플롯, 히스토그램 등 상상하는 많은 그래프들을 그릴 수 있습니다.  

# Cufflinks

한편, cufflinks는 Interactive data visualization이 가능하도록 Plotly와 판다스 데이터프레임을 연결해주는 역할을 하는 모듈입니다.  
역시 모듈을 임포트하고 오프라인에서 사용하겠다는 설정을 해줍니다.  

### 설치  
```
import cufflinks as cf  
cf.set_config_file(offline=True)  
```  
[Cufflink](https://plot.ly/python/v3/ipython-notebooks/cufflinks/)를 사용하면 다음과 같이 판다스와 plotly가 연동이 가능합니다.  

### 간단한 api 설명  
```
df = pd.DataFrame(np.random.rand(6, 3), columns=['A','B','C']
df.iplot(kind='bar', barmode='stack')  
```  
이제 Cufflinks를 사용함으로서 판다스 데이터프레임 객체에서 바로  `.iplot()`을 사용하는 것이 가능해졌습니다. 

Plotly가 듣던 것처럼 강력한 시각화 툴인 것을 체감하면서, Plotly와 관련된 API를 공부해나갈 예정입니다.  

#### Reference  
[Day (7) — Data Visualization — How to use Plotly and Cufflinks for Interactive Data Visualizations](https://medium.com/@kbrook10/day-7-data-visualization-how-to-use-plotly-and-cufflinks-for-interactive-data-visualizations-3a4b85fdd999)  
[It’s 2019 — Make Your Data Visualizations Interactive with Plotly](https://towardsdatascience.com/its-2019-make-your-data-visualizations-interactive-with-plotly-b361e7d45dc6)

