---
layout: post
title:  "Pycharm에 IPython 환경 적용하기"  
subtitle:   "ipython"
categories: computing
tags: python
comments: true
---

- PyCharm에 IPython 환경을 적용하는 방법에 대한 글입니다.  

---  

처음 Python 사용을 시작했을 때는 대화형 인터프리터 환경의 Jupyter Notebook을 사용하다가, 어느 순간부턴가 PyCharm IDE을 이용하는 경우가 많아진 것 같습니다. 그런데, `%matplotlib inline`나 `from Ipython.display import HTML` 등과 같이 IPython에서 활용할 수 있는 모듈 및 명령어들을 PyCharm에서는 사용할 수 없는 경우들이 생기기 시작했고, 뿐만 아니라, 특정 부분만 취사선택해서 실행하며 결과를 그때그때 확인해볼 수 없다는 점이 초보자의 입장에서는 적잖게 불편했던 것 같습니다. 대화형 인터프리터를 사용할 수 있을 방법이 없을까 고민을 하고 있던 와중에, [유튜브의 한 동영상](https://www.youtube.com/watch?v=6JpLmAWa6lA)에서 그 해결책을 제시해 주었습니다.

다음과 같은 단계를 따르면 됩니다.

- Step1: File -> Setting을 클릭한다.  
  ![ipython1](https://user-images.githubusercontent.com/43376853/97465719-ad134c80-1985-11eb-9b86-4072a642f0e3.png)  
  
- Step2: Build, Execution, Deployment Toggle 내 Console Toggle에서 `Use IPython if available`이 체크되어있는지 확인한다.  
  ![ipython2](https://user-images.githubusercontent.com/43376853/97465721-adabe300-1985-11eb-8cd2-1d0db2d0bc03.png)  
  
- Step3: Project Interpreter 란을 클릭한 후 우측의 + 칸을 클릭한다.  
  ![ipython3](https://user-images.githubusercontent.com/43376853/97465725-adabe300-1985-11eb-8872-6dab91f68c68.png)  
  
- Step4: `ipython`을 입력하여 검색한 뒤, `Install Package` 클릭  
  ![ipython4](https://user-images.githubusercontent.com/43376853/97465709-ab498900-1985-11eb-9963-0b7e3f5fb18a.png)  
  
- Step5: Success할 때까지 기다리기.  
  ![ipython5](https://user-images.githubusercontent.com/43376853/97465713-ac7ab600-1985-11eb-847b-d25f99e749f6.png)  
  
- Step6: PyCharm을 껐다가 다시 실행하면 끝!  



- 실제로, 설치이전에는, `>>>` 모양의 파이썬 쉘(Shell)이 콘솔에 존재했던 반면, 설치 후에는 Jupyter Notebook에서 볼 수 있는 [1]
과 같은 숫자 표시로 바뀌게 됩니다.  
  ![ipython8](https://user-images.githubusercontent.com/43376853/97465716-ad134c80-1985-11eb-8417-cfe918f4c5fd.png)  
  ![ipython7](https://user-images.githubusercontent.com/43376853/97465714-ac7ab600-1985-11eb-8203-4f7f53304a0f.png)  
  
- 실행하고 싶은 부분을 지정해준 뒤 `Alt+Shift+E`를 누르면, 선택된 행만을 취사선택하여 실행할 수 있게 됩니다.  


---  

#### References  
[How to add ipython console to pycharm?](https://www.youtube.com/watch?v=6JpLmAWa6lA)  
