---

layout: post
title:  "Et cetra-프로젝트 조직화하기"
subtitle:   "et cetra-프로젝트 조직화하기"
categories: etc
tags: detection
comments: true

---

- 프로젝트 조직화(Project Organizing)와 `cookiecutter`에 대한 정리입니다.  

---  

프로젝트를 수행함에 있어서 명심해두어야 할 것들 입니다.  
```  
1. 버전 컨트롤
2. Reproducible result를 위한 가상환경 설치  
3. Raw data, Intermediate data, Final data 분리  
4. 작업 문서화  
5. 모듈과 커스텀 함수 분리    
```

### 1. 버전 컨트롤  

- 깃을 사용하여 commit, push 할 때마다 참조할 수 있는 새 버전이 만들어지는 것  
- 프로젝트를 장기간 진행하면서 commit이 쌓이면 나중에는 '대체 이게 뭐야;;'하는 순간이 생기게 될 것  
- 항상 커밋 메시지를 명확하게 써줄 것  
  ```  
  A properly formed Git commit subject line should always be able to complete the following sentence:  
  "If applied, this commit will your subject line here"  
  ```  
  → 이같은 명확한 commit은 특히 모델을 여러 개 만들 때에 굉장히 도움이 됩니다.  
  예. `git commit -m 'Added additional layer with 50 neurons.'`
  
- 시간 절약에 굉장히 도움이 될 것입니다.  

---  

### 2. 가상환경  

- 가상환경을 사용하는 주된 이유는 당연 `사용한 패키지의 버전 컨트롤`일 것입니다.    
    ```  
    # 가상환경 설정하기

    # Python2  
    $ virtualenv env  

    # PYthon3.6+  
    $ python3 -m venv <<name of directory>>  

    # 가상환경 activate하기  
    $ source name_of_directory/bin/activate  

    # 가상환경 deactivate하기  
    $ deactivate 
    ```  
- 그 후 환경 내에 Jupyter를 설치합니다.  
    ```  
    (env_name) $ pip3 install jupyter  
    ```  
- +$$\alpha$$ : [cookiecutter](https://github.com/drivendata/cookiecutter-data-science) 사용하기  
  - cookiecutter를 사용하면 구조를 쉽게 볼 수 있을 뿐만 아니라, 여러 옵션들을 쉽게 사용할 수 있습니다.  
    ```  
    # Install  
    $ pip3 install cookiecutter  
    
    # cd to project directory  
    ```  
  - 누군가 똑같은 버전의 패키지들을 사용하고 싶어 한다면?  
    ```
    $ pip freeze > requirements.txt  
    $ pip install -r requirements.txt  
    ```
---  

### 3. 데이터 분리  
- 네 가지 표준 폴더  
  ```  
  1. External  
  2. Interim  
  3. Processed  
  4. Raw  
  ```
  - 이렇게 파일을 세그먼트로 나눔으로써 진행중인 작업을 저장하고 원본 데이터가 자신의 모습 그대로 보관할 수 있습니다.  
  - 저장되는 파일 자체가 다르기 때문에 `processing1`, `processing2`와 같이 헷갈리게 파일을 계속 만들어야 할 필요도 없습니다.  

---  

### 4. 작업 문서화  
- 문서화에도 역시 cookiecutter를 이용할 수 있습니다.  
  ex)  'Your `matplotlib` functions can now live in `./src/viz` and all the results from 
  those funcitons can be saved to `./reports/figures'

---  
### 5. 커스텀함수 사용  
- 역시 `cookiecutter`를 사용하면 `import os`를 여기저기 사용할 필요 없이 
   커스텀 함수를 쉽게 저장하고 쉽게 불러올 수 있습니다.  


#### Reference  
[Cookiecutter Data Science — Organize your Projects — Atom and Jupyter](https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e) 
차후에 Atom을 사용하게 된다면 Jupyter와 Atom을 함께 사용하는 방법 또한 공부해볼 것입니다. 
