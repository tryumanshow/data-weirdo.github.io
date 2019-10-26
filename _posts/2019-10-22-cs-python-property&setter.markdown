---  

layout: post
title:  "파이썬 Property와 Setter 메소드"
subtitle:   "python-property&setter method"
categories: cs
tags: python
comments: true

---

- Python에서의 @property와 @setter 데코레이터에 대한 글입니다.   

---  

자료구조에서 `단순 연결 리스트`를 공부하다가 @property와 @setter에 대한 언급이 있었는데, 이 부분에 대해서는 생소해서 조사 해보았습니다. 
자바의 클래스 생성자 중에 getter, setter라는 것이 있다는데, 파이썬의 경우는 getter, setter 대신 @property, @setter를 사용한다고 합니다.  

파이썬에서 클래스를 생성하고 난 뒤 때때로 값을 저장하거나 가져오기도 합니다. 이 때 값 가져오는 메소드를 getter, 값을 저장하는 메소드를 
setter라고 한다고 합니다. 

학생의 출신 지역을 정의하는 클래스가 있다고 해보겠습니다.  
```
class Person:  
    def __init__(self):  
        self.__area = '알 수 없음'  
 
    def get_area(self):           # getter  
        return self.__area  
    
    def set_area(self, value):    # setter  
        self.__area = value  
 
ryu = Person()  
ryu.set_area('대구')  
print(ryu.get_age())  

```  

이 코드를 `@property`와 `@setter`를 사용하여 다음과 같이 만들 수 있습니다.  

```
class Person:  
    def __init__(self):  
        self.__area = '알 수 없음'  
 
    @property
    def area(self):           # getter  
        return self.__area  
        
    @setter
    def area(self, value):    # setter  
        self.__area = value  
 
ryu = Person()  
ryu.area = '대구'  
print(ryu.area)  

```  

주목할 점은 getter, setter 메서드의 이름이 area로 통일되었다는 것입니다. getter에 @property를, setter에 @setter 데코레이터를 붙였고, 
값 할당과 가져오기가 훨씬 수월해졌습니다.  

#### Reference  
[파이썬 코딩 도장 - 프로퍼티 사용하기](https://dojang.io/mod/page/view.php?id=2476)
