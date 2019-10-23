---

layout: post
title:  "Python-클래스"
subtitle:   "python-class"
categories: cs
tags: python
comments: true

---

- 파이썬의 클래스에 대해서 정리해보았습니다.  

---  

파이썬의 클래스가 무엇인지에 대해서는 파이썬을 공부하기 시작한 극초기부터 공부를 해보았으나 (사실 파이썬을 독학하기 시작한지도 얼마 되지 않았습니다.)
자료구조의 단순 연결 리스트를 공부하다가 데코레이터에 대해 궁금해졌고, 데코레이터에 대해 공부하다보니, 클래스, 퍼스트클래스 함수, 클로저 등에 대해 
궁금해졌습니다. 클래스에 대해서도 조금 더 진정성 있게 공부를 해야겠다는 생각이 들어 클래스에 대해 다시 공부하게 되었습니다. 
 
먼저, 파이썬에는 `네임스페이스`라는 게 있습니다. 
네임스페이스라는 것은 `변수가 객체를 바인딩할 때 그 둘 사이의 관계를 저장하고 있는 공간`을 의미합니다. 
파이썬 상에서 a=5라고 하면 a라는 변수가 5라는 객체를 가리키는 주소 또한 저장하고 있는데 이 연결 관계를 저장하고 있는 공간이 바로 네임스페이스라는 겁니다. 
파이썬의 클래스 또한 새로운 객체(인스턴스)를 정의하는 데에 사용되며 마찬가지로 네임스페이스를 가집니다. 

클래스 상의 네임스페이스 관계도를 그림으로 나타내보면 다음과 같습니다.  
![](http://schoolofweb.net/media/uploads/2016/09/22/object_name_resolution_400x400.png)  
이 말인 즉슨 파이썬의 클래스들 간에도 서로를 참조하는 데에 엄연한 룰과 순서가 있다는 것입니다. 
이 그림은 '자식이 부모의 네임스페이스는 찾아갈 수 있는데, 부모가 자식의 네임스페이스를 찾아가는 것은 불가능하다.`라는 것을 암시합니다. 
예를 들어서 설명해보겠습니다. 

([스쿨오브웹](http://schoolofweb.net/blog/posts/%ED%8C%8C%EC%9D%B4%EC%8D%AC-oop-part-3-%ED%81%B4%EB%9E%98%EC%8A%A4-%EB%B3%80%EC%88%98class-variable/) 
이상희님의 설명자료와 코드를 참고하였습니다. 문제가 될 시 삭제하겠습니다) 아래 코드는 연봉 협상에 대한 클래스입니다.  

```  
# Salary.py
# -*- coding: utf-8 -*-

class Employee(object):
    
    raise_amount = 1.1
    
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first.lower() + '.' + last.lower() + '@github.com'
        
    def full_name(self):
        return '{} {}'.format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * raise_amount)  ###

emp_1 = Employee('Seungwoo', 'Ryu', 100000000)  

print(emp_1.pay)  # 기존 연봉
emp_1.apply_raise()  # 인상률 적용
print(emp_1.pay)  # 오른 연봉
```  
이 코드의 실행 결과는 다음과 같습니다.  
```  
NameError: name 'raise_amount' is not defined  
```  
Employee라는 클래스로부터 emp_1이라는 새로운 인스턴스를 만들면, emp_1이라는 별도의 네임스페이스가 생깁니다. 
기존의 Salary.py 코드에 덧대어 `__dict__`를 통해 emp_1에 대한 네임스페이스를 확인해보면 다음과 같습니다.  
```  
<입력>
print(emp_1.__dict__)  
```  
```  
<출력>
{'email': 'seungwoo.ryu@github.com',  
 'first': 'Seungwoo',  
 'last': 'Ryu',  
 'pay': 100000000}  
   }  
```   
여기서도 알 수 있듯, 분명 emp_1라는 객체에서 first, last, pay에 대한 언급만 해주었습니다. 그래서 raise_amount에 대한 정보는 
emp_1 이라는 새로운 인스턴스에서는 정의되지 않은 상태입니다. 결과값에서도 raise_amount가 정의되지 않았다고 합니다. 
그렇다면 위의 코드의 ### 부분을 수정해보겠습니다.  

```  
# Salary.py
# -*- coding: utf-8 -*-

class Employee(object):
    
    raise_amount = 1.1
    
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first.lower() + '.' + last.lower() + '@github.com'
        
    def full_name(self):
        return '{} {}'.format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)  ###

emp_1 = Employee('Seungwoo', 'Ryu', 100000000)  
print(emp_1.pay)  # 기존 연봉
emp_1.apply_raise()  # 인상률 적용
print(emp_1.pay)  # 오른 연봉
```  
그 결과를 출력해보면 다음과 같습니다.  
```  
100000000  
110000000  
```  
오류없이 잘 출력이 되었습니다. 이는 인스턴스에서 raise_amount가 정의되지 않았었기 때문에, emp_1 인스턴스의 클래스로 가서 
자동으로 raise_amount를 찾았습니다. Employee 클래스에서는 분명 raise_amount=1.1이었고 이 값을 가져와 없던 값을 대체한 것입니다. 
요컨대 위 그림의 네임스페이스 접근 방식에 충실했습니다. 
반면, 인스턴스에 없던 메소드가 클래스 내에도 존재하지 않는다면 역시 오류가 발생하게 될 것입니다.  

그간 모르고 지나쳤던 부분이라 신기하기만 합니다.  



#### Reference  
[School of Web 파이썬 - OOP Part 3. 클래스 변수(Class Variable)](http://schoolofweb.net/blog/posts/%ED%8C%8C%EC%9D%B4%EC%8D%AC-oop-part-3-%ED%81%B4%EB%9E%98%EC%8A%A4-%EB%B3%80%EC%88%98class-variable/)  
[4. 클래스 네임스페이스](https://wikidocs.net/1743)
