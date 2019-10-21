---

layout: post
title:  "Javascript-숫자와 문자"
subtitle:   "javascript-implement"
categories: cs
tags: javascript
comments: true

---

- Javascript에서의 숫자와 숫자 연산, 문자와 문자 연산에 대해 살펴보았습니다.  

## 시작하기 전에

```
여담1. 데이터는 정보이고, 정보는 그 마다의 타입이 있는데, 숫자, 문자도 하나의 타입입니다.  
여담2. 에디터를 사용하면, 에디터는 자주 사용하는 작업들을 알고 있어서 html만 입력하고 tab 키를 누르면
자동으로 기본적인 form이 만들어진다고 말씀하셨습니다. 하지만, 이는 Sublime Text에 대한 말씀이셨고,
vscode에서는 HTML Snippets이라는 플러그인을 다운 받으면 동일한 기능을 실행할 수 있었습니다. 
```
```
<!DOCTYPE html>  
<html>  
    <head>  
        <title></title>  
    </head>  
    <body>  
        <script type ="text/javascript">  
            alert(1);  
        </script>  
    </body>  
</html>  
```
---

alert(1)을 제외하고는 HTML 부분이고, 따라서 javascript 파트인 alert(1)에만 신경 쓰도록 합니다.  

기본적으로 alert는 경고창을 띄우는 코드였고, 경고창을 출력하기 싫다면 alert대신 console.log를 사용하면 됩니다.

---  
---  

# 숫자

정수를 입력하고 싶다면 alert()의 () 사이에 그냥 정수를 (ex. alert(1);) 
실수를 입력하고 싶다면 실수를 입력하면 됩니다. (ex. alert(1.1);) 
더하기는 '+' (ex. alert(1+1)), 곱하기는 '$$*$$'(asterisk) (ex. alert(2*4)), 나누기는 '/'를 사용하면 됩니다.
통계학을 공부하며 R로 시작해, 파이썬을 거치고 Javascript를 공부하고 있기 때문에 이 form에 대해서는 익숙합니다.

## 숫자 연산

기본적으로 컴퓨터는 애초에 계산을 위해 고안되었기 때문에, 계산과 관련된 기능들을 많이 갖고 있습니다.
Javascript를 통해 연산을 하고 싶다면 `Math`라는 명령어를 쓰고 그 뒤에 하고 싶은 연산의 명령어를 써주면 됩니다.
Math는 수학과 관련된 명령들의 카테고리입니다.

다음은 몇 가지 예입니다.

```
Math.pow(3, 2);  // 9, 3의 2승
Math.round(10.6); // 11, 10.6을 반올림
Math.ceil(10.2); // 11, 10.2를 올림
Math.floor(10.6); // 10, 10.6을 버림
Math.sqrt(9); // 3, 9의 제곱근
Math.random(); // 0과 1.0 사이의 랜덤한 숫자를 출력
```  
이를 활용하여, 100보다 작은 난수를 생성하고 싶다면, 100$$*$$Math.random()을 사용하면 되고,
0과 100 사이의 정수를 출력하고 싶다면, Math.round(100$$*$$Math.random())과 같은 형식으로 접근하면 됩니다.  

---  
---  

# 문자  
문자는 반드시 큰 따옴표(")나 작은 따옴표(') 둘 중 하나로 감싸야 합니다. 단, 둘을 혼용해서 감싸면 안됩니다. 
```
<!DOCTYPE html>
<html>  
    <head>
        <title></title>
    </head>
    <body>
        <script type ="text/javascript">
            alert('coding everybody');
        </script>
       
    </body>
</html>
```  
자바스크립트에 있어서 '나 "를 입력해준다는 것은 (이제 문자열 입력을 시작하겠다)라는 의미입니다. 따라서, 
`alert('coding everybody")`와 같이 입력하면 문자열 입력을 시작했지만 그 끝맺음을 하지 않은 것이 됩니다. 
다음과 같은 에러가 출력될 것입니다.  
![](https://s3.ap-northeast-2.amazonaws.com/opentutorials-user-file/module/532/1501.gif)  

만약 어떤 이유에 의해서 작은 따옴표 둘 사이에 작은 따옴표를 넣고 싶다면 어떻게 해야 할까요? 
다음과 같은 방식으로 오류를 피할 수 있습니다. 
`alert('Woo\'s coding everybody')` 이 때는 의도했던 대로 Woo's coding everybody가 출력됩니다. 
\(역슬래시) 바로 다음의 문자 하나는 그냥 문자 그 자체로서 해석되고 해당 예에서 \뒤에 '를 위치 시킴으로써
'가 원래 갖고 있었던 임무로부터 탈출했기 때문에 이를 `escape`라고 부릅니다.  

(cf. \n: 줄바꿈)

#### 숫자와 문자

1 과 '1'은 명백히 다른 데이터 형식입니다. 데이터의 형식은 `typeof` 라는 명령어를 통해 확인할 수 있습니다. 

alert(typeof '1') => string 출력  
alter(typeof 1) => number 출력  

## 문자 연산  
숫자 연산 뿐만 아니라 문자 연산 또한 가능합니다. 
예를 들어, `alert('coding' + ' everybody')`; 와 같이 입력하면 coding everybody를 출력할 것입니다.  
위 코드는 다음과 같이 표현해볼 수도 있습니다. `alert('coding' + ' ' + 'everybody')`

추가적으로 `'coding everybody'.length`를 입력하면 'coding everybody'의 길이인 16이 출력됩니다.  



문자열 및 숫자와 관련된 명령어들의 리스트는 [이곳](https://opentutorials.org/course/50/37)을 참고하면 됩니다.  
일단 지금은 이 모두를 다루지는 않지만, 이 명령어들을 조합하면 하나의 프로그램이 만들어집니다. 


#### Reference
[생활코딩-숫자와문자](https://opentutorials.org/course/743/4647)
