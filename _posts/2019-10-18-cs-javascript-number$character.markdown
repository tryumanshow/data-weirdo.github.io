---

layout: post
title:  "Javascript-숫자와 문자"
subtitle:   "javascript-implement"
categories: cs
tags: javascript
comments: true

---

Javascript 내에서의 숫자 및 연산, 그리고 문자에 대해 살펴보았습니다. 

## 시작하기 전에

```
여담1. 데이터는 정보이고, 정보는 그 마다의 타입이 있는데, 숫자, 문자도 하나의 타입입니다.  
여담2. 에디터를 사용하면, 에디터는 자주 사용하는 작업들을 알고 있어서 html만 입력하고 tab 키를 누르면
자동으로 기본적인 form이 만들어진다고 말씀하셨습니다. 하지만, 이는 Sublime Text에 대한 말씀이셨고,
vscode에서는 HTML Snippets이란느 플러그인을 다운 받으면 동일한 기능을 실행할 수 있었습니다. 
```
---
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

---

alert(1)을 제외하고는 HTML 부분이고, 따라서 javascript 파트인 alert(1)에만 신경 쓰도록 합니다.  

기본적으로 alert는 경고창을 띄우는 코드였고, 경고창을 출력하기 싫다면 alert대신 console.log를 사용하면 됩니다.

# 숫자

정수를 입력하고 싶다면 alert()의 () 사이에 그냥 정수를 (ex. alert(1);) 
실수를 입력하고 싶다면 실수를 입력하면 됩니다. (ex. alert(1.1);)
더하기는 '+' (ex. alert(1+1)), 곱하기는 '*'(asterisk) (ex. alert(2*4)), 나누기는 '/'를 사용하면 됩니다.
통계학을 공부하며 R로 시작해, 파이썬을 거치고 Javascript를 공부하고 있기 때문에 이 form에 대해서는 익숙합니다.

# 숫자 연산

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
이를 활용하여, 100보다 작은 난수를 생성하고 싶다면, 100*Math.random()을 사용하면 되고,
0과 100 사이의 정수를 출력하고 싶다면, Math.round(100*Math.random())과 같은 형식으로 접근하면 됩니다. 
