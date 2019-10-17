	---
	
	layout: post
	title:  "Javascript-실행방법과 실습환경"
	subtitle:   "javascript-implement"
	categories: cs
	tags: javascript
	comments: true
	
---

Javascript를 실행하는 방법과 실습 환경에 대해 정리해보았습니다. 

---

# 자바스크립트의 실행 방법  

브라우저에서 자바스크립트를 실행하기 위해서는 두 가지가 필요합니다.
첫번째는 브라우저, 두번째는 자바스크립트 코드를 작성할 수 있는 에디터 입니다.
에디터의 경우 윈도우에서는 메모장, 맥북에서는 텍스트 에디트, 리눅스에서는 우분투를 사용하면 됩니다.

우선 자바스크립트를 실행할 때 사용해볼 코드를 소개하겠습니다. 

```
	<!DOCTYPE html>
	<html>
    <head>
	    <meta charset="utf-8"/>
	  </head>
	  <body>
	    <script>
	      alert('Hello world');
	    </script>
	  </body>
	</html>

```  

여기서 자바스크립트에 해당하는 부분은 alert('Hello world');
이며 나머지 바깥쪽 부분은 HTML의 내용들입니다.
기본적으로 자바스크립트는 html 위에서 동작하기 때문에 기본적으로 HTML 코드가 필요합니다. 

## 1. 웹페이지에서 자바스크립트 작성 

이 내용을 입력한 후에 html 파일로 저장하고, 크롬을 열어서 `ctrl + o`를 한 뒤
저장한 파일을 열면 'Hello world!'가 출력될 것입니다. 그렇다면 성공입니다!
맥에서는 ctrl + o 대신 `command + o`를 사용하면 됩니다!

`자바스크립트를 실행하는 방법: 코드를 저장한 뒤 자신의 os에 맞는 커맨드를 입력한 후 저장 파일을 연다.` 

## 2. 크롬 개발자 도구 사용 

가끔 코드를 작성하다보면 파일로 직접 작성하기 귀찮을 때도 있다고 합니다. 
이 때는 파일을 거치지 않고 웹에서 바로 javascript를 실행해볼 수 있다고 합니다.  

먼저 F12를 누르면 개발자 도구가 열립니다 (맥: command + alt + R) 그 후,
상단의 `Show console`을 클릭하면 밑에 에디터 창이 뜰텐데, 이 창에는
HTML 코드가 전혀 필요하지 않고 순수하게 Javascript 코드만 작성하면 됩니다. 
즉석에서 자바스크립트를 실행해볼 수 있습니다. 
다만 긴 코드는 파일로 작성해서 작업하는 게 더 편할 것입니다. 
이 기느에 대해서 더 알고 싶다면 [이곳](https://opentutorials.org/course/580)을 참고하면 됩니다.
한 번 작성했던 코드를 또 쓰고 싶다면 키보드의 화살표 ↑키를 누르면 되고 몇 번이고 거슬러 올라가는 것이 가능합니다.  

```
cf.
	<!DOCTYPE html>
	<html>
    <head>
	    <meta charset="utf-8"/>
	  </head>
	  <body>
	    <script>
	      console.log('Hello world');
	    </script>
	  </body>
	</html>

위의 alert 대신 console.log를 쓰면 경고창 대신 Hello world라는 문구만 출력하게 됩니다. 
```  

ps.
`통합개발도구(Integrated Development Environment, IDE)`는 코드를 작성함에 있어서 필요한 여러가지 기능들을 
다 갖추고 있는 도구를 이야기 한다. 프로그래밍을 함에 있어서 좋은 도구를 선택하는 것은 굉장히 중요한 일이며, 
좋은 도구란 해당 도구가 제공하는 모든 기능들을 파악하기 위한 충분한 노력과 시간을 들일 수 있을 때 비로소 
그 도구는 좋은 도구가 된다. (유료이거나 Heavy한 에디터를 사용하는데 그 에디터가 제공하는 기능들을 잘 알지 못한다면,
그 도구는 좋은 도구라 할 수 없다.)
