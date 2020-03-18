---

layout: post
title:  "리눅스 원격 접속 방법 및 쉘 꾸미기"
subtitle:   "linux-리눅스 원격 접속 방법 및 쉘 꾸미기"
categories: cs
tags: linux
comments: true

---

- 리눅스 원격 접속 방법 및 쉘 꾸미기에 대한 정리입니다.  

---  

# 리눅스 원격 접속 방법 및 쉘 꾸미기  

## XRDP(원격데스크탑) 접속  

- Remote에 있는 시스템에 접속하는 방법은 여러가지  

__1. RDC Tool 사용하기 (Remote Desktop Connection)__  
  - `Windows + R` → RUN 실행창을 띄우고, `mstsc` 입력  
  - 컴퓨터란에 `IP번호:포트번호` 입력  
  - 본인 계정 및 비밀번호를 입력하면 연결 완료!  
  
__2. SSH로 접속하기(Secured Shell)  
  - PuTTY이용 (인터넷에서 쉽게 다운받을 수 있음)  
  - Host Name에 IP번호, Port에 포트 번호를 입력  
  
### 비밀번호 변경  

```  
1. cmd 실행 (단축키: Ctrl + Alt + T)  
2. 'passwd 본인계정' 입력
3. 현재 패스워드 → 변경할 패스워드 → 변경할 패스워드 재입력 순으로 진행  
```  

##  쉘 꾸미기  

```  
1. .bashrc 파일을 열기(ex: nano.bashrc) 
  - nano: 에디터!
  - nano를 쓰지 않고 vi 에디터를 쓰는 것도 가능!  
  
  - .bashrc 파일은 home directory에 있음  
  
2. 원하는 방식대로 쉘 꾸민 후 저장하기.  
3. cmd를 다시 실행하면 꾸밈이 반영됨  
```  

- 텍스트 에디터: Sublime Text 3, VS Code로 사용 가능   
