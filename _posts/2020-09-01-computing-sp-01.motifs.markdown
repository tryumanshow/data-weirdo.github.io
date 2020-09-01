---
layout: post
title:  "컴퓨터 시스템에 대한 개괄"  
subtitle:   "sp1"
categories: computing
tags: system_programming
comments: true
---

- 컴퓨터 시스템에 대한 Overview입니다.

---  

### 컴퓨터 상에서 Source File은 어떻게 실행이 될까?  
- 예를 들어, 다음과 같은 `Source File`이 있다고 가정  
  
  ```  
  # include <stdio.h>
  
  int main()
  {
  printf("hello, world\n");
  return 0;
  }
  ```  
  
  - `Source File`은 텍스트들로 이루어진 `Text File` (문자들의 Sequence로 만들어진 파일)  
  - `Text file`은 ASCII 코드의 Sequence로 저장된다.  
  - ASCII 코드는 8비트의 2진수로 저장된다.  
    ![SP01-1](https://user-images.githubusercontent.com/43376853/91790536-009a4000-ec4c-11ea-8ed1-e4dbd20c7678.png)  

  
  
