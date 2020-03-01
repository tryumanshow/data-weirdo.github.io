---
layout: post
title:  "알고리즘3 - 병합 정렬, sort()함수, 힙 정렬"
subtitle:   "algorithm-3"
categories: cs
tags: algorithm
comments: true
---  
- 병합 정렬과, C++ STL sort() 함수, 그리고 힙 정렬에 대한 글입니다.  
---  

# 8. 병합 정렬(Merge Sort)  
- `분할 정복` 방법 사용  
  : 역시 `O(NlogN)` ← 보장  
- '정확히' '반' 씩 나눈다.  
  
|-|퀵 정렬|병합 정렬|  
|-|------|---------|  
|시간복잡도|O(NlogN)|O(NlogN)|  
|최악의 경우|O(N^2): 편향되게 분할할 가능성|O(NlogN) 보장 !!|  
|피벗 값|있음|없음|  


- Idea:  
  `일단 반으로 나누고 나중에 합쳐서 정렬하면 어떨까?`  

- ex) `7 6 5 8 3 5 9 1` 정렬하기  
  - 시작: |7|6|5|8|3|5|9|1|  
  - 1단계: 6과 7, 5와 8, 3과 5, 9와 1을 비교  
    - |6 7|5 8|3 5|1 9|  
  - 2단계: 6, 7, 5, 8을 비교, 3, 5, 9,1을 비교  
    - |5 6 7 8|1 3 5 9|  
  - 3단계: 모두 한 데 모아 비교  
    - |1 3 5 5 6 7 8 9|  
    
  - 이 경우 데이터 개수: 8개 (`N`)  
    단계의 수: 밑이 2이고 지수가 8인 로그 (logN)  
  
  ![](http://drive.google.com/uc?export=view&id=1kCMy5IM_wT_o_arkA16kex7fg1KlOD-n)  
  
    - i, j 인덱스 옮겨가면서 총 4번만 비교해주면 돼~  
      
      ![](http://drive.google.com/uc?export=view&id=14Iq07MTFJHPZF-lhGeyaGJDnv5xIF6oV)  
    
  ```  
  #include <stdio.h>

  int number = 8;
  int sorted[8]; // 정렬 배열은 반드시 전역 변수로 선언 

  void merge(int a[], int m, int middle, int n){
    int i = m;
    int j = middle + 1; 
    int k = m;
    // 작은 순서대로 배열에 삽입 
    while(i <= middle && j <= n){
      if(a[i] <= a[j]){
        sorted[k] = a[i];
        i++;
      } else{
        sorted[k] = a[j];
        j++;
      }
      k++;
    }
    // 남은 데이터도 삽입
    if(i > middle){
      for (int t = j ; t <= n; t++){
        sorted[k] = a[t];
        k++;
      }
    } else {
      for(int t=i; t<=middle; t++){
        sorted[k] = a[t];
      }
    }
    // 정렬된 배열을 삽 입
    for(int t=m; t<=n; t++){
      a[t] = sorted[t];	
    } 
  }

  void mergeSort(int a[], int m, int n)
  {
    // 크기가 1보다 큰 경우
    if (m<n)
    {
      int middle = (m+n)/2;
      mergeSort(a, m, middle);
      mergeSort(a, middle + 1, n);
      merge(a, m, middle, n);
    } 
  }


  int main(void){
    int array[number] = {7, 6, 5, 8, 3, 5, 9, 1};
    mergeSort(array, 0, number - 1);
    for (int i=0; i<number; i++)
    {
      printf("%d ", array[i]);
    }
    return 0;
  }
  ```  
  
- 병합 정렬을 구현할 때, 정렬에 사용되는 배열은 `전역 변수`로 선언할 것.  
  - 함수 안에서 배열을 선언학 ㅔ된다면, 매 번 배열을 선언해야하기 때문에 비효율적 (메모리 낭비)  
- 병합 정렬은 '기존의 데이터를 담을 추가적인 배열 공간이 필요하다.'는 점에서 메모리 활용이 비효율적이라는 문제.  
  - cf. `힙 정렬`은 메모리의 비효율성 문제를 해결해줄 것  
  
- 병합 정렬의 시간 복잡도: `O(NlogN)` 보장  


