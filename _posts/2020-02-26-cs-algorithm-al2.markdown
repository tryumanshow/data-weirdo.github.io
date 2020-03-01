---
layout: post
title:  "알고리즘2 - 퀵 정렬과 기초 정렬 문제 풀어보기"
subtitle:   "algorithm-2"
categories: cs
tags: algorithm
comments: true
---  

- 퀵 정렬과, 기초 정렬 문제 풀이에 대한 글입니다.  

---  

# 5. 퀵 정렬(Quick Sort)의 시간 복잡도와 작동 원리  
- 앞의 선택, 버블, 삽입정렬 모두 시간 복잡도 O(N^2)  
  - 하지만, 데이터 수가 많아지면 상당히 비효율적  
    (데이터가 10만 개만 넘어가도 일반적인 상황에서 사용하기 매우 어려움)  
  - 더 빠른 알고리즘 필요  
  
- 퀵 정렬  
  - `분할정복` 알고리즘 사용  
  - 시간복잡도: `O(NlogN)`  
    - cf) NlogN은 상당히 효율적인 것! (거의 상수라고 봐도 무방)  
    
      ```  
      2^10: 약 1000  
      2^20: 약 1000000  
      
      밑이 2인 log: 지수가 2^20일 때, 20밖에 안됨!  
      ```  
       
- Idea:  
  `특정한 값을 기준으로 큰 숫자와 작은 숫자를 나누면 어떨까?`  
  
  - 퀵 정렬에는 `기준 값` (혹은 `피벗(Pivot)`)이 있음.  
    - 그 특정값을 기준으로 큰 숫자와 작은 숫자로 나눈다. (분할)  
    - 보통 첫 번째 원소를 피벗 값으로 설정함.  
    
- ex1) `3 7 8 1 5 9 6 10 2 4` 정렬하기  

  - Process  
    - '분할'이 일어났을 때, 그 기준이 되는 값, 혹은 고정된 값을 []로 표시  
    - '피벗 값'을 ''로 표시  
    
    - 피벗값을 기준으로, 왼쪽에서 오른쪽으로, 또 오른쪽에서 왼쪽으로 이동  
      - 왼쪽에서 오른쪽 이동 시: 피벗값보다 큰 값을 선택을 한다.  
      - 오른쪽에서 왼쪽으로 이동 시: 피벗값보다 작은 값을 선택한다.  
	- 이 때의 큰값과 작은 값을 서로 바꾸어 줌  
    - 엇갈림 발생시, 더 작은 값과 피벗값의 위치를 바꾸어주면 됨  
    
  - '3' 7 8 1 5 9 6 10 2 4  
  - '3' 2 8 1 5 9 6 10 7 4  
  - '3' 2 1 8 5 9 6 10 7 4  // 엇갈림!  
  - 1 2 [3] 8 5 9 6 10 7 4  
  - '1' 2 [3] '8' 5 9 6 10 7 4  
  - [1] '2' [3] '8' 5 9 6 10 7 4  
  - [1] [2] [3] '8' 5 9 6 10 7 4  
  - [1] [2] [3] '8' 5 4 6 10 7 9  
  - [1] [2] [3] '8' 5 4 6 7 10 9  // 엇갈림!  
  - [1] [2] [3] '7' 5 4 6 [8] '10' 9 // 8의 좌측에서, 또 '엇갈림'  
  - [1] [2] [3] 6 5 4 7 [8] '10' 9  
  - ...  
  
  - 한 번 정렬을 수행했을 때, 그 값을 기준으로 왼쪽과 오른쪽이 나뉘어진다!  
  
  
- ex2) `1 2 3 4 5 6 7 8 9 10`  
  - if 선택정렬:  N^2: 대략 100  
  - if 퀵정렬: 5^2 + 5^2 = 50 (분할정복)  
    - 쪼개서 연산을 하고 나중에 합치면 결론적으로 연산의 수는 더 적다.  
    
    - `N*logN`에서  
      - N: 데이터의 수  
      - logN: 반씩 쪼개서 들어가기 때문에 절약되는 부분  
  
    - 백만개 정도도 어렵지 않게 수행할 수 있다!  
    
# 6. 퀵 정렬 (Quick Sort)의 구현 및 한계점 분석  

```  
#include <stdio.h>

int number= 10;
int data[10] = {1, 10, 5, 8, 7, 6, 4, 3, 2, 9};

void quickSort(int *data, int start, int end){
	if (start >= end) // 원소가 한 개인 경우 
	{
		return;
	}
	
	int key = start; // 키는 첫번째 원소
	int i = start + 1; // i: 왼쪽 출발 지점  
	int j = end; // j: 오른쪽 출발 지점   
	int temp; 
	
	while(i <= j){ // 엇갈릴 때까지 반복  
		while(data[i] <= data[key]) // 키 값보다 큰 값 만날 때까지  
		{
			i++;
		}
		while(data[j] >= data[key] && j > start) // 키 값보다 작은 값을 만날 때까지 
		{
			j--;
		}
		if (i>j) // 현재 엇갈린 상태면 키 값과 교체 
		{
			temp = data[j];
			data[j] = data[key];
			data[key] = temp;
		} else {
			temp = data[j];
			data[j] = data[i];
			data[i] = temp;
		}
	}
	// 재귀적 함수 이용 
	quickSort(data, start, j-1);
	quickSort(data, j+1, end);  
}


int main(void){
	quickSort(data, 0, number -1);
	for(int i=0; i<number; i++)
	{
		printf("%d ", data[i]);
	}
	return 0;
}   
```  
    
- 한계점  
  - 퀵정렬은 피벗값의 설정에 따라, `최악의 경우`에는 시간 복잡도가 `O(N^2)`가 나올 수 있다.  
    (평균 시간 복잡도는 O(NlogN))  
    
  - 흔히 알고리즘 대회에서 O(NlogN)을 요구하는 경우 퀵 정렬을 이용하면 틀리기도 함  
  
- ex) `1 2 3 4 5 6 7 8 9 10`  
  - '1' 2 3 4 5 6 7 8 9 10  
  - [1] '2' 3 4 5 6 7 8 9 10  
  - [1] [2] '3' 4 5 6 7 8 9 10  
  
  ... 이런 경우 분할 정복의 이점을 전혀 사용하지 못하고 반복적으로 O(N^2)만큼 수행하게 됨  
  - 이 경우는 오히려 `삽입 정렬`이 더 빠르다!  
  
  ```  
  '항상' A라는 정렬이 B라는 정렬보다 빠르다! 라는 식으로 구분하지 말 것.  
  ```  
    
- 내림차순~  

  ```  
 	while(i <= j){ 
	while(data[i] >= data[key]) // 이 부분  
	{
		i++;
	}
	while(data[j] <= data[key] && j > start) // 이 부분 
	{
		j--;
	}
  ```  
  
  - 두 부분만 바꾸면 내림차순으로 쉽게 바뀐다~  
  
# 7. 기초 정렬 알고리즘 문제 풀이  
- [백준 온라인 저지](www.acmicpc.net)  
  - '단계별로 풀어보기' 탭 - '정렬' 탭 - '수 정렬하기'  
    - 주어지는 데이터: 1000개 미만  
      이 말은, n^2을 해도 백만 밖에 안 됨  
    - 일반적으로 이런 온라인 채점 시스템의 경우 1초에 대략 1억 번 정도 연산을 할 수 있다고 가정해도 됨  
      이 때, 백만 번은 1억에 비해 매우 작은 수!  
      - 따라서, 선택정렬, 버블정렬, 삽입정렬 중 하나를 선택해서 풀어도 큰 무리가 없을 것  
      
  - 선택정렬 case  
  
    ```  
    #include <stdio.h>

		int array[1001];

		int main(void){
			int number, i, j, min, index, temp;;
			scanf("%d", &number);
			for(i=0; i<number; i++)
			{
				scanf("%d", &array[i]);
			}

			for(i=0; i<number; i++)
			{
				min = 1001; // 절대값이 1000보다 작거나 같은 수이기 때문에  
				for (j=i; j<number; j++)
				{
					if (min > array[j])
					{
						min = array[j];
						index = j;
					}
				}
				temp = array[i];
				array[i] = array[index];
				array[index] = temp;
			}

			for(i=0; i<number; i++)
			{
				printf("%d\n", array[i]);
			}

			return 0;
		}
    ```  
		
	- 세 숫자 정렬 case  
		: 앞의 것 일부분을 바꾸어주면 됨  
	
		```  
		#include <stdio.h>

		int array[3];

		int main(void){
			int i, j, min, index, temp;;
			for(i=0; i<3; i++)
			{
				scanf("%d", &array[i]);
			}

			for(i=0; i<3; i++)
			{
				min = 1000001; // 절대값이 1000보다 작거나 같은 수이기 때문에  
				for (j=i; j<3; j++)
				{
					if (min > array[j])
					{
						min = array[j];
						index = j;
					}
				}
				temp = array[i];
				array[i] = array[index];
				array[index] = temp;
			}

			for(i=0; i<3; i++)
			{
				printf("%d ", array[i]);
			}

			return 0;
		}
		```  
		
	- 100만 개 정렬(#2751)  
		- 시간 복잡도 NlogN을 요구  
			(수의 개수가 백만 개가 주어지면, 무조건 nlogn으로 풀어야 돼!)  
			-  퀵, 병합, 힙 정렬에 기반한 알고리즘이어야 정답으로 인정받을 수 있다!  
				- 다만, 퀵 정렬 같은 경우 최악의 경우 NlogN을 보장할 수 없기 때문에 병합 정렬이나 힙 정렬을 사용해야 할 것  
		- 그럼에도 불구하고 '퀵 정렬' 사용 해보겠음.  
			 
		```  
		#include <stdio.h>

		int number, data[1000001];

		void quickSort(int *data, int start, int end){
			if (start >= end)
			{
				return;
			}
			int key = start;
			int i = start + 1, j= end, temp;
			while(i <= j){
				while(data[i] < data[key]){
					i++;
				}
				while(data[j] >= data[key] && j > start)
				{
					j--;
				}
				if(i>j){
					temp = data[j];
					data[j] = data[key];
					data[key] = temp;
				} else {
					temp = data[i];
					data[i] = data[j];
					data[j] = temp;
				}
			}
			quickSort(data, start, j-1);
			quickSort(data, j+1, end);
		}

		int main(void){
			scanf("%d", &number);
			for (int i=0; i<number; i++)
			{
				scanf("%d", &data[i]);
			}
			quickSort(data, 0, number-1);

			for (int i=0; i<number; i++)
			{
				printf("%d\n", data[i]);
			}
			return 0; 
		} 
		```  
   
	 	- cf) 위와 같이 퀵정렬을 사용해서 푸는 방법 이외에도, 
			C++의 알고리즘 라이브러리를 활용해서 (`#include <algorithm`) 굉장히 쉽고 빠르게 정렬을 할 수 도 있음.  
			- sort 함수: 퀵 정렬이 갖고 있는, O(N^2)라는 한계점을 효과적으로 해결  
			
			```  
			#include <stdio.h>
			#include <algorithm>

			int number, data[1000000];

			int main(void){
				scanf("%d", &number);
				for (int i=0; i<number; i++){
					scanf("%d", &data[i]);
				}
				std::sort(data, data+number);
				for (int i=0; i<number; i++){
					printf("%d\n", data[i]);
				}
				return 0;
			} 
			```  
