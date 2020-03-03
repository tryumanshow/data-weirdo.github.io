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
  - 시작  
    |7|6|5|8|3|5|9|1|  
  - 1단계: 6과 7, 5와 8, 3과 5, 9와 1을 비교  
    |6 7|5 8|3 5|1 9|  
  - 2단계: 6, 7, 5, 8을 비교, 3, 5, 9,1을 비교  
    |5 6 7 8|1 3 5 9|  
  - 3단계: 모두 한 데 모아 비교  
    |1 3 5 5 6 7 8 9|  
    
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
  - 함수 안에서 배열을 선언하게 된다면, 매 번 배열을 선언해야하기 때문에 비효율적 (메모리 낭비)  
- 병합 정렬은 '기존의 데이터를 담을 추가적인 배열 공간이 필요하다.'는 점에서 메모리 활용이 비효율적이라는 문제.  
  - cf. `힙 정렬`은 메모리의 비효율성 문제를 해결해줄 것  
  
- 병합 정렬의 시간 복잡도: `O(NlogN)` 보장  

# 9. C++ STL sort 함수 다루기 1  
- 실제 알고리즘 대회에서 정렬 문제가 나오면, 정렬 알고리즘을 직접 구현하지 말고 `sort()`함수를 써!  
- `정렬`은 컴퓨터 공학의 오래된 연구 분야 → 이미 훌륭한 관련 정렬 라이브러리가 존재.  

- (여기서부터는 C문법이 아닌  C++ 문법을 사용하게 됨)  
  (실제로, 코드를 쓸 때, C, C++ 문법을 적절히 섞어서 사용하기도 함)  
  - ex) 입출력에 관해서는 C의 문법을, 벡터 및 sort와 같은 라이브러리의 경우는 C++의 문법을 사용  
    
- sort() 함수의 기본 사용법?  
  - C++의 algorithm 헤더에 포함되어 있음  

  ```  
  #include <iostream>
  #include <algorithm>

  using namespace std;

  int main(void){

    int a[10] = {9, 3, 5, 4, 1, 10, 8, 6, 7, 2};
    sort(a, a + 10); 
    // a: 배열 자체는 '메모리 주소'를 의미하는 하나의 변수이기 때문에.  
    // a+10: 정렬할 마지막 원소가 있는 '메모리 주소'를 넣어주면 됨   

    for(int i=0; i<10; i++)
    {
      cout << a[i] << ' ';	
    }

    return 0;
  }
  ```  
  
  >> 결과  
    1 2 3 4 5 6 7 8 9 10  

- 정렬할 기준 직접 설정해주기.  

  ```  
  #include <iostream>
  #include <algorithm>

  using namespace std;

  bool compare(int a, int b){
    return a > b;
  }

  int main(void){

    int a[10] = {9, 3, 5, 4, 1, 10, 8, 6, 7, 2};
    sort(a, a + 10, compare); 
    for(int i=0; i<10; i++)
    {
      cout << a[i] << ' ';	
    }

    return 0;
  }
  ```  
  
  >> 결과  
    10 9 8 7 6 5 4 3 2 1이 됨  
    
- 데이터를 묶어서 정렬하기  
  - 위의 예제와 같은 단순 데이터 정렬 기법은 실무에서 사용하기에는 그다지 적합하지는 x  
  - 왜냐하면, 실무에서는 모든 데이터들이 `객체`로 정리되어, 내부적으로 여러 변수를 포함하고 있기 때문  
  - 때문에, `특정한 변수를 기준`으로 정렬하는 것이 중요  
  
  - 여러 학생이 있을 때 학생들을 점수가 낮은 학생부터 정렬하도록 하기  
  
    ```  
    #include <iostream>
    #include <algorithm>

    using namespace std;

    class Student { 
    public:
      string name;
      int score; 
      Student(string name, int score) {
        this -> name = name;
        this -> score = score;
      }
      // 정렬 기준은 '점수가 낮은 순서' 
      bool operator <(Student &student) {
        return this->score < student.score;
      } 
    };

    int main(void){
      Student students[] = {
        Student("김철수", 90),
        Student("박영희", 93),
        Student("나철수", 97),
        Student("배짱구", 87),
        Student("오준성", 92) 
      };
      sort(students, students+5);
      for(int i=0; i<5; i++){
        cout << students[i].name << ' ';
      }

    }
    ```  
    
    - `class`: 여러 개의 변수를 하나로 묶기 위해 사용  
    - class 내의 `Student`: 생성자  
      - 생성자: 특정 객체를 초기화하기 위해 사용  
    - bool operator 부분  
      - '다른 학생과 비교를 할 때 내 점수가 더 낮다면 우선 순위가 높다!'  
      
    - `클래스를 이용해서 정렬하는 방식이 실무에서 많이 활용된다!`  
    
# 10. C++ STL sort 함수 다루기 2  
- 9강에서 했던, 클래스를 정의하는 방식은 프로그래밍 속도 측면에서는 별로 유리하지는 않아.  
- 클래스를 이용하는 방식은 `실무`에 적합한 방식  
  (일반적으로 빠른 개발이 필요한 경우에는 Pair 라이브러리를 사용하는 것이 더 효율적)  
  

- 9강의 클래스를 이용한 이름 정렬을, `Vector`와 `Pair`라이브러리를 이용해 시행  

  ```  
  #include <iostream> // C++ 헤더 
  #include <vector> // C++을 사용해서 프로그래밍할 때는 vector 라이브러리를 많이 사용함 
  #include <algorithm>

  using namespace std;

  int main(void){
    vector<pair<int, string> > v;
    v.push_back(pair<int, string>(90, "김철수"));
    v.push_back(pair<int, string>(93, "박영희"));
    v.push_back(pair<int, string>(97, "나철수"));
    v.push_back(pair<int, string>(87, "배짱구"));
    v.push_back(pair<int, string>(92, "오준성"));

    sort(v.begin(), v.end());
    for(int i=0; i<v.size(); i++)
    {
      cout << v[i].second << ' ';
    }
    return 0;
  } 
  ```  

  - vector: C++을 사용해서 프로그래밍할 때는 vector 라이브러리를 많이 사용  
    - 언제 array를 쓰고, 언제 vector를 쓸 지는 프로그래밍을 하다보면 나중에 자연스레 알게 될 것  
  - pair: 한 쌍의 데이터를 다루기 위해 사용하는 라이브러리  
  
  - `vector<pair<int, string> > v;`  
    - 한 쌍의 데이터를 int형과 string형으로 묶어준 것  
  - `v.push_back()`: `push_back`: 리스트의 마지막 부분에 삽입을 하겠다는 것  
  - `v.size()`: 현재, v라는 벡터 안에 총 몇 개의 데이터가 들어가있는지 (벡터의 크기)를 나타냄  
  - `v.begin()`, `v.end()`  
  
- 소스코드를 짧게 해주는 기법: `숏 코딩 (Short Coding)`  
  - 시간 복잡도가 동일하다면 소스코드가 짧을수록 좋음  
  
- `벡터(Vector) STL`: 마치 배열과 같이 작동  
  - 원소를 선택적으로 `삽입(Push)` 및 `삭제(Pop)`할 수 있음  
  - 단순한 배열을 보다 사용하기 쉽게 개편한 자료구조  
  
- `페어(Pair) STL`
  - 한 쌍의 데이터를 처리할 수 있도록 해주는 자료구조  
  
- `이런 STL을 얼마나 많이 알고있느냐가 굉장히 중요!`  

- 변수가 3개 일 때, 두 개의 변수를 기준으로 정렬하기  
  - 학생정보: 이름, 성적, 생년월일.  
  - 성적이 동일하면, 나이가 어릴 때 더 우선순위가 높도록  

    ```  
    #include <iostream> // C++ 헤더 
    #include <vector> // C++을 사용해서 프로그래밍할 때는 vector 라이브러리를 많이 사용함 
    #include <algorithm>

    using namespace std;

    bool compare(pair<string, pair<int, int> > a,
           pair<string, pair<int, int> > b) {
      if(a.second.first == b.second.first) {
        return a.second.second > b.second.second;
      } else {
        return a.second.first > b.second.first;
      }
    }

    int main(void){
      vector<pair<string, pair<int, int> > > v;
      v.push_back(pair<string, pair<int, int> >("김철수", pair<int, int>(90, 19961222)));;
      v.push_back(pair<string, pair<int, int> >("박영희", pair<int, int>(97, 19930518)));;
      v.push_back(pair<string, pair<int, int> >("나철수", pair<int, int>(95, 19930203)));;
      v.push_back(pair<string, pair<int, int> >("배짱구", pair<int, int>(90, 19921207)));
      v.push_back(pair<string, pair<int, int> >("오준성", pair<int, int>(88, 19900302)));

      sort(v.begin(), v.end(), compare);
      for(int i=0; i<v.size(); i++)
      {
        cout << v[i].first << ' ';
      }
      return 0;
    } 
    ```  
  
    - `vector<pair<string, pair<int, int> > > v;`: 이중페어 사용한 것  
    - 이렇듯, 정렬의 기준이 많다고 하더라도, 적절하게 pair를 섞으면 good  
      - 하지만, 기준이 4개가 넘어가면 클래스를 만드는 게 더 나을 수도 있음. (페어 지나치게 복잡해짐)  
