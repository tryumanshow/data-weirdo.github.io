---
layout: post
title:  "다변량분석에 대한 간단한 소개"
subtitle:   "bayesian-gaussian&poisson"
categories: statistics
tags: multivariate
comments: true

---

- 다변량 분석에 대한 간단한 소개글입니다.  

---

- Multivariate Data?  
  - 변수 개수가 둘 이상인 경우  
  - 다변량 데이터 분석시, 그 기저의 다변량 `분포`(multivariate distribution)에 관심  
  - 샘플에 근거해 다변량 분포에 대해 `추론(inference)`하는 것이 목적  
    - ex) Parametric Estimation  
    - ex) Hypothesis testing  
    
- 다변량 데이터 표기  
  - n items, p variables  (nxp)  
    
    ![](https://latex.codecogs.com/gif.latex?X%3D%5Cbegin%7Bbmatrix%7D%20x_%7B11%7D%20%26%20x_%7B12%7D%20%26%20...%20%26%20x_%7Bij%7D%20%26%20...%20%26%20x_%7B1p%7D%5C%5C%20x_%7B21%7D%20%26%20x_%7B22%7D%20%26%20...%20%26%20x_%7B2j%7D%20%26%20...%20%26%20x_%7B2p%7D%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%26%20%5Cvdots%20%26%20%26%20%5Cvdots%5C%5C%20x_%7Bn1%7D%20%26%20x_%7Bn2%7D%20%26%20...%20%26%20x_%7Bnj%7D%20%26%20...%20%26%20x_%7Bnp%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20X_1%27%5C%5C%20X_2%27%5C%5C%20%5Cvdots%20%5C%5C%20X_n%27%5C%5C%20%5Cend%7Bbmatrix%7D)  
    

- 다변량 데이터의 기술 통계량 표기  

  ![](https://latex.codecogs.com/gif.latex?%5Cbar%7B%24x%24%7D%20%3D%5Cbegin%7Bbmatrix%7D%20%5Cbar%7Bx_1%7D%5C%5C%20%5Cbar%7Bx_2%7D%5C%5C%20%5Cvdots%20%5C%5C%20%5Cbar%7Bx_p%7D%5C%5C%20%5Cend%7Bbmatrix%7D%20%2C%20S%3D%5Cbegin%7Bbmatrix%7D%20s_%7B11%7D%20%26%20s_%7B12%7D%20%26%20...%20%26%20s_%7B1p%7D%20%5C%5C%20s_%7B21%7D%20%26%20s_%7B22%7D%20%26%20...%20%26%20s_%7B2p%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%26%20%5Cvdots%20%5C%5C%20s_%7Bp1%7D%20%26%20s_%7Bp2%7D%20%26...%26%20s_%7Bpp%7D%20%5Cend%7Bbmatrix%7D%2C%20R%3D%5Cbegin%7Bbmatrix%7D%201%20%26%20r_%7B12%7D%20%26%20...%20%26%20r_%7B1p%7D%20%5C%5C%20r_%7B21%7D%20%26%201%20%26%20...%20%26%20r_%7B2p%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%26%20%5Cvdots%20%5C%5C%20r_%7Bp1%7D%20%26%20r_%7Bp2%7D%20%26...%26%201%20%5Cend%7Bbmatrix%7D)  
  
  - 각각 `sample mean vector`, `sample(variance-)covariance matrix`, `sample correlation matrix`  
  - S: symmetric  
  - ![](https://latex.codecogs.com/gif.latex?r_%7Bjj%7D) for 1≤j≤p  


- 거리 측정  
  ```  
  1. Euclidean Distance  
  2. Statistical Distance (Scaled version of Euclidean Distanc)
  3. Rotating
  ```  
  
  - __1__. Euclidean Distance  
    - ex) 원점으로부터 ![](https://latex.codecogs.com/gif.latex?%24x%24%20%3D%20%28x_1%2C%20x_2%2C%20...%2C%20x_p%29%5ET)까지의 유클리디안 거리  

      ![](https://latex.codecogs.com/gif.latex?d%28%5Cmathbf%7B0%7D%2C%20%5Cmathbf%7Bx%7D%29%20%3D%20%5Csqrt%7Bx_1%5E2&plus;x_2%5E2&plus;...&plus;x_p%5E2%7D%20%3D%20%5Csqrt%7B%5Cmathbf%7Bx%7D%5ET%5Cmathbf%7Bx%7D%7D%20%3D%20%5Cleft%20%5C%7C%20%5Cmathbf%7Bx%7D%20%5Cright%20%5C%7C)  

      - 유클리디안 거리의 단점: 모든 좌표축이 거리에 동일하게 영향을 미친다는 점  
        - `두 축 각각의 변동성을 고려하지 못함.`  
        - 이는 통계학적으로는 맞지 않음  
        - 변수가 다르면 변동의 정도도 다르다.  
        - Though distance can be same, meaning can be different.  
        
    - 기하학적인 형태?  
      - 이변량의 경우: 원  

  - __2__. Scaled Version Euclidean Distance  
    
    - 편의를 위해 이변량의 경우 가정  
    
      ![](https://latex.codecogs.com/gif.latex?d%28%5Cmathbf%7B0%7D%2C%20%5Cmathbf%7Bx%7D%29%20%3D%20%5Csqrt%7B%5Cfrac%7Bx_1%5E2%7D%7Bs_%7B11%7D%7D&plus;%5Cfrac%7Bx_2%5E2%7D%7Bs_%7B22%7D%7D%7D%20%3D%20%5Csqrt%7B%28%5Cfrac%7Bx_1%7D%7B%5Csqrt%7Bs_%7B11%7D%7D%7D%29%5E2&plus;%28%5Cfrac%7Bx_2%7D%7B%5Csqrt%7Bs_%7B22%7D%7D%7D%7D%29%5E2)  
    
    - 이 때 ![](https://latex.codecogs.com/gif.latex?s_%7B11%7D%2C%20s_%7B22%7D): 고정되지 않은 점으로부터 생김.  
      - 변수 간의 variation을 고려한 Euclidean distance의 scaled version  
    - 두 변수 간 변동성이 같고, 서로 간 독립이라면 Euclidean distance를 쓰는 것은 적절.  
    
    - 기하학적인 형태?  
      - 이변량의 경우: (일반적으로) 타원  
        - Major axis와 Minor axis가 생긴다.  
        
        ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bx_1%5E2%7D%7B9%7D&plus;%5Cfrac%7Bx_2%5E2%7D%7B4%7D%20%3D%201)  
        ![](http://drive.google.com/uc?export=view&id=1st9XHtw_zPHOCDim4zQLch_KJYvo3hTS)  
        
        
