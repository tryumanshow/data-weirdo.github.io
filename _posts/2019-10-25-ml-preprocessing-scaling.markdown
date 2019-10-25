---

layout: post
title:  "Preprocessing- 스케일링"
subtitle:   "Preprocessing-스케일링"
categories: ml
tags: preprocessing
comments: true

---  

- 데이터 스케일링에 대한 정리입니다.  

---  

사이킷런에서 제공하는 4가지 피쳐 스케일링에 대한 정리입니다.  

## 1. Standard Scaler  

- 데이터 내 각 피쳐들이 정규분포를 따른다는 가정  
- 해당 분포들을 N(0,1)로 돌려주는 과정  
  ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bx_i-mean%28x%29%7D%7Bstd%28x%29%7D)  
- 데이터들이 정규분포를 따르지 않는다면 그리 좋은 scaler는 아님  
  ![](https://miro.medium.com/max/351/1*zlZSLJ4923EJ4rVJOFlJeA.png)  

## 2. Min-Max Scaler  

- 가장 인기있는 스케일링 알고리즘 중의 하나  
  ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bx_i%20-%20min%28x%29%7D%7Bmax%28x%29-min%28x%29%7D)  
- 범위를 0에서 1사이로 줄임 (음수값이 있다면 -1~1도 가능합니다.)  
- Standard Scaler가 잘 작동하지 않는 곳에서 Standard Scaler보다 잘 작동  
  (정규분포를 따르지 않거나 표준편차가 매우 작은 경우)  
- outlier에 민감. 따라서 outlier들이 많다면 Robust Scaler 사용 고려  
  ![](https://miro.medium.com/max/343/1*a_0N3pDkH5ySpuV55qEPqQ.png)  

## 3. Robust Scaler  

- Min-Max Scaler와 비슷하지만 최대·최소값 대신 분위수값을 사용  
- outlier에 robust  
  ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bx_i%20-%20Q_1%28x%29%7D%7BQ_3%28x%29-Q_1%28x%29%7D)  
- Scaling을 해주면 outlier들은 Scaling 이후의 새로운 분포에서 데이터의 대부분이 모여있는 곳의 바깥으로 나가버림  
  ![](https://miro.medium.com/max/495/1*7Ofb8EPbUDjjQDvCqzbZWg.png)  

## 4. Normalizer  

  - feature가 n개라면 n 차원의 공간 내에서 n 개의 점들을 normalize 시켜주는 방식  
  - 예를 들어 3개의 피쳐 x,y,z가 있으면 해당 공간에 대응되는 점 (x1, y1, z1)을 다음과 같이 만들어줍니다.  
    ![](https://latex.codecogs.com/gif.latex?%28x_1%2C%20y_1%2C%20z_1%29) → ![](https://latex.codecogs.com/gif.latex?%28x_1%27%2C%20y_1%27%2C%20z_1%27%29)  
    , ![](https://latex.codecogs.com/gif.latex?x_1%27%3D%5Cfrac%7Bx_1%7D%7B%5Csqrt%7Bx_1&plus;y_1&plus;z_1%7D%7D%2Cy_1%27%3D%5Cfrac%7By_1%7D%7B%5Csqrt%7Bx_1&plus;y_1&plus;z_1%7D%7D%2Cz_1%27%3D%5Cfrac%7Bz_1%7D%7B%5Csqrt%7Bx_1&plus;y_1&plus;z_1%7D%7D)  
  - 각 포인트들은 n차원의 카르테시안 좌표계 속에서 모두 1-unit 안에 들어오게 됩니다.   
    (3차원이라면 원점으로부터 반지름 1인 구 내에 모든 점들이 위치)  
    ![](https://miro.medium.com/max/471/1*tu0QKOibSvubLiOcfhykXw.png)  
  
  
    
