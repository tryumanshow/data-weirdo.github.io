---
layout: post
title:  "랜덤벡터와 샘플"
subtitle:  "multivariate-randomvector"
categories: statistics
tags: multivariate
comments: true

---

- 랜덤 벡터들의 통계량과 그 다변량 표기법에 대한 글입니다.  

---  

- Random Vector란?  
  - Random Variable들의 벡터  
  
    ![](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BX%7D%20%3D%20%5BX_1%2CX_2%2C...%2CX_p%5D%5ET)  
    
    - ![](https://latex.codecogs.com/gif.latex?%5Cboldsymbol%7BX%7D%20%5C%3B%20is%20%5C%3Ba%20%5C%3Brandom%20%5C%3B%20vector)  
    - ![](https://latex.codecogs.com/gif.latex?X_j%2C%201%5Cleq%20j%20%5Cleq%20p%20%5C%3B%20is%20%5C%3B%20a%5C%3B%20random%20%5C%3B%20variable)  
    
    - 모평균벡터  

      ![](https://latex.codecogs.com/gif.latex?E%28%5Cboldsymbol%7BX%7D%29%20%3D%20%5Cbegin%7Bbmatrix%7D%20E%28X_1%29%5C%5C%20E%28X_2%29%5C%5C%20%5Cvdots%20%5C%5C%20E%28X_p%29%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cboldsymbol%7B%5Cmu%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cmu_1%5C%5C%20%5Cmu_2%5C%5C%20%5Cvdots%20%5C%5C%20%5Cmu_p%20%5Cend%7Bbmatrix%7D)  

    - 모분산행렬  

      ![](https://latex.codecogs.com/gif.latex?Var%28%5Cboldsymbol%7BX%7D%29%20%3D%20E%5B%28%5Cboldsymbol%7BX-%5Cmu%7D%29%28%5Cboldsymbol%7BX-%5Cmu%7D%29%5ET%5D)  

      ![](https://latex.codecogs.com/gif.latex?%3D%20%5Cbegin%7Bbmatrix%7D%20%5Csigma_%7B11%7D%20%26%20%5Csigma_%7B12%7D%20%26%20...%20%26%20%5Csigma_%7B1p%7D%5C%5C%20%5Csigma_%7B21%7D%20%26%20%5Csigma_%7B22%7D%20%26%20...%20%26%20%5Csigma_%7B2p%7D%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%26%5Cvdots%20%5C%5C%20%5Csigma_%7Bp1%7D%20%26%20%5Csigma_%7Bp2%7D%20%26%20...%5C%20%26%20%5Csigma_%7Bpp%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cboldsymbol%7B%5CSigma%20%7D)  
      
      - Var(X)는 대칭행렬  
      - 랜덤벡터 X의 어떤 두 확률변수가 독립이라면 그에 해당하는 Σ의 off-diagonal element는 0  
    
  - ![](https://latex.codecogs.com/gif.latex?%5Cboldsymbol%7BX%7D%20%5Cin%20R%5Ep%20%5C%3B%20and%20%5C%3B%20%5Cboldsymbol%7BY%7D%20%5Cin%20R%5Eq)인 두 개의 Random Vector가 있을 때,  
    
    - 공분산행렬  
      
      ![](https://latex.codecogs.com/gif.latex?Cov%28%5Cboldsymbol%7BX%2CY%7D%29%29%20%3D%20E%5B%5Cboldsymbol%7B%28X-%5Cmu_X%29%28Y-%5Cmu_Y%29%5ET%7D%5D%20%3D%20E%28%5Cboldsymbol%7BXY%5ET%7D%29%20-%20E%28%5Cboldsymbol%7BX%7D%29E%28%5Cboldsymbol%7BY%5ET%7D%29)  
      
      - size: pxq matrix  
      - Cov(__X__,__X__) = Var(__X__) = Cov(__X__)  
      
