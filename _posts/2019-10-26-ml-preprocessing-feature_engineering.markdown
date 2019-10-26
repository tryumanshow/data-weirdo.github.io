---  

layout: post
title:  "특성공학(Feature Engineering)의 기본적인 테크닉들"
subtitle:   "preprocessing-feature_engineering"
categories: ml
tags: preprocessing
comments: true

---  

- Medium 지의 '[Fundamental Techniques of Feature Engineering for Machine Learning](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)' 글을 보고 작성한 글입니다.  

---  

- Feature Engineering이 필요한 이유는 두 가지  
  1. 머신러닝 알고리즘이 필요로 하는 형태로 입력 데이터를 만들기 위해  
  2. 머신러닝 모델의 퍼포먼스 향상을 위해  
  
  ```  
  당신이 사용하는 피쳐가 다른 무엇들 보다도 결과에 큰 영향을 미친다. 
  어떤 단일한 알고리즘도 올바른 피쳐 엔지니어링을 통해 얻어지는 정보를 보충할 수는 없다.  
  - Luca Massaron -  
  ```

- 데이터 사이언티스트들은 80%의 시간을 Data Preparation에 사용  
  ![Source: https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/](https://miro.medium.com/max/960/0*-dn9U8gMVWjDahQV.jpg)  
  - 이 표는 곧 Feature Engineering의 중요성을 역설  
  
- Technique List  

  ```  
  1. Imputation  
  2. Handling Outliers  
  3. Binning  
  4. Log Transformation  
  5. One-Hot Encoding  
  6. Grouping Operations  
  7. Feature Split  
  8. Scaling  
  9. Extracting Date  
  ```  
 
## 1. Imputation  

  - 데이터 상의 빈 값들 채우기  
  - 빈 값은 ML 모델의 성능에 영향을 끼침  
  
  - 대부분의 ML 모델들은 빈 값이 있는 데이터를 받지 않고 에러를 뱉음  
  - 어떤 ML 모델들은 자동적으로 missing value가 있는 행을 지운다.  
    → 훈련 셋의 사이즈가 줄기 때문에 모델 성능이 떨어짐  
   
  - 처리 방법?  
    ### 방법 1. Remove values  
      - 빈 값이 있는 행을 지우거나, 전체 컬럼을 지워버린다.  
      - Threshold를 기준으로 행을 지우거나, 전체 컬럼을 지울 수 있음  
      
        ```  
        threshold = 0.7  
        
        #Dropping columns with missing value rate higher than threshold  
        data = data[data.columns[data.isnull().mean() < threshold]]  

        #Dropping rows with missing value rate higher than threshold  
        data = data.loc[data.isnull().mean(axis=1) < threshold]  
        ```  
        
    ### 방법 2. Numerical Imputation
      - 방법 1보다는 선호 되는 방법  
      - 그러나 무엇으로 빈값들을 채울 것인가에 대한 선택 중요  
        - 상황에 따라 잘 판단할 것  
        - 필자는 해당 컬럼의 median으로 채우는 게 적절한 방법이라 생각  
          - 이유: mean은 outlier에 대해 sensitive 함에 반해, median은 그렇지 않기 때문에  
          
        ```  
        #Filling all missing values with 0  
        data = data.fillna(0)  
          
        #Filling missing values with medians of the columns  
        data = data.fillna(data.median())  
        ```
      
    ### 방법 3. Categorical Imputation  
      - 범주형 변수에서 빈 값을 최빈값으로 채우는 것도 좋은 옵션  
      - 하지만 해당 범주의 분포가 uniform 하다는 생각이 들 경우 다른 값들로 채우는 것도 좋은 옵션  
        → Random Selection과 같은 형태로 수렴할 것  
        ```  
        #Max fill function for categorical columns  
        data['column_name'].fillna(data['column_name'].value_counts()  
        .idxmax(), inplace=True)  
        ```  

## 2. Handling Outliers  
  - Outlier를 탐지하는 가장 좋은 방법은 시각화일 것  
  - 모든 통계학적인 방법론들은 실수를 만들기 쉬움  
  
  - 탐지 방법?  
    ### 방법 1. Z-Score  
    
    ### 방법 2. Percentile 사용하기  
      - 상위 몇 %, 하위 몇 %가 outlier일 것이라고 가정할 수 있다.  
      - %에 대한 선택은 자신의 데이터가 어떠한 형태의 분포를 따르냐에 달림  
      
      ```  
      #Dropping the outlier rows with Percentiles
      upper_lim = data['column'].quantile(.95)
      lower_lim = data['column'].quantile(.05)

      data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]  
      ```  
      
  - outlier에 대한 고민  
    - 버릴 것인가? 놔둘 것인가? → 놔두자
   
    ```  
    # Capping the outlier rows with Percentiles  
    upper_lim = data['column'].quantile(.95)  
    lower_lim = data['column'].quantile(.05)  
    data.loc[(df[column] > upper_lim),column] = upper_lim  
    data.loc[(df[column] < lower_lim),column] = lower_lim  
    ```  
    
 ## 3. Binning  
   ![](https://miro.medium.com/max/981/0*XWta_U67Nv9udfY-.png)   
   - 수치형, 범주형 데이터 모두에 적용 가능  
   
   ```  
   # Numerical Binning Example  
   Value      Bin  
   0-30   ->  Low       
   31-70  ->  Mid       
   71-100 ->  High  
   
   # Categorical Binning Example  
   Value      Bin  
   Spain  ->  Europe      
   Italy  ->  Europe       
   Chile  ->  South America  
   Brazil ->  South America  
   ```
  - 오버피팅을 막기 위한 액션  
  
  - 수치형 데이터에서는 몇몇 심한 오버피팅의 경우를 제외하고는 굳이 binning을 안해줘도 될 것 같음  
  - 범주형 데이터에서는 간헐적으로 출현하는 값들이 통계학적인 모델들에 부정적 영향을 끼침  
    → 출현 빈도가 적은 범주형 데이터는 새로운 카테고리인 'Other'로 넣어주는 것과 같은 방법  
    
  ```  
  # Numerical Binning Example  
  data['bin'] = pd.cut(data['value'], bins=[0,30,70,100], labels=["Low", "Mid", "High"])  
     value   bin  
  0      2   Low  
  1     45   Mid  
  2      7   Low  
  3     85  High  
  4     28   Low  
  
  # Categorical Binning Example  
        Country  
  0      Spain  
  1      Chile  
  2  Australia  
  3      Italy  
  4     Brazil  
  conditions = [  
      data['Country'].str.contains('Spain'),  
      data['Country'].str.contains('Italy'),  
      data['Country'].str.contains('Chile'),  
      data['Country'].str.contains('Brazil')]  

  choices = ['Europe', 'Europe', 'South America', 'South America']  

  data['Continent'] = np.select(conditions, choices, default='Other')  
      Country      Continent  
  0      Spain         Europe  
  1      Chile  South America  
  2  Australia          Other  
  3      Italy         Europe  
  4     Brazil  South America  
  ```  
## 4. Log Transform  
   - 피쳐 엔지니어링에서 가장 많이 사용되는 Transformation 방법 중 하나  
   
   - 장점?  
   
    ```  
    - skewd data를 정규분포에 근사  
    - 데이터들 간의 강도 차이를 보정  
    - 아웃라이어의 효과 감소  
    ```  
    
   - log transformation을 하려면 모든 데이터값이 양수여야  
   - log(x+1)을 해주는 방법도 존재  
   
   ```  
   # Log Transform Example  
   data = pd.DataFrame({'value':[2,45, -23, 85, 28, 2, 35, -12]})  
   data['log+1'] = (data['value']+1).transform(np.log)  
   
   #Negative Values Handling  
   #Note that the values are different  
   data['log'] = (data['value']-data['value'].min()+1) .transform(np.log)  
   ```  
   
## 5. One-hot encoding  
  ![](https://miro.medium.com/max/1064/1*ZX99GOZ6-9_yJg6rZchTEA.png)  
  - 범주형 데이터를 정보 손실 없이 숫자형으로 바꾸어줌  
  - get_dummies 사용이 더 간단  
  
  ```  
  encoded_columns = pd.get_dummies(data['column'])  
  data = data.join(encoded_columns).drop('column', axis=1)  
  ```  
  
## 6. Grouping Operations  
  - 'Tidy' 데이터를 준비하는 것이 굉장히 중요  
    - Tidy라 함은 모든 열은 feature 하나하나를 나타내고, 각자의 항은 관찰값 마다의 정보를 모두 포함하도록 정리된 데이터  
  
  - 수치형 Feature  
    - avg, sum 등을 사용  
    - ex. total count나 ratio 컬럼을 얻고 싶을 때  
    
      ```  
      #sum_cols: List of columns to sum  
      #mean_cols: List of columns to average  
      grouped = data.groupby('column_to_group')  

      sums = grouped[sum_cols].sum().add_suffix('_sum')  
      avgs = grouped[mean_cols].mean().add_suffix('_avg')  

      new_df = pd.concat([sums, avgs], axis=1)  
      ```  
    
  - 범주형 Feature  
    - 복잡  
    - Way1  
      - 가장 빈도가 높은 라벨값 선택  
      
        ```  
        data.groupby('id').agg(lambda x: x.value_counts().index[0])  
        ```  
    - Way2  
      - 피봇 테이블 만들기  
        ![](https://miro.medium.com/max/1070/1*VWBbZRkTrHJQrQfWlPQWUg.png)  
        
        ```  
        # Pivot table Pandas Example
        data.pivot_table(index='column_to_group', columns='column_to_encode', values='aggregation_column', aggfunc=np.sum, fill_value = 0)
        ```  
     - etc ...   
      
## 7.  Feature Split  
  - 컬럼들로부터 이용가능한 부분들만 뽑아내기  
  
    ```  
    # Example  
    data.name  
    0  Luther N. Gonzalez  
    1    Charles M. Young  
    2        Terry Lawson  
    3       Kristen White  
    4      Thomas Logsdon  
    
    # Extracting first names  
    data.name.str.split(" ").map(lambda x: x[0])  
    0     Luther  
    1    Charles  
    2      Terry  
    3    Kristen  
    4     Thomas  
    ```  
    
## 8. Scaling  
  - Scaling 과정을 거친 후 연속형 변수들의 range가 맞추어짐  
  - Scaling을 하지 않아도 되는 알고리즘들도 존재  
  - 거리 계산에 기반을 둔 ML 알고리즘들은 scaling이 필수 (ex. KNN, K-Means)  
  
## 9. Extracting Date  

  - 날짜 형식 그대로는 ML 알고리즘이 인식을 못함  

  - 방법1. 날짜로 부터 정보를 독립적으로 추출하여 각각의 컬럼에 저장한다.  
    - ex. Year, month, day,  etc  
  - 방법2. 컬럼의 날짜와 오늘의 날짜의 차이를 추출한다. (연도별, 월별, 일수 별)
  - 방법3. 해당 날짜의 특별한 피쳐를 추출한다.  
    - ex. weekday or not, weekend or not, holiday or not, etc  
    
  ```  
  from datetime import date

  data = pd.DataFrame({'date':  
  ['01-01-2017',  
  '04-12-2008',  
  '23-06-1988',  
  '25-08-1999',  
  '20-02-1993',  
  ]})  

  # Transform string to date  
  data['date'] = pd.to_datetime(data.date, format="%d-%m-%Y")  

  # Extracting Year  
  data['year'] = data['date'].dt.year  

  # Extracting Month  
  data['month'] = data['date'].dt.month  

  # Extracting passed years since the date  
  data['passed_years'] = date.today().year - data['date'].dt.year  

  # Extracting passed months since the date  
  data['passed_months'] = (date.today().year - data['date'].dt.year) * 12 + date.today().month - data['date'].dt.month  

  # Extracting the weekday name of the date  
  data['day_name'] = data['date'].dt.day_name()  
  ```  
  
` Garbage in, garbage out! `    

## References  
[Fundamental Techniques of Feature Engineering for Machine Learning](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)
