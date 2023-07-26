# AI Software Upgrade Project



 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Figma](https://img.shields.io/badge/figma-%23F24E1E.svg?style=for-the-badge&logo=figma&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)  
![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white) ![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white) ![Zoom](https://img.shields.io/badge/Zoom-2D8CFF?style=for-the-badge&logo=zoom&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)



| project name | AI Software Upgrade Project | 
| ------------ | -------------- |
| project period | 2023.007.19 - 2023.07.31 |
| member | 김우영, 김보미, 이강우, 최재용 | 
| project keywords | 딥러닝, numpy, 레거시 코드 리펙터링, UIUX 개선, AI 성능 향상 |


| name | Role | 
| ---- | ---- |
| 김우영|- |
|김보미|-|
|이강우|-|
|최재용|-|

<br>
<br>

## Project Process
1. 데이터 가져오기
  * Regression
  * Binary Classification
  * Multi Classification
2. 데이터 전처리
3. 모델링
4. 웹 배포
  
<br>
<br>
  
# Abstract
3개의 데이터에 대해 최적의 성능을 내기 위해 데이터 설명, 전처리 과정에 따른 효과, 모델링, 최적의 성능을 가진 모델을 가지고 웹에 배포하여 새로운 데이터에 대해 예측을 수행하고, 데이터 분석을 해보겠습니다.
<br>
<br>

# Data 1 - Regression
### Data

- 조개의 물리적 특성과 연령에 대한 정보가 있다. **조개류는 성장에 따라 ring 수가 증가한다.**
- 데이터 개수 : 4177개
- 독립변수 8개, 종속변수 1개
- Sex : M/F/I(Instant: 성체 미성숙)
- Length : 길이
- Diameter : 지름
- Height : 높이
- Whole weight : 전체 무게
- Shucked wieght : 껍질 제거 후 무게
- Viscera weight : 내장 무게
- Shell weight : 껍질 무게
- **Rings (Target Feature) : 조개류의 나이**를 나타낸다.

> **조개 껍데기에 새겨진 성장 줄무늬로 조개 나이를 추정한다.** (나무의 나이테와 비슷한 역할)
> 
> - (**비례 관계**) 높이, 껍질 무게, 전체 무게, 지름, 길이가 클수록, 조개 나이가 많아진다.
> - (**반비례 관계**) 내장 무게, 껍질 벗긴 무게가 작을수록, 조개 나이가 많아진다.
> 
> ⇒ 쉽게 말해서, **나이가 든 조개일수록 몸통은 커지나, 내장(이런 안쪽) 부분이 작아진다.**
> 

### Data Analysis

- PCA 효과 성능 크지 않음 ( xgboost 모델에 했다.)
- 상관관계 큰 것들 많음. ⇒ 하나씩 열 빼며 성능 비교해봤으나, 뻬면 성능 더 안 좋아졌음.
- 모든 독립변수 열에서, ‘Sex’열의 2:instant(성체 미성숙)이 가장 작은 수치를 보였으며, 0:m(남자)일수록 가장 높은 수치를 보여줌.
    

### EDA 및 각 효과

- sex열 : 범주형 → **onehotencoding** 적용
- **이상치 제거**
    - z-score 이용
        - 효과 : loss 줄어듬.
        - z-score의 절댓값이 3보다 크면 이상치로 판단하고 제거함.
    - “전체 무게 >= 조개껍질 벗긴 무게 + 내장 무게 + 껍질 무게"를 만족하지 않는 행들 제거
        - 효과 : loss 줄어듬.

> EDA 마친 데이터 개수 : 4177개 → 3876개 (301개 제거)
> 

### 모델 성능 측정

ML 모델인 XGBoost가 성능이 가장 좋았다. 그 다음으로는, DL 모델인, RNN이 성능이 가장 좋다. 

| Model | MSE | Accuracy | r2_score |
| --- | --- | --- | --- |
| (BEST) XGBoost | 3.92 | 0.86 | 0.54 |
| RNN | 4.34 | 0.85 | 0.56 |
| Linear Regression | 4.83 | 0.83 | 0.51 |
| Ridge | 4.74 | 0.83 | 0.51 |
| Lasso | 4.81 | 0.83 | 0.51 |
| LightGBM | 4.74 | 0.84 | 0.52 |

모든 모델 모두, gridsearch cv 적용해 hyperparmeter tuning한 수치의 성능임.

<br>
<br>


# Data 2 - Binary Classification




<br>
<br>

# Data 3  Multi Classification
### Data

> 7가지 유형의 결함 중 **해당 강판의 결함 유형 분류**
> 
> 
> **⇒ 스테인레스 강판 생산 과정에서 품질 검사 및 결함 분류에 활용**
> 
- 데이터 개수 : 1941개
- 총 32 columns : 독립변수 25개, 종속변수 7개
    - 'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum' : 결함 영역 내 x, y축의 최대, 최소값
    - 'Pixels_Areas' : 결함 영역 내 픽셀 개수
    - 'X_Perimeter', 'Y_Perimeter' : 결함 영역의 x,y축 둘레
    - 'Sum_of_Luminosity','Minimum_of_Luminosity', 'Maximum_of_Luminosity' : 밝기값 합, 최소, 최대
    - 'Length_of_Conveyer' : 강판 이동시키는 운반 장치 길이
    - 'TypeOfSteel_A300', 'TypeOfSteel_A400' : 스테인레스 강판의 종류가 A300인지 A400인지 여부 (0 or 1)
    - 'Steel_Plate_Thickness' : 스테인레스 강판의 두께
    - 'Edges_Index', 'Edges_X_Index', 'Edges_Y_Index' : 강판 결험 영역의 가장자리 나타내는 지수, x,y축에 대한 강판 결함 영역의 가장자리를 나타내는 지수
    - ‘Empty_Index' : 결함 영역 내의 비어있는 영역 지수
    - ‘Square_Index' : 결함 영역 넓이 지수
    - 'Outside_X_Index', 'Outside_Global_Index'
    - 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index' : 결함영역 로그값, x,y축방향으로의 결함 영역 로그값
    - 'Orientation_Index' : 결함 영역 방향 지수
    - 'Luminosity_Index' : 결함 영역 밝기 지수
    - ‘'SigmoidOfAreas' : 결함 영역 넓이에 대한 시그모이드 값

> **Target Columns (7): Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults**
> 
> 
> ⇒ **강판 표면에 생긴 결함의 종류**
> 

1) **Pastry** : **빵 모양의 파손이 발생**한 결함

2) **Z_Scratch** : **Z자 모양의 스크래치**

3) **K_Scatch : K자 모양의 스크래치**

4) **Stains : 얼룩이나, 얼룩 모양의 더러움이 생긴 결함**

5) **Dirtiness : 더러운 자국이 있는 결함 (**먼지나 오염물)

6) **Bumps** : **덤불 모양으로 돌출된 결함**

7) **Other_Faults :** 위에 언급된 6가지 유형 **외에 다른 종류의 결함**

> 빵 모양, z자, k자, 얼룩, 더러운 자국, 덤불 모양, 그외것 의 결함들이 강판 표면에 있다.
> 

### Data Analysis

- 

### EDA 및 각 효과

- 합칠 수 있는 컬럼 합치기
    - target columns 7개 ⇒ class라는 1개의 새로운 열에 유형 분류 **(분류하고 다시 원핫인코딩으로 하면 처음이랑 똑같아 지지 않나??)**
    - x_perimeter + y_perimeter = Total_perimeter
    - min, max of Luminosity : mean 값으로 묶기 → 다중공산성 문제 해결
- 독립변수 데이터 표준화
- 이상치 제거
- **target 데이터 불균형 문제 ⇒  `smote` 적용**

![a59c61d0-abb3-4f4f-8de5-ec5688b9c98f.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3fdf204a-9bfd-4ec0-95d6-a4ba0fcb5a58/a59c61d0-abb3-4f4f-8de5-ec5688b9c98f.png)

⇒ 모든 class를 673개로 고정

- oversampling, undersampling, smote 적용시 smote가 성능 가장 good (recall, precision)
    - 무작위로 oversampling을 수행하는 방법보다 과적합 가능성이 적기 때문
    - undersampling에 비해 정보가 소실되지 않고, 데이터 수가 줄어들지 않기 때문
- 수치형 ⇒ category 타입 변환 : TypeOfSteel_A300, TypeOfSteel_A400, Outside_Global_Index
- 독립변수들의 skewed feature 확인
    - 왜도값이 클수록, 분포가 비대칭적이며, 0에 가까울수록 대칭적으로 분포됨.
    - 왜도값 0.75보다 큰 것은, 정규분포 형태로 변환

![d943571f-7e97-4ed6-9774-3d0f6df4d29c.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cf05291e-8bc0-42df-90cd-9ca824ee4d19/d943571f-7e97-4ed6-9774-3d0f6df4d29c.png)

![f371023b-a3c5-4d70-b768-bd5d4205ee15.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f887410-af43-40b8-97b3-5f8092f83e30/f371023b-a3c5-4d70-b768-bd5d4205ee15.png)

- 비대칭 데이터 여전히 존재, 표준화 작업 이후에도 이상치 여전히 존재 → 극단적 이상치 직접 삭제
- 다중공산성 문제
    - VIF: 다중공산성 평가 지표 : 각 독립변수가 다른 독립변수들과 얼마나 상관관계가 있는지를 나타냄. 값이 1에 가까울 수록, 상관관계가 적어 다중공산성 문제가 적다. 1보다 크면, 클수록 다중공산성 영향이 크다.
    - 다중공산성 : 각 독립변수가 다른 독립변수들과 얼마나 상관관계가 있는지를 나타냄.
    - 해결 방법 : VIF 값이 큰 독립변수들 제거하거나, 변수들을 변형시켜 상관관계를 줄이는 전처리 작업
- feature importance
    
    ![fd9db41d-ea4a-499f-a4b6-2a74c189b660.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4d4d1faa-51ac-4bdd-936d-10e17953687b/fd9db41d-ea4a-499f-a4b6-2a74c189b660.png)
    
- cardinality
    
    ![760a0bec-f123-42bc-b073-06b5ed1d4501.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/37f63014-010a-4d78-b39f-7026c78eaac8/760a0bec-f123-42bc-b073-06b5ed1d4501.png)
    
- 상관계수 0.95 이상인 열 제거
- pca, feature selection시, 성능 오히려 떨어짐.
- **컬럼 축소 더!!!!! (합칠 수 있는 컬럼 있으면 더 합치기)**

> EDA 마친 데이터 개수 : 4020개
> 

질문

- target columns 7개 ⇒ class라는 1개의 새로운 열에 유형 분류 **(분류하고 다시 원핫인코딩으로 하면 처음이랑 똑같아 지지 않나??)**
- smote가 가장 성능 좋다고 했는데, 왜 randomoversampler로 했는지??

### Modeling
Model		
(BEST) Random Forest		
Logistic Regression		
SVM		
Neural model		



<br>
<br>
    


# Web 배포




