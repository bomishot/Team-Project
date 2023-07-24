import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler

"""
회귀, 이진 분류, 다중 분류 데이터 불러오기 및 전처리
"""

def load_dataset_regression(): 
    """
    회귀 모델 
    """
    df = pd.read_csv('./data/Regression_data.csv')

    # sex열 범주형 -> 수치형 변환
    sex_mapping = {'M':0, 'F':1, 'I':2}
    df['Sex'] = df['Sex'].map(sex_mapping)

    # 이상치 제거 (height 열)
    df.drop(df.Height[df.Height >0.3].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # "전체 무게 >= 조개껍질 벗긴 무게 + 내장 무게 + 껍질 무게"를 만족하지 않는 행들 제거
    df = df[df['Whole weight'] >= df['Shucked weight'] + df['Viscera weight'] + df['Shell weight']]
    df = df.reset_index(drop=True)
    
    return df





# 이진분류
def load_dataset_binary_classification():
    """
    이진 분류 : 데이터 불균형 문제 (1 희소)
    """
    df = pd.read_csv('./data/binary_classification_data.csv')

    # Downsampling : Target class
    # 0: 16259 -> 1870
    # 1: 1639
    for i in range(16000):
        if df['target_class'][i] == 0:  
            df = df.drop(i)
    return df



# 다중분류 - 나중에 성능 향상시킬 때, 다시 볼 수도 있으니 두 분 것 다 넣겠습니다!
# 우영님 woo로 표시했고, 강우님 gang으로 함수 뒤에 표시했습니다! (나중에 차차 수정)

def load_dataset_multi_classification_woo(): 
    """
    다중 분류 - 우영님
    """
    df = pd.read_csv('./data/mulit_classification_data.csv')

    # 컬럼 전처리
    df['type'] = 'TypeOfSteel_A300'
    df.loc[df['TypeOfSteel_A400'] == 1, 'type'] = 'TypeOfSteel_A400'
    df.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400'], axis=1, inplace=True)
    df['type'].replace({"TypeOfSteel_A300":0,"TypeOfSteel_A400":1},inplace=True)

    # X_Perimeter + Y_Perimeter = Total_Perimeter 
    df['Total_Perimeter'] = df['X_Perimeter'] + df['Y_Perimeter']
    df.drop(['X_Perimeter', 'Y_Perimeter'], axis=1, inplace=True)

    # Mean_of_Luminosity 컬럼으로 합치기
    df['Mean_of_Luminosity'] = (df['Minimum_of_Luminosity'] + df['Maximum_of_Luminosity']) / 2
    df.drop(['Minimum_of_Luminosity', 'Maximum_of_Luminosity'], axis=1, inplace=True)

    # target 데이터 -> int bool 타입으로 변경 
    target_df = [
        df['Pastry'],
        df['Z_Scratch'],
        df['K_Scatch'],
        df['Stains'],
        df['Dirtiness'],
        df['Bumps'],
        df['Other_Faults'] 
    ]
    targets = list(map(lambda i: i.astype(bool), target_df))
    choices = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    df.drop(df[choices].columns, axis=1, inplace=True)
    df['class'] = np.select(targets, choices)

    # class외의 독립변수 데이터 표준화
    df_1 = df.iloc[:, :-1]
    # StandardScaler 객체 생성
    standard_scaler = StandardScaler()
    np_scaled = standard_scaler.fit_transform(df_1)
    df_norm = pd.DataFrame(np_scaled, columns=list(df_1.columns))

    # 이상치 제거 
    low, high = .05, .95
    quantiles = df_norm.quantile([low, high])
    quantile_norm = df_norm.apply(lambda col: col[(col >= quantiles.loc[low, col.name]) & 
                                        (col <= quantiles.loc[high, col.name])], axis=0)

    # 상관계수 행렬 생성
    corr_matrix = df_norm.corr().abs()
    # 상삼각 행렬 부분(대각선 기준으로 위쪽)만 남기기 위해 적용
    under = corr_matrix * (np.triu(np.ones(corr_matrix.shape), k=1))
    # 상관계수가 0.95보다 큰 변수들 찾아서 제거
    to_drop = [column for column in under.columns if any(under[column] > 0.95)]
    df_norm = df_norm.drop(df_norm[to_drop], axis=1)

    # target 데이터 LabelEncoder
    X = df_norm
    le = LabelEncoder()

    # df_norm DataFrame에서 'class' 컬럼을 범주형 타겟 데이터로 사용
    targets = df['class']
    Y = le.fit_transform(targets)

    # X와 Y를 하나의 데이터프레임으로 합치기 위해 Y를 Series로 변환하고, 열 이름을 'target'으로 지정
    Y = pd.Series(Y, name='targets')

    # 클래스 비중 조절을 위한 RandomOverSampler 객체 생성
    ros = RandomOverSampler(random_state=0)

    # 클래스 비중 조절을 위해 fit_resample() 메서드를 사용하여 X_train, y_train을 샘플링
    X_resampled, y_resampled = ros.fit_resample(X, Y)

    # X_resampled와 y_resampled를 DataFrame으로 변환
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)

    # 'targets' 컬럼 추가
    df_resampled['targets'] = y_resampled
    df = df_resampled

    return df


def load_dataset_multi_classification_gang():
    """
    다중 분류 - 강우님
    """
    df = pd.read_csv('./data/mulit_classification_data.csv')

    # 데이터 엔지니어링
    df_column = ['X_Minimum',                   # X_Minimum, X_Maximum -> X_Point
                'X_Maximum',                    # //
                'Y_Minimum',                    # Y_Minimum, Y_Maximum -> Y_Point
                'Y_Maximum',                    # //
                'Pixels_Areas',                 # pass
                'X_Perimeter',                  # pass
                'Y_Perimeter',                  # pass
                'Sum_of_Luminosity',            # pass
                'Minimum_of_Luminosity',        # pass
                'Maximum_of_Luminosity',        # pass
                'Length_of_Conveyer',           # 어느 정도 분류가 되지만 현재 정규화로 진행할 예정
                'TypeOfSteel_A300',             
                'TypeOfSteel_A400',
                'Steel_Plate_Thickness',
                'Edges_Index',
                'Empty_Index',
                'Square_Index',
                'Outside_X_Index',
                'Edges_X_Index',
                'Edges_Y_Index',
                'Outside_Global_Index',
                'LogOfAreas',
                'Log_X_Index',
                'Log_Y_Index',
                'Orientation_Index',
                'Luminosity_Index',
                'SigmoidOfAreas',
                'Pastry',
                'Z_Scratch',
                'K_Scatch',
                'Stains',
                'Dirtiness',
                'Bumps',
                'Other_Faults'
                ]
    
    
    # X_Minimum, X_Maximum, Y_Minimum, Y_Maximum 컬럼 X_Point, Y_Point로 반환
    df_sample = df.copy()
    
    df_sample['X_Point'] = (df_sample['X_Maximum'] + df_sample['X_Minimum']) / 2
    df_sample['Y_Point'] = (df_sample['Y_Maximum'] + df_sample['Y_Minimum']) / 2
    # df_sample = df_sample.reindex(columns=['X_Point', 'Y_Point'] + df_sample.columns[:-2].tolist())
    
    # apply 함수와 lambda 함수를 사용하여 조건에 따라 값 설정
    df_sample['TypeOfSteel'] = df_sample.apply(lambda row: 1 if row['TypeOfSteel_A300'] == 1
                                                else (0 if row['TypeOfSteel_A400'] == 1 else 2), axis=1)
    df_sample.drop(columns=['TypeOfSteel_A300', 'TypeOfSteel_A400'], inplace=True)
    
    # 10으로 나눈 결과를 "Thickness_group" 컬럼에 저장
    df_sample['Steel_Plate_Thickness_10units'] = df_sample['Steel_Plate_Thickness'] // 10
    df_sample.drop(columns="Steel_Plate_Thickness", inplace=True)


    # "Outside_Global_Index" 값을 범주화하여 "Outside_Global_Category" 컬럼에 저장 (한 줄로 작성)
    df['Outside_Global_Category'] = df['Outside_Global_Index'].map({1.00: 2, 0.50: 1, 0.00: 0})

    edit_column = ["X_Point", "Y_Point",            # 'X_Minimum', 'X_Maximum','Y_Minimum','Y_Maximum',
                'Pixels_Areas',
                'X_Perimeter',
                'Y_Perimeter',
                'Sum_of_Luminosity',
                'Minimum_of_Luminosity',
                'Maximum_of_Luminosity',
                'Length_of_Conveyer',
                "TypeOfSteel",                      # 'TypeOfSteel_A300', 'TypeOfSteel_A400',
                "Steel_Plate_Thickness_10units",    # 'Steel_Plate_Thickness',
                'Edges_Index',
                'Empty_Index',
                'Square_Index',
                'Outside_X_Index',
                'Edges_X_Index',
                'Edges_Y_Index',
                "Outside_Global_Category",          # 'Outside_Global_Index',
                'LogOfAreas',
                'Log_X_Index',
                'Log_Y_Index',
                'Orientation_Index',
                'Luminosity_Index',
                'SigmoidOfAreas',
                'Pastry',
                'Z_Scratch',
                'K_Scatch',
                'Stains',
                'Dirtiness',
                'Bumps',
                'Other_Faults'
                ]

    df_sample = df_sample.reindex(columns=edit_column)
    
    return df
