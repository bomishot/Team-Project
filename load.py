import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from scipy import stats

"""
회귀, 이진 분류, 다중 분류 데이터 불러오기 및 전처리
"""

def load_dataset_regression():
    """
    회귀 모델 
    """
    data = pd.read_csv('./data/Regression_data.csv')
    # 'Sex' 열을 원-핫 인코딩으로 변환
    data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

    # 이상치 제거를 위해 확인할 열 선택
    columns_to_check = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

    # z-점수를 이용하여 이상치 제거
    z_scores = stats.zscore(data[columns_to_check])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]
    df =data

    # "전체 무게 >= 조개껍질 벗긴 무게 + 내장 무게 + 껍질 무게"를 만족하지 않는 행들 제거
    df = df[df['Whole weight'] >= df['Shucked weight'] + df['Viscera weight'] + df['Shell weight']]
    return df.reset_index(drop=True)


def load_dataset_binary_classification():
    """
    이진 분류: 데이터 불균형 문제 (1 희소)
    """
    df = pd.read_csv('./data/binary_classification_data.csv')

    df_1 = df.iloc[:, :-1]
    standard_scaler = StandardScaler()
    np_scaled = standard_scaler.fit_transform(df_1)
    df_norm = pd.DataFrame(np_scaled, columns=list(df_1.columns))

    # 이상치 제거
    low, high = .05, .95
    quantiles = df_norm.quantile([low, high])
    quantile_norm = df_norm.apply(lambda col: col[(col >= quantiles.loc[low, col.name]) &
                                                 (col <= quantiles.loc[high, col.name])], axis=0)
    X = df_norm
    targets = df['target_class']
    le = LabelEncoder()
    Y = le.fit_transform(targets)
    Y = pd.Series(Y, name='target_class')

    df = pd.DataFrame(X, columns=X.columns)
    df['targets'] = Y
    return df




def load_dataset_multi_classification(): 
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


