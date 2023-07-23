import pandas as pd

def load_dataset(): 
    """
    - 데이터 불러오기
    - 데이터 전처리
    """
    df = pd.read_csv('./Data/Regression_data.csv')

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