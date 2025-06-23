# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# ETS 모델과 관련 도구를 불러옵니다.
from statsmodels.tsa.api import ETSModel
from datetime import datetime, timedelta
from typing import List
import os

def process_and_prepare_master_data(input_df: pd.DataFrame, today_date: datetime.date) -> dict:
    """
    1단계: 데이터 준비 (Data Preparation)
    전달받은 데이터프레임을 정제하고, 시설 위치별 '일별 대표 데이터셋'(마스터 데이터)을 생성합니다.
    이 함수는 Prophet 버전과 동일하게 작동합니다.

    Args:
        input_df (pd.DataFrame): '일시', '시설 ID', '수온(℃)' 컬럼을 포함한 원본 데이터프레임.
        today_date (datetime.date): 예측 기준일("오늘"). 이 날짜 이전의 데이터만 사용합니다.

    Returns:
        dict: {'시설 코드': master_dataframe} 형태의 딕셔너리.
    """
    print("===== 1단계: 데이터 준비 시작 =====")
    
    df = input_df.copy()
    
    # 1. 학습 데이터 범위 정의 (오늘 데이터 제외)
    df = df[df['일시'] < datetime.combine(today_date, datetime.min.time())]
    print(f"-> {today_date} 이전 데이터만 사용. (총 {len(df)}개)")

    # 2. 2단계 이상치 정제 파이프라인
    df.sort_values(by=['시설 ID', '일시'], inplace=True)

    # 1차 (동적 필터링)
    df['temp_diff'] = df.groupby('시설 ID')['수온(℃)'].diff().abs()
    change_threshold = 5.0
    change_outliers = df['temp_diff'] > change_threshold
    print(f"-> 동적 필터링: 급격한 변화(>{change_threshold}℃) 데이터 {change_outliers.sum()}개 제거.")
    df.loc[change_outliers, '수온(℃)'] = np.nan
    df.drop(columns=['temp_diff'], inplace=True)
    
    # 2차 (정적 필터링)
    lower_bound, upper_bound = 5.0, 30.0
    abs_outliers = (df['수온(℃)'] < lower_bound) | (df['수온(℃)'] > upper_bound)
    print(f"-> 정적 필터링: 유효범위({lower_bound}℃ ~ {upper_bound}℃) 밖의 데이터 {abs_outliers.sum()}개 제거.")
    df.loc[abs_outliers, '수온(℃)'] = np.nan
    
    df.dropna(subset=['수온(℃)'], inplace=True)
    df['시설 코드'] = df['시설 ID'].str[:5]
    
    # 3. 시설별 대표 데이터 생성 (일별)
    master_datasets = {}
    for code, group_df in df.groupby('시설 코드'):
        daily_master = group_df.set_index('일시')[['수온(℃)']].resample('D').median()
        daily_master.interpolate(method='time', inplace=True)
        # ETS 모델은 Prophet과 달리 'ds', 'y' 컬럼명이 필수가 아닙니다.
        # 날짜 인덱스와 값 컬럼만 있으면 됩니다.
        master_datasets[code] = daily_master
        print(f"-> '{code}' 시설의 마스터 데이터 생성 완료 ({len(daily_master)}일치).")
        
    print("===== 1단계: 데이터 준비 완료 =====")
    return master_datasets

def train_and_predict_ets(df_master_daily: pd.DataFrame, freq_unit: str = 'D', periods: int = 7) -> pd.DataFrame:
    """
    2단계: ETS 모델 예측 실행 (Prediction Execution with ETS)
    지정된 시간 단위로 리샘플링 및 필터링하고, ETS 모델을 학습시켜 미래를 예측합니다.

    Args:
        df_master_daily (pd.DataFrame): 1단계에서 생성된 일별 마스터 데이터프레임.
        freq_unit (str): 예측 주기 단위 ('D', 'W', 'M').
        periods (int): 예측할 기간의 수.

    Returns:
        pd.DataFrame: '예측시점', '예측온도', '불확실성_최저', '불확실성_최고' 컬럼을 포함한 예측 결과.
    """
    df_to_process = df_master_daily.copy()
    
    # 1. 예측 단위별 데이터 리샘플링 및 필터링
    resample_freq = 'D'
    seasonal_periods = 365 # 기본값: 일별 데이터, 1년 계절성
    
    if freq_unit == 'W':
        resample_freq = 'W-MON'
        seasonal_periods = 52 # 주별 데이터, 1년 계절성
    elif freq_unit == 'M':
        resample_freq = 'MS'
        seasonal_periods = 12 # 월별 데이터, 1년 계절성

    df_resampled = df_to_process.resample(resample_freq).median()
    df_resampled.drop(df_resampled.tail(1).index, inplace=True) # 불완전한 마지막 기간 데이터 제외
    df_resampled.dropna(inplace=True)

    if len(df_resampled) < seasonal_periods * 2: # 안정적인 계절성 학습을 위해 최소 2주기 데이터 확보
        print(f"!!! 경고: 학습 데이터가 {len(df_resampled)}개로 너무 적어 예측을 건너뜁니다.")
        return pd.DataFrame()

    # 2. ETS 모델 학습
    # endog: 예측하려는 시계열 데이터
    # seasonal_periods: 계절성 주기 (예: 1년 주기의 일별 데이터는 365)
    # trend='add', seasonal='add': 추세와 계절성 모두 덧셈(additive) 모델 사용
    model = ETSModel(
        df_resampled['수온(℃)'],
        seasonal_periods=seasonal_periods,
        trend='add',
        seasonal='add'
    ).fit()

    # 3. 미래 예측 (7개 기간)
    # get_prediction()을 사용하여 예측값과 신뢰구간을 함께 얻습니다.
    prediction = model.get_prediction(steps=periods)
    
    # 4. 결과 추출 및 반환
    # alpha=0.2 는 80% 신뢰구간을 의미 (Prophet 기본값과 동일)
    df_forecast = prediction.summary_frame(alpha=0.2)
    
    result = df_forecast[['mean', 'pi_lower', 'pi_upper']].copy()
    result.reset_index(inplace=True)
    result.rename(columns={
        'index': '예측시점',
        'mean': '예측온도',
        'pi_lower': '불확실성_최저',
        'pi_upper': '불확실성_최고'
    }, inplace=True)
    
    return result

def run_aquaculture_forecast_pipeline(raw_df: pd.DataFrame, units_to_run: List[str] = ['D', 'W', 'M']) -> pd.DataFrame:
    """
    데이터 정제부터 ETS 모델 기반 주기별 예측까지 전체 파이프라인을 실행하고,
    통합된 예측 결과를 반환합니다.

    Args:
        raw_df (pd.DataFrame): '일시', '시설 ID', '수온(℃)' 컬럼을 포함한 원본 데이터프레임.
        units_to_run (List[str]): 예측을 실행할 단위 리스트. (예: ['D', 'W']).

    Returns:
        pd.DataFrame: 모든 시설과 예측단위에 대한 통합 예측 결과.
    """
    today = datetime.now().date()
    all_forecast_results = []
    UNIT_MAP = {'D': '일', 'W': '주', 'M': '월'}
    
    try:
        master_datasets = process_and_prepare_master_data(raw_df, today)

        for facility_code, df_master in master_datasets.items():
            print(f"\n===== '{facility_code}' 시설 예측 시작 (ETS 모델) =====")
            
            for freq in units_to_run:
                name = UNIT_MAP.get(freq)
                if not name:
                    print(f"!!! 경고: 알 수 없는 예측 단위 '{freq}'는 건너뜁니다.")
                    continue
                
                print(f"--- {name} 단위 예측 (향후 7{name}) ---")
                try:
                    # ETS 예측 함수 호출
                    forecast_result = train_and_predict_ets(df_master, freq_unit=freq, periods=7)
                    if not forecast_result.empty:
                        forecast_result['시설코드'] = facility_code
                        forecast_result['예측단위'] = name
                        all_forecast_results.append(forecast_result)
                    
                except Exception as e:
                    print(f"!!! '{name}' 단위 예측 중 오류 발생: {e}")

        if all_forecast_results:
            return pd.concat(all_forecast_results, ignore_index=True)

    except Exception as e:
        print(f"전체 파이프라인 실행 중 오류가 발생했습니다: {e}")
        
    return pd.DataFrame()

def get_dataframe():
    """
    이전 테스트 코드에서 사용하였던 데이터 loading 및 전처리 함수입니다.
    이 함수는 현재 서버에 저장된 데이터에 맞춰 수정이 가능합니다.
    """
    folder_path = './farm_data'  # CSV 파일이 있는 폴더 경로
    print(f"'{folder_path}' 폴더에서 CSV 파일을 읽습니다...")

    # 폴더가 존재하는지 확인
    if not os.path.isdir(folder_path):
        print(f"오류: '{folder_path}' 폴더를 찾을 수 없습니다. 폴더를 생성하고 CSV 파일을 넣어주세요.")
        exit()

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print(f"오류: '{folder_path}' 폴더에 CSV 파일이 없습니다.")
        exit()

    dataframes = []
    for f in csv_files:
        file_path = os.path.join(folder_path, f)
        try:
            # CSV 파일을 읽어올 때, 8번째와 16번째 열을 float로 지정
            df_temp = pd.read_csv(file_path, dtype={8: float, 16: float})
            dataframes.append(df_temp)
        except Exception as e:
            print(f"오류 발생 {file_path}: {e}")
            continue

    # 모든 데이터프레임을 하나로 합칩니다.
    raw_df = pd.concat(dataframes, ignore_index=True)
    print("-> 모든 CSV 파일 통합 완료")
    print("-> 상세 전처리를 시작합니다...")

    # df에서 지역명, 양식장 이름, 시설 이름, 시설 ID를 추출하여 dictionary 생성
    facility_id_dict = {}
    for index, row in raw_df.iterrows():
        # .get(key, default_value)를 사용하여 키가 없을 때 오류 방지
        region = row.get('지역명', '')
        farm_name = row.get('양식장 이름', '')
        facility_name = row.get('시설 이름', '')
        facility_id = row.get('시설 ID')
        
        if pd.isna(facility_id): continue # 시설 ID가 없는 행은 건너뜀
        
        facility_names = f"{region} {farm_name} {str(facility_name).replace(' ', '')}"
        
        if facility_id in facility_id_dict:
            if facility_id_dict[facility_id] != facility_names:
                print(f"시설 ID {facility_id}의 이름이 다릅니다: {facility_id_dict[facility_id]} vs {facility_names}")
        else:
            facility_id_dict[facility_id] = facility_names

    # df에서 일자와 시간을 합쳐서 datetime 형식으로 변환 후 일시라는 항목으로 저장
    raw_df['일시'] = pd.to_datetime(raw_df['일자'].astype(str) + ' ' + raw_df['시간'].astype(str), format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # 미사용 column 제거
    columns_to_drop = [
        '일자','시간','지역명','양식장 이름','시설 이름','양식장 ID','지역코드',
        '유입수 탁도(fnu)', '유량1(m3/h)','유량2(m3/h)','유량3(m3/h)',
        'ORP(mV)', '전압', '전력', 'RPM', '주파수','PH(pH)', '유입수 염도(PPT)', 'DO(mg/L)'
    ]
    df = raw_df.drop(columns=columns_to_drop, errors='ignore')
    return df

if __name__ == "__main__":
    # --- 이 스크립트 실행을 위한 가정 ---
    # 이 스크립트가 호출되기 전에, 이미 'raw_df'라는 이름의
    # 데이터프레임이 외부에서 로드되어 있다고 가정합니다.
    raw_df = get_dataframe()

    # --- 사용 예시 ---
    # 1. 일별 예측만 실행하고 싶을 경우
    print("\n\n>>> 예시 1: 일별 예측 실행")
    daily_forecast = run_aquaculture_forecast_pipeline(raw_df, units_to_run=['D'])
    if not daily_forecast.empty:
        print("\n===== 일별 통합 예측 결과 =====")
        print(daily_forecast)
    else:
        print("\n예측된 결과가 없습니다.")