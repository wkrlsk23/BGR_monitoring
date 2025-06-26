# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
from typing import List, Tuple
import argparse

# Matplotlib은 시각화 기능이 호출될 때만 사용되므로, 그대로 둡니다.
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False 

# 1단계: 원본 데이터를 정제하고, 시설 위치별 '일별 대표 데이터셋'을 생성합니다.
def process_and_prepare_master_data(input_df: pd.DataFrame, today_date: datetime.date) -> dict:
    """
    전달받은 데이터프레임을 2단계 필터링(동적, 정적)으로 정제하고,
    시설별로 일별 중앙값 데이터를 계산하여 마스터 데이터셋을 만듭니다.

    Args:
        input_df (pd.DataFrame): '일시', '시설 ID', '수온(℃)' 컬럼을 포함한 원본 데이터프레임.
        today_date (datetime.date): 예측 기준일("오늘"). 이 날짜 이전의 데이터만 사용합니다.

    Returns:
        dict: {'시설 코드': master_dataframe} 형태의 딕셔너리.
    """
    print("===== 1단계: 데이터 준비 시작 =====")
    
    df = input_df.copy()
    
    df['일시'] = pd.to_datetime(df['일시'])
    df = df[df['일시'] < datetime.combine(today_date, datetime.min.time())]
    print(f"-> {today_date} 이전 데이터만 사용. (총 {len(df)}개)")

    df.sort_values(by=['시설 ID', '일시'], inplace=True)

    df['temp_diff'] = df.groupby('시설 ID')['수온(℃)'].diff().abs()
    change_threshold = 5.0
    change_outliers = df['temp_diff'] > change_threshold
    print(f"-> 동적 필터링: 급격한 변화(>{change_threshold}℃) 데이터 {change_outliers.sum()}개 제거.")
    df.loc[change_outliers, '수온(℃)'] = np.nan
    df.drop(columns=['temp_diff'], inplace=True)
    
    lower_bound, upper_bound = 5.0, 30.0
    abs_outliers = (df['수온(℃)'] < lower_bound) | (df['수온(℃)'] > upper_bound)
    print(f"-> 정적 필터링: 유효범위({lower_bound}℃ ~ {upper_bound}℃) 밖의 데이터 {abs_outliers.sum()}개 제거.")
    df.loc[abs_outliers, '수온(℃)'] = np.nan
    
    df.dropna(subset=['수온(℃)'], inplace=True)
    df['시설 코드'] = df['시설 ID'].str[:5]
    
    master_datasets = {}
    for code, group_df in df.groupby('시설 코드'):
        daily_master = group_df.set_index('일시')[['수온(℃)']].resample('D').median()
        daily_master.interpolate(method='time', inplace=True)
        daily_master.reset_index(inplace=True)
        daily_master.rename(columns={'일시': 'ds', '수온(℃)': 'y'}, inplace=True)
        master_datasets[code] = daily_master
        print(f"-> '{code}' 시설의 마스터 데이터 생성 완료 ({len(daily_master)}일치).")
        
    print("===== 1단계: 데이터 준비 완료 =====")
    return master_datasets

# 2단계: Prophet 모델을 사용하여 미래 수온을 예측합니다.
def train_and_predict_prophet(df_master_daily: pd.DataFrame, today_date: datetime.date, freq_unit: str = 'D', periods: int = 7) -> Tuple[pd.DataFrame, Prophet, pd.DataFrame, pd.DataFrame]:
    """
    '오늘'을 기준으로, 주어진 주기에 맞춰 데이터를 리샘플링하고 Prophet 모델을 학습시켜
    미래 7개 기간의 수온을 예측합니다.

    Args:
        df_master_daily (pd.DataFrame): 1단계에서 생성된 일별 마스터 데이터프레임.
        today_date (datetime.date): 예측 기준일.
        freq_unit (str): 예측 주기 단위 ('D', 'W', 'M').
        periods (int): 예측할 기간의 수.

    Returns:
        Tuple: 예측 결과(df), 학습된 모델, 전체 예측 데이터(df), 학습에 사용된 데이터(df).
    """
    start_date = df_master_daily['ds'].min()
    end_date = today_date - timedelta(days=1)
    
    if start_date.date() > end_date:
        print("!!! 경고: 학습 데이터가 '오늘' 이후에만 존재하여 예측을 건너뜁니다.")
        return pd.DataFrame(), None, None, None

    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    history_df = pd.DataFrame(full_date_range, columns=['ds'])
    
    df_padded = pd.merge(history_df, df_master_daily, on='ds', how='left')
    
    df_to_process = df_padded.copy().set_index('ds')
    
    if freq_unit == 'D':
        df_resampled = df_to_process.resample('D').median()
    elif freq_unit == 'W':
        df_resampled = df_to_process.resample('W-MON').median()
    elif freq_unit == 'M':
        df_resampled = df_to_process.resample('M').median()
    else:
        raise ValueError("freq_unit은 반드시 'D', 'W', 'M' 중 하나여야 합니다.")
        
    df_resampled.reset_index(inplace=True)

    if len(df_resampled.dropna()) < 15:
        print(f"!!! 경고: 실제 학습 데이터가 {len(df_resampled.dropna())}개로 너무 적어 예측을 건너뜁니다.")
        return pd.DataFrame(), None, None, None

    model = Prophet()
    model.fit(df_resampled)
    future = model.make_future_dataframe(periods=periods, freq=freq_unit)
    forecast = model.predict(future)

    result = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    result.rename(columns={'ds' : '예측시점', 'yhat': '예측온도', 'yhat_lower': '불확실성_최저', 'yhat_upper': '불확실성_최고'}, inplace=True)
    
    df_resampled.dropna(inplace=True)
    return result, model, forecast, df_resampled

# 메인 파이프라인: 데이터 로딩, 정제, 예측, 결과 변환의 전체 과정을 조율합니다.
def run_aquaculture_forecast_pipeline(
    csv_file_path: str,
    units_to_run: List[str] = ['D', 'W', 'M'],
    chunk_size: int = 1000000
) -> pd.DataFrame:
    """
    대용량 CSV 파일을 청크 단위로 처리하여 데이터 정제부터 주기별 예측까지
    전체 파이프라인을 실행하고, 통합된 예측 결과를 반환합니다.

    Args:
        csv_file_path (str): 원본 데이터 CSV 파일 경로.
        units_to_run (List[str]): 예측을 실행할 단위 리스트.
        chunk_size (int): 한 번에 읽어올 데이터 청크 크기.

    Returns:
        pd.DataFrame: 모든 시설과 예측단위에 대한 통합 예측 결과.
    """
    today = datetime.now().date()
    
    REGION_MAP = {
        '64AF1': '제주 비봉수산', '64AF3': '제주 신양수산', '61AF3': '완도 신호수산',
        '65AF1': '해남 정우수산', '64AF2': '제주 바다드림영어조합법인', '61AF1': '완도 그린수산'
    }
    allowed_codes = set(REGION_MAP.keys())

    print("===== 청크 단위 데이터 로딩 및 기본 정제 시작 =====")
    required_cols = ['wt', 'fclty_id', 'collect_de', 'collect_time']
    try:
        chunk_iterator = pd.read_csv(csv_file_path, usecols=required_cols, chunksize=chunk_size, low_memory=False)
    except FileNotFoundError:
        print(f"!!! 오류: '{csv_file_path}' 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    except ValueError as e:
        print(f"!!! 오류: CSV 파일에 필요한 컬럼이 없습니다. (세부사항: {e})")
        return pd.DataFrame()

    facility_data_accumulator = {}
    for i, chunk in enumerate(chunk_iterator):
        print(f"-> 청크 {i+1} 처리 중...")
        try:
            chunk.rename(columns={'wt': '수온(℃)', 'fclty_id': '시설 ID'}, inplace=True)
            chunk['일시'] = pd.to_datetime(chunk['collect_de'] + ' ' + chunk['collect_time'], errors='coerce')
        except KeyError as e:
            print(f"!!! 오류: 청크에 필요한 컬럼({e})이 없습니다. CSV 파일의 컬럼명을 확인해주세요.")
            continue 

        chunk.dropna(subset=['일시', '수온(℃)', '시설 ID'], inplace=True)
        
        chunk['시설 코드'] = chunk['시설 ID'].str[:5]
        chunk = chunk[chunk['시설 코드'].isin(allowed_codes)]
        
        if chunk.empty:
            continue

        for code, group in chunk.groupby('시설 코드'):
            if code not in facility_data_accumulator:
                facility_data_accumulator[code] = []
            facility_data_accumulator[code].append(group[['일시', '시설 ID', '수온(℃)']])

    print("===== 모든 청크 처리 완료 =====")
    
    if not facility_data_accumulator:
        print("!!! 처리할 데이터가 없습니다.")
        return pd.DataFrame()

    full_raw_df = pd.concat([pd.concat(chunks) for chunks in facility_data_accumulator.values()])
    
    master_datasets = process_and_prepare_master_data(full_raw_df, today)

    all_forecast_results = []
    for facility_code, df_master in master_datasets.items():
        print(f"\n===== '{facility_code}' 시설 예측 시작 =====")
        for freq in units_to_run:
            print(f"--- {freq} 단위 예측 (향후 7{freq}) ---")
            try:
                forecast_result, _, _, _ = train_and_predict_prophet(df_master, today, freq_unit=freq, periods=7)
                
                if not forecast_result.empty:
                    forecast_result['시설코드'] = facility_code
                    forecast_result['예측단위'] = freq
                    all_forecast_results.append(forecast_result)
            except Exception as e:
                print(f"!!! '{freq}' 단위 예측 중 오류 발생: {e}")

    if all_forecast_results:
        final_df = pd.concat(all_forecast_results, ignore_index=True)
        
        final_df.rename(columns={
            '예측시점': 'prediction_date', '시설코드': 'code', '불확실성_최저': 'minimum_uncertainty',
            '예측온도': 'prediction_temp', '불확실성_최고': 'maximum_uncertainty', '예측단위': 'units_facility'
        }, inplace=True)
        
        final_df['region'] = final_df['code'].map(REGION_MAP).fillna('Unknown')
        final_df['idx'] = final_df.index
        
        transformed_df = final_df[[
            'idx', 'prediction_date', 'region', 'code', 'minimum_uncertainty', 
            'prediction_temp', 'maximum_uncertainty', 'units_facility'
        ]]
        return transformed_df
        
    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="양식장 수온 예측 파이프라인 (대용량 파일 처리)")
    parser.add_argument('-path', '--datapath', required=True, help="데이터 파일('raw_data.csv')이 위치한 폴더 경로")
    parser.add_argument('-u', '--unit', choices=['D', 'W', 'M'], default=None, help="예측을 실행할 단일 단위(D, W, M). 지정하지 않으면 모든 단위 실행.")
    args = parser.parse_args()
    csv_file_path = os.path.join(args.datapath, 'raw_data.csv')

    if args.unit:
        units = [args.unit]
    else:
        units = ['D', 'W', 'M']

    final_df = run_aquaculture_forecast_pipeline(csv_file_path=csv_file_path, units_to_run=units)

    if not final_df.empty:
        print("\n\n===== 최종 통합 예측 결과 =====")
        with pd.option_context('display.float_format', '{:,.2f}'.format):
            print(final_df)
        
        json_output_path = os.path.join(args.datapath, 'forecast_results.json')
        print(f"\n예측 결과를 '{json_output_path}' 파일로 저장합니다...")

        # [수정] JSON 저장을 위해 날짜 컬럼을 'YYYY-MM-DD' 형식의 문자열로 변환
        df_to_save = final_df.copy()
        df_to_save['prediction_date'] = df_to_save['prediction_date'].dt.strftime('%Y-%m-%d')
        
        df_to_save.to_json(json_output_path, orient='records', force_ascii=False, indent=4)
        print("JSON 파일 저장 완료.")
    else:
        print("\n\n최종 예측된 결과가 없습니다.")
