# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
from typing import List, Tuple

# Matplotlib과 한글 폰트 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 부분을 제거하고, 마이너스 기호만 정상 표시되도록 유지합니다.
plt.rcParams['axes.unicode_minus'] = False 

def process_and_prepare_master_data(input_df: pd.DataFrame, today_date: datetime.date) -> dict:
    """
    1단계: 데이터 준비 (Data Preparation)
    전달받은 데이터프레임을 정제하고, 시설 위치별 '일별 대표 데이터셋'(마스터 데이터)을 생성합니다.
    """
    print("===== 1단계: 데이터 준비 시작 =====")
    
    df = input_df.copy()
    df['일시'] = pd.to_datetime(df['일시'])
    
    # 학습 데이터는 '오늘' 이전의 모든 데이터를 포함합니다.
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

def train_and_predict_prophet(df_master_daily: pd.DataFrame, today_date: datetime.date, freq_unit: str = 'D', periods: int = 7) -> Tuple[pd.DataFrame, Prophet, pd.DataFrame, pd.DataFrame]:
    """
    2단계: Prophet 모델 예측 실행 (Prediction Execution with Prophet)
    [수정] 데이터의 마지막 날짜와 상관없이, '오늘'을 기준으로 미래를 예측하도록 로직을 수정합니다.
    """
    # 1. 학습 데이터의 시작일부터 "어제"까지의 전체 기간을 정의합니다.
    #    이렇게 하면 데이터가 중간에 끊겼더라도 모델이 그 공백 기간을 인지하게 됩니다.
    start_date = df_master_daily['ds'].min()
    end_date = today_date - timedelta(days=1)
    
    # [수정] Pandas Timestamp와 datetime.date 객체 비교 오류 해결
    if start_date.date() > end_date:
        print("!!! 경고: 학습 데이터가 '오늘' 이후에만 존재하여 예측을 건너뜁니다.")
        return pd.DataFrame(), None, None, None

    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    history_df = pd.DataFrame(full_date_range, columns=['ds'])
    
    # 전체 기간에 실제 마스터 데이터를 합칩니다. 데이터가 없는 날은 'y'값이 NaN이 됩니다.
    df_padded = pd.merge(history_df, df_master_daily, on='ds', how='left')
    
    # 2. 예측 단위별 데이터 리샘플링 및 필터링
    df_to_process = df_padded.copy().set_index('ds')
    
    if freq_unit == 'D':
        df_resampled = df_to_process.resample('D').median()
    elif freq_unit == 'W':
        df_resampled = df_to_process.resample('W-MON').median()
    elif freq_unit == 'M':
        df_resampled = df_to_process.resample('MS').median()
    else:
        raise ValueError("freq_unit은 반드시 'D', 'W', 'M' 중 하나여야 합니다.")
    
    # Prophet은 학습 데이터의 y값이 NaN이어도 괜찮으므로, dropna는 reset_index 후에 수행합니다.
    df_resampled.reset_index(inplace=True)

    if len(df_resampled.dropna()) < 15: # 실제 데이터 포인트가 너무 적으면 예측하지 않음
        print(f"!!! 경고: 실제 학습 데이터가 {len(df_resampled.dropna())}개로 너무 적어 예측을 건너뜁니다.")
        return pd.DataFrame(), None, None, None

    # 3. Prophet 모델 학습
    model = Prophet()
    model.fit(df_resampled)

    # 4. 미래 예측 (7개 기간)
    # 이제 모델의 학습 데이터 마지막 날짜는 '어제'이므로, 미래 예측은 '오늘'부터 시작됩니다.
    future = model.make_future_dataframe(periods=periods, freq=freq_unit)
    forecast = model.predict(future)

    # 5. 결과 추출 및 반환
    result = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    result.rename(columns={'ds' : '예측시점', 'yhat': '예측온도', 'yhat_lower': '불확실성_최저', 'yhat_upper': '불확실성_최고'}, inplace=True)
    
    # df_resampled는 실제 값만 있는 원본 리샘플링 데이터로 재정의하여 시각화에 사용
    df_resampled.dropna(inplace=True)
    return result, model, forecast, df_resampled

def run_aquaculture_forecast_pipeline(
    raw_df: pd.DataFrame, 
    units_to_run: List[str] = ['D', 'W', 'M'],
    visualize_for_facility: str = None,
    visualize_units: List[str] = ['D'],
    save_graph_path: str = 'graph'
) -> pd.DataFrame:
    """
    데이터 정제부터 Prophet 모델 기반 주기별 예측까지 전체 파이프라인을 실행합니다.
    """
    today = datetime.now().date()
    all_forecast_results = []
    UNIT_MAP_KO = {'D': '일', 'W': '주', 'M': '월'} 
    UNIT_MAP_EN = {'D': 'Day', 'W': 'Week', 'M': 'Month'}

    try:
        master_datasets = process_and_prepare_master_data(raw_df, today)

        for facility_code, df_master in master_datasets.items():
            print(f"\n===== '{facility_code}' 시설 예측 시작 (Prophet 모델) =====")
            
            for freq in units_to_run:
                name_ko = UNIT_MAP_KO.get(freq)
                name_en = UNIT_MAP_EN.get(freq)
                if not name_ko:
                    print(f"!!! 경고: 알 수 없는 예측 단위 '{freq}'는 건너뜁니다.")
                    continue
                
                print(f"--- {name_ko} 단위 예측 (향후 7{name_ko}) ---")
                try:
                    # [수정] today 변수를 예측 함수에 전달합니다.
                    forecast_result, model, forecast_df, df_resampled = train_and_predict_prophet(df_master, today, freq_unit=freq, periods=7)
                    
                    if not forecast_result.empty:
                        forecast_result['시설코드'] = facility_code
                        forecast_result['예측단위'] = name_ko
                        all_forecast_results.append(forecast_result)
                        
                        if facility_code == visualize_for_facility and freq in visualize_units:
                            print(f"--- '{facility_code}' ({name_ko} 단위) 그래프 생성 및 저장 중 ---")
                            
                            target_folder = os.path.join(save_graph_path, facility_code)
                            os.makedirs(target_folder, exist_ok=True)
                            
                            # 그래프 1: 마스터 데이터 그래프 저장
                            fig1, ax1 = plt.subplots(figsize=(12, 6))
                            df_master.plot(x='ds', y='y', ax=ax1, label='Daily Representative Temp.', color='royalblue')
                            ax1.set_title(f"[{facility_code}] Master Water Temperature Data")
                            ax1.set_xlabel("Date"); ax1.set_ylabel("Temperature (℃)")
                            plt.legend(); plt.tight_layout()
                            plt.savefig(os.path.join(target_folder, f"1_master_data.png")); plt.close(fig1)
                            
                            # 그래프 2: 모델 학습 결과 및 전체 추세선 저장
                            fig_fit, ax_fit = plt.subplots(figsize=(12, 6))
                            ax_fit.scatter(df_resampled['ds'], df_resampled['y'], color='k', label='Actual Data', s=10, alpha=0.6)
                            ax_fit.plot(forecast_df['ds'], forecast_df['yhat'], color='dodgerblue', label='Model Fit & Forecast')
                            ax_fit.fill_between(
                                forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                                color='skyblue', alpha=0.4, label='Uncertainty Interval'
                            )
                            ax_fit.set_title(f"[{facility_code}] Model Fit and Full Trendline")
                            ax_fit.set_xlabel("Date"); ax_fit.set_ylabel("Temperature (℃)")
                            ax_fit.legend(); ax_fit.grid(True, linestyle='--', alpha=0.6)
                            plt.tight_layout(); plt.savefig(os.path.join(target_folder, f"2_{freq}_model_fit.png")); plt.close(fig_fit)

                            # 그래프 3: 미래 예측 구간 그래프 저장
                            fig2, ax2 = plt.subplots(figsize=(12, 6))
                            future_forecast_df = forecast_df[forecast_df['ds'] >= datetime.combine(today, datetime.min.time())]
                            ax2.plot(future_forecast_df['ds'], future_forecast_df['yhat'], ls='--', marker='o', color='dodgerblue', label='Forecasted Temperature')
                            ax2.fill_between(
                                future_forecast_df['ds'], future_forecast_df['yhat_lower'], future_forecast_df['yhat_upper'],
                                color='skyblue', alpha=0.4, label='Uncertainty Interval'
                            )
                            ax2.set_title(f"[{facility_code}] Future Forecast for next 7 {name_en}s")
                            ax2.set_xlabel("Date"); ax2.set_ylabel("Temperature (℃)")
                            ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)
                            fig2.autofmt_xdate(); plt.tight_layout()
                            plt.savefig(os.path.join(target_folder, f"3_{freq}_future_forecast.png")); plt.close(fig2)

                            # 그래프 4: 패턴 분석 그래프 저장
                            fig3 = model.plot_components(forecast_df)
                            fig3.suptitle(f"[{facility_code}] Data Pattern Analysis", y=1.02)
                            plt.tight_layout(); plt.savefig(os.path.join(target_folder, f"4_{freq}_components.png")); plt.close(fig3)
                            print(f"-> '{target_folder}'에 그래프 4종 저장 완료.")
                    
                except Exception as e:
                    print(f"!!! '{name_ko}' 단위 예측 중 오류 발생: {e}")

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
    import pickle
    pickle_file = 'raw_df.pkl'
    if os.path.exists(pickle_file):
        print(f"'{pickle_file}' 파일에서 데이터프레임을 불러옵니다...")
        with open(pickle_file, 'rb') as f:
            raw_df = pickle.load(f)
        return raw_df
    else:
        print(f"'{pickle_file}' 파일이 존재하지 않습니다. 새로 데이터를 읽어옵니다...")
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
        # df를 피클로 저장
        with open(pickle_file, 'wb') as f:
            pickle.dump(df, f)
        print(f"-> 데이터프레임을 '{pickle_file}'로 저장했습니다.")
        return df

if __name__ == "__main__":
    # --- 이 스크립트 실행을 위한 가정 ---
    # 이 스크립트가 호출되기 전에, 이미 'raw_df'라는 이름의
    # 데이터프레임이 외부에서 로드되어 있다고 가정합니다.
    raw_df = get_dataframe()

    # --- 사용 예시 ---
    # 예시 1: '64AF1' 시설에 대해 '일별' 예측 그래프를 함께 생성하면서 전체 파이프라인 실행
    print("\n\n>>> 1. 특정 시설 그래프를 포함하여 파이프라인 실행")
    final_df_with_viz = run_aquaculture_forecast_pipeline(
        raw_df, 
        units_to_run=['D'], 
        visualize_for_facility='64AF1',
        visualize_units=['D']
    )
    if not final_df_with_viz.empty:
        print("\n\n===== '64AF1' 그래프와 함께 생성된 예측 결과 =====")
        with pd.option_context('display.float_format', '{:,.2f}'.format):
            print(final_df_with_viz)
            
    # 예시 2: 시각화 없이 예측 결과만 생성
    print("\n\n>>> 2. 시각화 없이 예측만 실행")
    final_df_only = run_aquaculture_forecast_pipeline(raw_df)
    if not final_df_only.empty:
        print("\n\n===== 최종 통합 예측 결과 =====")
        with pd.option_context('display.float_format', '{:,.2f}'.format):
            print(final_df_only)