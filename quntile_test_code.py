import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (예시: 나눔고딕)
# 사용하시는 환경에 따라 폰트 경로를 확인하고 설정해야 합니다.
# Colab 등에서는 아래 주석을 해제하고 실행하여 폰트를 설치하고 런타임을 재시작해야 할 수 있습니다.
# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf
# plt.rc('font', family='NanumGothic')

# 그래도 폰트가 깨질 경우, 아래 코드로 사용 가능한 한글 폰트를 확인하고 설정하세요.
# print([font.name for font in fm.fontManager.ttflist if 'Nanum' in font.name or 'AppleGothic' in font.name or 'Malgun' in font.name]) # 사용 가능한 폰트 목록 확인
try:
    plt.rcParams['font.family'] = 'NanumGothic' # Windows, Colab
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic' # MacOS
    except:
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
        except:
            print("한글 폰트를 찾을 수 없습니다. 기본 폰트로 설정됩니다. 그래프의 한글이 깨질 수 있습니다.")
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지


# 1. 샘플 시계열 데이터 생성
np.random.seed(42)
n_points = 100
time = np.arange(n_points)
# 시간에 따라 증가하는 추세와 계절성, 그리고 노이즈를 포함하는 데이터 생성
values = 0.5 * time + 10 * np.sin(time * 0.2) + np.random.normal(0, 2, n_points)
# 데이터 값의 변동성이 시간에 따라 증가하도록 설정 (이분산성 예시)
values_heteroscedastic = values * (1 + 0.02 * time) # 시간에 따라 분산이 커지도록 함

data = pd.DataFrame({'시간': time, '값': values_heteroscedastic})

print("## 생성된 샘플 데이터 (처음 5개 행):")
print(data.head())
print("\n")

# 2. 분위수 회귀 모델 학습
# X는 시간, y는 값
# statsmodels는 종속 변수와 독립 변수 간의 관계를 모델링합니다.
# 여기서는 간단한 선형 관계를 가정하고 시간(독립변수)을 사용합니다.
# 좀 더 복잡한 시계열 패턴(예: 계절성)이 있다면, 이를 반영하는 변수(예: 푸리에 항)를 추가할 수 있습니다.
X = data['시간']
y = data['값']

# 분위수 설정
quantiles = [0.05, 0.5, 0.95]
models = []
predictions = pd.DataFrame({'시간': data['시간'], '실제값': data['값']})

for q in quantiles:
    # 모델 정의: '값 ~ 시간'은 값이 시간에 따라 변하는 것을 의미합니다.
    # 데이터프레임을 직접 사용할 경우, 컬럼명을 그대로 사용합니다.
    model = smf.quantreg(f'값 ~ 시간', data) # formula 방식
    res = model.fit(q=q)
    models.append(res)
    predictions[f'q_{q}'] = res.predict(data[['시간']]) # 예측 시에는 학습에 사용된 독립변수와 동일한 형태의 데이터를 전달
    print(f"## {q*100:.0f}% 분위수 회귀 모델 요약:")
    print(res.summary())
    print("\n")


# 3. 결과 시각화
plt.figure(figsize=(12, 7))
plt.plot(predictions['시간'], predictions['실제값'], 'bo', markersize=4, label='실제 데이터')
plt.plot(predictions['시간'], predictions['q_0.5'], color='red', linestyle='-', linewidth=2, label='중앙값 예측 (Q50)')
plt.plot(predictions['시간'], predictions['q_0.05'], color='gray', linestyle='--', linewidth=2, label='하위 5% 예측 (Q5)')
plt.plot(predictions['시간'], predictions['q_0.95'], color='gray', linestyle='--', linewidth=2, label='상위 95% 예측 (Q95)')

plt.fill_between(predictions['시간'], predictions['q_0.05'], predictions['q_0.95'], color='lightgray', alpha=0.6, label='90% 예측 범위')

plt.title('시계열 데이터 분위수 회귀 예측 범위', fontsize=16)
plt.xlabel('시간', fontsize=14)
plt.ylabel('값', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n## 예측 결과 (처음 5개 행):")
print(predictions.head())