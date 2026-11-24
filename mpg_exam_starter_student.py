"""
==============================================================================
[학생 배포용 코드] 정보과학과제연구 기말 수행평가 2026.11.24(화)
주제: 자동차 제원 데이터(mpg)를 활용한 연비 예측 파이프라인 구축 및 오차 분석
------------------------------------------------------------------------------
학번: (여기에 입력)
이름: (여기에 입력)
------------------------------------------------------------------------------
[제출 및 유의사항]
1. 오픈북 수행평가이며, 전체 작업 과정의 OBS 녹화 영상과 GitHub 저장소 링크를 제출합니다.
2. 모든 코드 실행 시 에러가 없어야 하며, 각 단계 하단의 [분석 및 해석 작성] 주석에
   본인의 의사결정 근거와 결과 해석을 반드시 서술해야 합니다.
3. 데이터 누수(Data Leakage) 주의: 테스트 세트는 1단계 분할 직후부터
   3단계 최종 평가 전까지 EDA, 시각화, 전처리 fit 등에 절대 사용하지 마시오.
4. VS Code / PyCharm에서 '# %%' 단위 셀 실행 또는 전체 스크립트 실행 모두 가능합니다.
==============================================================================
"""

# %% [0] 기본 라이브러리 임포트
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 생성된 그래프를 저장할 폴더 (필요 시 활용)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# %% =========================================================================
# 1단계: 데이터 탐색(EDA) 및 계층적 샘플링
# ============================================================================
# TODO 1-1: sns.load_dataset("mpg")로 데이터를 불러오고,
#           결측치 개수, 데이터 타입, 수치형 변수의 기술통계 및 왜도(skew)를 확인하시오.
mpg = sns.load_dataset("mpg")


# TODO 1-2: origin 또는 구간화한 weight를 기준으로 훈련 세트(80%)와 테스트 세트(20%)로
#           계층적 샘플링(StratifiedShuffleSplit 등)을 수행하시오.
#           분할 후 원본/훈련/테스트 세트의 계층 비율이 유지되었는지 표(DataFrame)로 출력하시오.
strat_train_set = None
strat_test_set = None


# TODO 1-3: 훈련 세트(strat_train_set)만을 사용하여 mpg와 주요 특성 간의 관계를
#           시각화(히트맵, 산점도 등 최소 2종)하시오.
#           (그래프는 plt.show()로 확인하거나 outputs/ 폴더에 저장하시오.)


"""
[1단계 분석 및 해석 작성]
1. 결측치 및 데이터 분포(왜도) 특징:
   - (여기에 작성)
2. 계층적 샘플링 기준 선택 이유 및 비율 검증 결과:
   - (여기에 작성)
3. 시각화 결과 해석 (mpg에 가장 큰 영향을 미치는 특성 및 비선형성 여부):
   - (여기에 작성)
"""


# %% =========================================================================
# 2단계: 커스텀 전처리 및 파이프라인 구축
# ============================================================================
# TODO 2-0: 훈련 세트와 테스트 세트에서 특성(X)과 타깃(y = 'mpg')을 분리하시오.
# X_train = ...
# y_train = ...
# X_test = ...
# y_test = ...


# TODO 2-1: 도메인 기반 파생변수(최소 2개)를 생성하는 커스텀 변환기 또는 FunctionTransformer를 작성하시오.
#           예시: 마력 대 무게비(horsepower / weight), 배기량 대 실린더비(displacement / cylinders)
class AutoFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # TODO: 파생변수를 생성하여 기존 배열에 결합 후 반환하시오.
        pass


# TODO 2-2: (심화 가산점 - 선택) name 컬럼에서 차량 브랜드(첫 단어)를 추출하는 변환기를 구현하시오.


# TODO 2-3: ColumnTransformer와 Pipeline을 결합하여 단일 전처리 파이프라인(preprocessing)을 구축하시오.
#           포함 요소: 결측치 대체(SimpleImputer), 파생변수 추가, 로그 변환/표준화(StandardScaler),
#           범주형 원-핫 인코딩(OneHotEncoder)
preprocessing = None


"""
[2단계 분석 및 해석 작성]
1. 결측치 대체 전략 및 파생변수 2종을 설계한 물리적/도메인 근거:
   - (여기에 작성)
2. 왜도가 큰 변수 처리 및 파이프라인(ColumnTransformer) 구성 이유:
   - (여기에 작성)
"""


# %% =========================================================================
# 3단계: 모델 비교, 하이퍼파라미터 튜닝 및 평가
# ============================================================================
# TODO 3-1: 전처리 파이프라인과 결합된 최소 3종의 회귀 모델을 5-폴드 교차 검증(cross_val_score)으로
#           비교하고, 각 모델의 평균 RMSE와 표준편차를 출력하시오.


# TODO 3-2: 가장 우수한 성능의 모델 1종을 선택하여 GridSearchCV 또는 RandomizedSearchCV로
#           하이퍼파라미터를 튜닝하고, 최적 파라미터와 교차 검증 RMSE를 출력하시오.


# TODO 3-3: 최종 선정된 모델을 테스트 세트(X_test, y_test)로 평가(RMSE)하고,
#           예측 오차의 95% 신뢰 구간을 NumPy/SciPy로 계산하여 출력하시오.


"""
[3단계 분석 및 해석 작성]
1. 교차 검증 모델 비교 결과 및 최종 모델 선정 근거:
   - (여기에 작성)
2. 하이퍼파라미터 튜닝 전후 성능 변화 및 테스트 세트 95% 신뢰 구간 해석:
   - (여기에 작성)
"""


# %% =========================================================================
# 4단계: 오차 분석 및 GitHub 제출
# ============================================================================
# TODO 4-1: 테스트 세트에서 실제값(actual_mpg)과 예측값(pred_mpg)의 절대 오차가 가장 큰
#           상위 3개 차량을 Pandas로 추출하여 제원과 함께 출력하시오.


# TODO 4-2: (선택) 실제값 vs 예측값 산점도를 그려 상위 3개 오차 차량의 위치를 확인하시오.


"""
[4단계 분석 및 해석 작성]
1. 오차가 가장 큰 상위 3개 차량 목록 및 제원 특징:
   - (여기에 작성)
2. 모델이 해당 차량의 연비를 잘못 예측한 원인 추론 및 향후 모델 개선 아이디어:
   - (여기에 작성)
"""
