import time
start = time.time()
print('ライブラリインポート中')

import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# 自作関数
from dataload.data_load import data_load
from func.my_func \
import \
    fit_MTS, predict_MD, determine_threshold, predict_MTSBag_ImpAgg, make_result_df, fit_WMTS, cal_WMD_by_reduced_model, predict_WMTS, predict_WMTSBag_ImpAgg

print('実験開始')

ex_name = input('実験名は?')

n_experiment = 10
data_list = [
    'yeast', 
    # 'wine', 
    # 'abalone', 
    # 'car',
    # 'cancer', 
    # 'letter'
    ]




# print('MTS開始！')

# for data in data_list:
#     print(data)
#     result_df = pd.DataFrame(
#                         columns=['AUC', 'accuracy', 'recall', 'Specificity', 'precision', 'gmeans', 'RS'],
#                         index=range(n_experiment))
#     for m in tqdm(range(n_experiment)):
        
#         X, y = data_load(data)

#         # バギング側の話
#         # ブートストラップサンプリングの個数
#         n = 10
#         seed = random.randint(0, n)

#         # 使用する7つの変数をランダムに取得する
#         # バギングをする際はそれぞれのサブサンプルで7つの変数を選択する．
#         random.seed(seed)
#         random_s = random.sample(list(X.columns), len(X.columns) if len(X.columns) < 7 else 7)
#         X = X[random_s]

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#         result_scaler, result_inv_C, select_columns = fit_MTS(X_train, y_train)

#         y_train_pred = predict_MD(X_train, result_scaler, result_inv_C, select_columns)

#         threshold = determine_threshold(y_train, y_train_pred)

#         y_proba = predict_MD(X_test, result_scaler, result_inv_C, select_columns)
#         y_pred = y_proba > threshold

#         result_df = make_result_df(result_df, y_test, y_pred, y_proba, m)
        
#     result_df.to_csv(f'../data/output/{ex_name}_MTS_{data}_result.csv')


# print('WMTS開始！')

# for data in data_list:
#     print(data)
#     result_df = pd.DataFrame(
#                         columns=['AUC', 'accuracy', 'recall', 'Specificity', 'precision', 'gmeans', 'RS'],
#                         index=range(n_experiment))
#     for m in tqdm(range(n_experiment)):
        
#         X, y = data_load(data)

#         # バギング側の話
#         # ブートストラップサンプリングの個数
#         n = 10
#         seed = random.randint(0, n)

#         # 使用する7つの変数をランダムに取得する
#         # バギングをする際はそれぞれのサブサンプルで7つの変数を選択する．
#         random.seed(seed)
#         random_s = random.sample(list(X.columns), len(X.columns) if len(X.columns) < 7 else 7)
#         X = X[random_s]

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#         reduced_model_scaler, reduced_model_inv_C, select_columns, select_columns_weight = fit_WMTS(X_train, y_train)

#         y_train_pred = cal_WMD_by_reduced_model(X_train, reduced_model_scaler, reduced_model_inv_C, select_columns, select_columns_weight)

#         threshold = determine_threshold(y_train, y_train_pred)

#         proba, pred = predict_WMTS(X_test, reduced_model_scaler, reduced_model_inv_C, select_columns, select_columns_weight, threshold)

#         result_df = make_result_df(result_df, y_test, pred, proba, m)
        
#     result_df.to_csv(f'../data/output/{ex_name}_WMTS_{data}_result.csv')



# print('ImpAggMTSBag開始')

# for data in data_list:
#     print(data)
#     result_df = pd.DataFrame(
#                         columns=['AUC', 'accuracy', 'recall', 'Specificity', 'precision', 'gmeans', 'RS'],
#                         index=range(n_experiment))
#     for m in tqdm(range(n_experiment)):
        
#         X, y = data_load(data)

#         # パラメータ
#         n_estimators = 10
#         max_samples = 0.5

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#         # 実行するところ

#         # n_estimators:再標本化の回数 SIZE:再標本化されたもののサンプルサイズ
#         SIZE = int(len(X) * max_samples)

#         # 予測に必要なパラメータ
#         select_columns = [0] * n_estimators
#         result_scaler = [0] * n_estimators
#         result_inv_C = [0] * n_estimators
#         threshold = [0] * n_estimators

#         for i in range(n_estimators):
#             # bootstrap sampling
#             resampled_data_x, resampled_data_y = resample(X_train, y_train, n_samples = SIZE)
#             random_s = random.sample(
#                 list(resampled_data_x.columns), 
#                 len(resampled_data_x.columns) 
#                 if len(resampled_data_x.columns) < 7 
#                 else 7
#                 )
#             resampled_data_x = resampled_data_x[random_s]

#             result_scaler[i], result_inv_C[i], select_columns[i] = fit_MTS(resampled_data_x, resampled_data_y)

#             y_train_pred = predict_MD(resampled_data_x, result_scaler[i], result_inv_C[i], select_columns[i])

#             threshold[i] = determine_threshold(resampled_data_y, y_train_pred)
            
#         y_proba, y_pred = predict_MTSBag_ImpAgg(X_test, result_scaler, result_inv_C, select_columns, threshold, n_estimators)

#         result_df = make_result_df(result_df, y_test, y_pred, y_proba, m)
        
#     result_df.to_csv(f'../data/output/{ex_name}_MTSBagImpAgg_{data}_result.csv')


print('ImpAggWMTSBag開始')

for data in data_list:
    print(data)
    result_df = pd.DataFrame(
                        columns=['AUC', 'accuracy', 'recall', 'Specificity', 'precision', 'gmeans', 'RS'],
                        index=range(n_experiment))
    for m in tqdm(range(n_experiment)):
        
        X, y = data_load(data)

        # パラメータ
        n_estimators = 10
        max_samples = 0.5

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # 実行するところ

        # n_estimators:再標本化の回数 SIZE:再標本化されたもののサンプルサイズ
        SIZE = int(len(X) * max_samples)

        # 予測に必要なパラメータ
        select_columns = [0] * n_estimators
        select_columns_weight = [0] * n_estimators
        result_scaler = [0] * n_estimators
        result_inv_C = [0] * n_estimators
        threshold = [0] * n_estimators

        for i in range(n_estimators):
            # bootstrap sampling
            resampled_data_x, resampled_data_y = resample(X_train, y_train, n_samples = SIZE)
            random_s = random.sample(
                list(resampled_data_x.columns), 
                len(resampled_data_x.columns) 
                if len(resampled_data_x.columns) < 7 
                else 7
                )
            resampled_data_x = resampled_data_x[random_s]

            result_scaler[i], result_inv_C[i], select_columns[i], select_columns_weight[i] = fit_WMTS(resampled_data_x, resampled_data_y)

            y_train_pred = cal_WMD_by_reduced_model(resampled_data_x, result_scaler[i], result_inv_C[i], select_columns[i], select_columns_weight[i])

            threshold[i] = determine_threshold(resampled_data_y, y_train_pred)

        y_proba, y_pred = predict_WMTSBag_ImpAgg(X_test, result_scaler, result_inv_C, select_columns, select_columns_weight, threshold, n_estimators)

        result_df = make_result_df(result_df, y_test, y_pred, y_proba, m)
        
    result_df.to_csv(f'../data/output/{ex_name}_WMTSBagImpAgg_{data}_result.csv')

process_time = time.time() - start

print('実行時間は', process_time)