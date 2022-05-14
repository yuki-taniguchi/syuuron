import time

print('ライブラリインポート中')

import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# 自作関数
from dataload.data_load import data_load
from func.my_func import *

print('実験開始')

ex_name = input('実験名は?')

n_experiment = 10
data_list = [
    # 'yeast', 
    'wine', 
    'abalone', 
    'car',
    'cancer', 
    'letter'
    ]

def exp_ImpAggMTSBag():

    print('ImpAggMTSBag開始')

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

                result_scaler[i], result_inv_C[i], select_columns[i] = fit_MTS(resampled_data_x, resampled_data_y)

                y_train_pred = predict_MD(resampled_data_x, result_scaler[i], result_inv_C[i], select_columns[i])

                threshold[i] = determine_threshold(resampled_data_y, y_train_pred)
                
            y_proba, y_pred = predict_MTSBag_ImpAgg(X_test, result_scaler, result_inv_C, select_columns, threshold, n_estimators)

            result_df = make_result_df(result_df, y_test, y_pred, y_proba, m)
            
        result_df.to_csv(f'../data/output/{ex_name}_MTSBagImpAgg_{data}_result.csv')


def exp_ImpAggMTGSBag():

    print('ImpAggMTGSBag開始')

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
            result_scaler = [0] * n_estimators
            result_t = [0] * n_estimators
            result_ips = [0] * n_estimators
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

                result_scaler[i], result_t[i], result_ips[i],  select_columns[i] = fit_MTGS(resampled_data_x, resampled_data_y)

                y_train_pred = predict_gram_MD(resampled_data_x, result_scaler[i], result_t[i], result_ips[i],  select_columns[i])

                threshold[i] = determine_threshold(resampled_data_y, y_train_pred)
                
            y_proba, y_pred = predict_MTGSBag_ImpAgg(X_test, result_scaler, result_t, result_ips, select_columns, threshold, n_estimators)

            result_df = make_result_df(result_df, y_test, y_pred, y_proba, m)
            
        result_df.to_csv(f'../data/output/{ex_name}_MTGSBagImpAgg_{data}_result.csv')


def exp_ImpAggWMTSBag():
    
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

def exp_ImpAggWMTGSBag():
    
    print('ImpAggWMTGSBag開始')

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
            reduced_model_scaler = [0] * n_estimators
            reduced_model_t = [0] * n_estimators
            reduced_model_ips = [0] * n_estimators
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

                reduced_model_scaler[i], reduced_model_t[i], reduced_model_ips[i], select_columns[i], select_columns_weight[i] = fit_WMTGS(resampled_data_x, resampled_data_y)

                y_train_pred = cal_gram_WMD_by_reduced_model(resampled_data_x, reduced_model_scaler[i], reduced_model_t[i], reduced_model_ips[i], select_columns[i], select_columns_weight[i])

                threshold[i] = determine_threshold(resampled_data_y, y_train_pred)

            y_proba, y_pred = predict_WMTGSBag_ImpAgg(X_test, reduced_model_scaler, reduced_model_t, reduced_model_ips, select_columns, select_columns_weight, threshold, n_estimators)

            result_df = make_result_df(result_df, y_test, y_pred, y_proba, m)
            
        result_df.to_csv(f'../data/output/{ex_name}_WMTGSBagImpAgg_{data}_result.csv')

start = time.time()

exp_ImpAggWMTGSBag()
exp_ImpAggMTGSBag()
exp_ImpAggMTSBag()
exp_ImpAggWMTSBag()

process_time = time.time() - start

print('実行時間は', process_time)