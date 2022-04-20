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
from func.my_func \
import \
    fit_MTS, predict_MD, determine_threshold, predict_MTSBag_, make_result_df

from concurrent.futures import ThreadPoolExecutor

def wrap_multi_thread_run(args):
    return multi_thread_run(*args)

def multi_thread_run(X_train, y_train, SIZE):
    # bootstrap sampling
    resampled_data_x, resampled_data_y = resample(X_train, y_train, n_samples = SIZE)
    random_s = random.sample(
        list(resampled_data_x.columns), 
        len(resampled_data_x.columns) 
        if len(resampled_data_x.columns) < 7 
        else 7
        )
    resampled_data_x = resampled_data_x[random_s]

    _scaler, _inv_C, _columns = fit_MTS(resampled_data_x, resampled_data_y)

    y_train_pred = predict_MD(resampled_data_x, _scaler, _inv_C, _columns)

    _threshold = determine_threshold(resampled_data_y, y_train_pred)

    return _scaler, _inv_C, _columns, _threshold

from multiprocessing import Pool, cpu_count

def main():
    start = time.time()

    n_experiment = 1
    data_list = [
        # 'yeast', 
        # 'wine', 
        # 'abalone', 
        # 'car',
        # 'cancer', 
        'letter'
        ]
    print('MTSBag開始')

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

            job_args = list(zip([X_train] * n_estimators, [y_train] * n_estimators, [SIZE] * n_estimators))

            #========計算処理========
            p = Pool(processes=cpu_count()-1)
            ret = p.map(wrap_multi_thread_run, job_args)
            sms_multi = [r for r in ret]
            
                
            # y_proba, y_pred = predict_MTSBag_(X_test, result_scaler, result_inv_C, select_columns, threshold, n_estimators)

            # result_df = make_result_df(result_df, y_test, y_pred, y_proba, m)
    end = time.time()
    delta = end - start
    print('処理時間:{}s'.format(round(delta,3))) 

# import concurrent.futures
# import math

# PRIMES = [
#     112272535095293,
#     112582705942171,
#     112272535095293,
#     115280095190773,
#     115797848077099,
#     1099726899285419]

# def is_prime(n):
#     if n % 2 == 0:
#         return False

#     sqrt_n = int(math.floor(math.sqrt(n)))
#     for i in range(3, sqrt_n + 1, 2):
#         if n % i == 0:
#             return False
#     return True

# def main():
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
#             print('%d is prime: %s' % (number, prime))

if __name__ == '__main__':
    main()
#これがないと動かない！！！


# print('MTSBag開始')
# start = time.time()

# n_experiment = 1
# data_list = [
#         # 'yeast', 
#         # 'wine', 
#         # 'abalone', 
#         # 'car',
#         # 'cancer', 
#         'letter'
#         ]
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

#         # K:再標本化の回数 SIZE:再標本化されたもののサンプルサイズ
#         K = n_estimators
#         SIZE = int(len(X) * max_samples)

#         # 予測に必要なパラメータ
#         select_columns = [0] * K
#         result_scaler = [0] * K
#         result_inv_C = [0] * K
#         threshold = [0] * K

#         for i in range(K):
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
            
#         y_proba, y_pred = predict_MTSBag_(X_test, result_scaler, result_inv_C, select_columns, threshold, K)

#         result_df = make_result_df(result_df, y_test, y_pred, y_proba, m)

# end = time.time()
# delta = end - start
# print('処理時間:{}s'.format(round(delta,3))) 