import pandas as pd
import numpy as np

import math
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.utils import resample

from tqdm import tqdm 

import tensorflow as tf

# パラメータ
n_estimators = 10
max_samples = 0.5
select_data = str(input('どのデータ？'))


# データ取得
if select_data == 'letter':
    # データの取得
    df = pd.read_csv('../data/letter_recognition.csv', header=None)

    # Aのみを判定するため，Aを0，A以外を1にした．
    # 少数派のAを正常，その他を異常データと定義
    df[0] = df[0].apply(lambda x: 0 if x == 'A' else 1)

    #Xとyを入力
    X = df[range(1,17)]
    y = df[0]

elif select_data == 'wine':
    
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

    # ファイルのダウンロード
    dataset_path = tf.keras.utils.get_file('wine.data', dataset_url)

    print(dataset_path)

    column_names = ['Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline' 
    ]

    raw_data = pd.read_csv(dataset_path, names=column_names)
    raw_data['y'] = raw_data.index
    raw_data = raw_data.reset_index(drop=True)

    raw_data['y'] = raw_data['y'].apply(lambda x: 0 if x == 3 else 1)

    X = raw_data.drop('y', axis=1)
    y = raw_data['y']

elif select_data == 'abalone':

    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

    # ファイルのダウンロード
    dataset_path = tf.keras.utils.get_file('abalone.data', dataset_url)

    print(dataset_path)

    raw_data = pd.read_csv(dataset_path, names=range(8)).reset_index(drop=True)

    raw_data[7] = raw_data[7].apply(lambda x: 1 if x > 4 else 0)


    X = raw_data.drop(7, axis=1)
    y = raw_data[7]

else:
    print('そのデータはありません')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 必要な関数の定義

# 共分散行列の逆行列
def inv_cov(Z):
    #標準化後のベクトルを入力する
    #標準化した後なので相関行列と分散共分散行列は一致する
    c = np.cov(Z.T)
    return np.linalg.inv(c)

#マハラノビス汎距離
def cal_MD(Z, inv_C):
    '''
    Z:標準化したベクトル
    inv_C:標準化後の共分散行列
    '''
    MD = np.zeros(len(Z))
    for i in range(len(Z)):
        _a = np.dot(Z[i], inv_C)
        _MD = np.dot(_a, Z[i].T)
        _MD = _MD / Z.shape[1]
        MD[i] = _MD
    return MD

# MTSを実行
def fit_MTS(X, y):
    
    # 正常データのみを使用して標準化
    scaler = StandardScaler()
    scaler.fit(X[y == 0])
    normal_Z = scaler.transform(X[y == 0])
    anomaly_Z = scaler.transform(X[y == 1])

    # 正常データのみを使用して共分散行列を計算
    inv_C = inv_cov(normal_Z)

    # いったん飛ばす，削除の基準は？削除しない方法もあるっぽい？
        #１度目の仮のマハラノビス距離を計算
        # MD_1st = cal_MD(normal_Z, inv_C)
        # もしもマハラノビス距離が余りにも大きいサンプルがあれば任意で削除する
        # 削除後のデータを使用して標準化と共分散行列を計算

    # 異常データと直交表を用いてSN比を計算
    #L8直行表
    l8 = np.array([
        [1,1,1,1,1,1,1],
        [1,1,1,2,2,2,2],
        [1,2,2,1,1,2,2],
        [1,2,2,2,2,1,1],
        [2,1,2,1,2,1,2],
        [2,1,2,2,1,2,1],
        [2,2,1,1,2,2,1],
        [2,2,1,2,1,1,2]
        ])
    l8 = (l8 == 1)

    #異常データのマハラノビス距離
    result = np.zeros((l8.shape[0], anomaly_Z.shape[0]))
    for i, l8_row in enumerate(l8):
        result[i] = cal_MD(anomaly_Z[:, l8_row], inv_C[l8_row][:,l8_row])

    #SN比
    sn = np.zeros(l8.shape[0])
    for idx, row in enumerate(result):
        sum_MD = 0
        for i in range(len(row)):
            sum_MD += 1 / row[i]
        sn[idx] = -10 * math.log10(sum_MD / len(row))
        
    # SN比を利用し，不要と思われる変数を削除する
    #変数選択
    df_sn = pd.DataFrame(index=X.columns, columns=['SN比','残す'])
    for i, clm in enumerate(X.columns):
        df_sn.loc[df_sn.index == clm, 'SN比'] = sum(sn[l8.T[i]]) - sum(sn[~l8.T[i]])
        df_sn.loc[df_sn.index == clm, '残す'] = sum(sn[l8.T[i]]) - sum(sn[~l8.T[i]]) > 0
    #使用した変数を保存
    select_columns = df_sn[df_sn['残す']].index
    
    if len(select_columns) > 1:
        # 選択変数でのスケーラーと共分散行列を計算
        result_scaler = StandardScaler()
        result_scaler.fit(X[select_columns][y == 0])
        result_Z = result_scaler.transform(X[select_columns][y == 0])
        result_inv_C = inv_cov(result_Z)
    else:
        select_columns = df_sn['SN比'].astype(float).idxmax()
        result_scaler = 0
        result_inv_C = 0

    # 単位空間のスケーラーと共分散行列と選択した変数を出力
    return result_scaler, result_inv_C, select_columns

# 新しいデータのマハラノビス距離を計算する
def predict_MTS(X, scaler, inv_C, select_columns):
    Z = scaler.transform(X[select_columns])
    MD = cal_MD(Z, inv_C)
    return MD

# 閾値をジニ係数が最小になるように決定する
def determine_threshold(y_true, y_pred):
    df_pred = pd.DataFrame(y_true)
    df_pred['pred'] = y_pred
    df_pred = df_pred.sort_values('pred').reset_index(drop=True)

    min_gini = np.inf
    threshold = 0
    for i in range(len(df_pred)):
        
        neg = df_pred.iloc[:i+1]
        pos = df_pred.iloc[i:]

        p_neg = sum(neg[y_true.name]) / len(neg)
        gini_neg = 1 - ( p_neg ** 2 + ( 1 - p_neg ) ** 2 )

        p_pos = sum(pos[y_true.name]) / len(pos)
        gini_pos = 1 - ( p_pos ** 2 + ( 1 - p_pos ) ** 2 )

        gini_split = (len(neg) / len(df_pred) * gini_neg) + (len(pos) / len(df_pred) * gini_pos)

        if min_gini > gini_split:
            min_gini = gini_split
            threshold = df_pred.iloc[i]['pred']
            threshold_idx = i

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Best paramater')
    print(threshold_idx, min_gini, threshold)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # print('AUC : ', roc_auc_score(y_true.values, y_pred))

    # recall = df_pred.iloc[threshold_idx + 1:][y_true.name].sum() / df_pred[y_true.name].sum()
    # print('recall : ', recall)

    # precision = df_pred.iloc[threshold_idx + 1:][y_true.name].mean()
    # print('precision :', precision)

    # g_mean = np.sqrt(recall * precision)
    # print('g_mean : ', g_mean)

    # RS = recall / precision
    # print('RS : ', RS)
    return threshold

def predict_MTSBag(X, scaler, inv_C, select_columns, threshold):
    result = np.ndarray((K, len(X_test)), dtype=bool)
    for i in range(K):
        if scaler[i] != 0:
            Z = scaler[i].transform(X[select_columns[i]])
            MD = cal_MD(Z, inv_C[i])
            result[i] = MD > threshold[i]
        else:
            result[i] = X[select_columns[i]] > threshold[i]
    return result.sum(axis=0) / K, result.sum(axis=0) > (K/2)



# 実行するところ

# K:再標本化の回数 SIZE:再標本化されたもののサンプルサイズ
K = n_estimators
SIZE = int(len(X) * max_samples)

# 予測に必要なパラメータ
select_columns = [0] * K
result_scaler = [0] * K
result_inv_C = [0] * K
threshold = [0] * K

for i in tqdm(range(K)):
    # bootstrap sampling
    resampled_data_x, resampled_data_y = resample(X_train, y_train, n_samples = SIZE)
    random_s = random.sample(list(resampled_data_x.columns), 7)
    resampled_data_x = resampled_data_x[random_s]

    result_scaler[i], result_inv_C[i], select_columns[i] = fit_MTS(resampled_data_x, resampled_data_y)

    if result_scaler[i] != 0:
        y_pred = predict_MTS(resampled_data_x, result_scaler[i], result_inv_C[i], select_columns[i])
    else:
        y_pred = resampled_data_x[select_columns[i]]

    threshold[i] = determine_threshold(resampled_data_y, y_pred)
    

y_proba, y_pred = predict_MTSBag(X_test, result_scaler, result_inv_C, select_columns, threshold)

print('AUC : ', roc_auc_score(y_test, y_proba))

from sklearn.metrics import confusion_matrix
print('confusion_matrix : ')
print(confusion_matrix(y_true=y_test, y_pred=y_pred))

from sklearn.metrics import accuracy_score
print('accuracy_score : ', accuracy_score(y_test, y_pred))

from sklearn.metrics import precision_score
print('precision_score : ', precision_score(y_test, y_pred))

from sklearn.metrics import recall_score
print('recall_score : ', recall_score(y_test, y_pred))

g_mean = np.sqrt(recall_score(y_test, y_pred) * precision_score(y_test, y_pred))
print('g_mean : ', g_mean)

RS = recall_score(y_test, y_pred) / precision_score(y_test, y_pred)
print('RS : ', RS)

