import math

import pandas as pd
import numpy as np
import random
from sklearn.utils import resample


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# 必要な関数の定義

# 共分散行列の逆行列
def inv_cov(Z):
    #標準化後のベクトルを入力する
    #標準化した後なので相関行列と分散共分散行列は一致する
    c = np.cov(Z.T)
    return np.linalg.pinv(c)

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
    # 選択された変数が一つ以下の場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
    else:
        select_columns = df_sn['SN比'].astype(float).idxmax()
        result_scaler = X[select_columns][y == 0].mean()
        result_inv_C = X[select_columns][y == 0].std()

    # 単位空間のスケーラーと共分散行列と選択した変数を出力
    return result_scaler, result_inv_C, select_columns

# 新しいデータのマハラノビス距離を計算する
def predict_MD(X, result_scaler, result_inv_C, select_columns):
    # select_columnsがfloatになることがある？
    if type(result_scaler) == StandardScaler:
        Z = result_scaler.transform(X[select_columns])
        MD = cal_MD(Z, result_inv_C)
    # 変数が一つしか選択されなかった場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
    else:
        MD = ((X[select_columns] - result_scaler) / result_inv_C) ** 2
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

    return threshold

def predict_MTSBag(X, result_scaler, result_inv_C, select_columns, threshold, K):
    result = np.ndarray((K, len(X)), dtype=bool)
    for i in range(K):
        MD = predict_MD(X, result_scaler[i], result_inv_C[i], select_columns[i])
        result[i] = MD > threshold[i]
    # 個々の計算方法を変えれば出力を異常度にできそう！！！（1/31）
    return result.sum(axis=0) / K, result.sum(axis=0) > (K/2)

def predict_MTSBag_(X, result_scaler, result_inv_C, select_columns, threshold, K):
    predict = np.ndarray((K, len(X)), dtype=bool)
    proba = np.ndarray((K, len(X)), dtype=float)
    for i in range(K):
        MD = predict_MD(X, result_scaler[i], result_inv_C[i], select_columns[i])
        predict[i] = MD > threshold[i]
        proba[i] = MD
    # 各分類器のMDの平均をpredict_probaとして保存（2/28）
    # 各分類器で閾値を決めてそれらで投票したものを分類の答えとしている
    return proba.mean(axis=0), predict.sum(axis=0) > (K/2)

def make_result_df(result_df, y_test, y_pred, y_proba, m):
    # for文の中で回す用
    cm = confusion_matrix(y_test, y_pred)
    TP, FN, FP, TN = cm.flatten()
    result_df['AUC'][m] = roc_auc_score(y_test, y_proba)
    result_df['accuracy'][m] = (TP + TN ) / (TP + FP + TN + FN)
    result_df['recall'][m] = TP / (TP + FN)
    result_df['precision'][m] = TP / (TP + FP)
    result_df['Specificity'][m] = TN / (TN + FP)
    result_df['gmeans'][m] = np.sqrt((TN / (TN + FP)) * (TP / (TP + FN)))
    result_df['RS'][m] = (TP / (TP + FN)) / (TN / (TN + FP))
    return result_df


