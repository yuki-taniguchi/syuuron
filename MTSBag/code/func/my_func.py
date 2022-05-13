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
    inv_C:標準化後の共分散行列の逆行列
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

# MTSを実行し，変数選択をするのではなく，効果ゲインによって変数重みづけを行う
def fit_WMTS(X, y):
    """
    input: X, y
    output: reduced_model_scaler, reduced_model_inv_C, select_columns, select_columns_weight

    reduced_model_scaler: 縮小モデルのスケーラー
    reduced_model_inv_C: 縮小モデルの共分散行列の逆行列
    select_columns: 選択された変数
    select_columns_weight: 選択された変数の重み

    """
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

    # 異常データのマハラノビス距離
    anomaly_MD = np.zeros((l8.shape[0], anomaly_Z.shape[0]))
    for i, l8_row in enumerate(l8):
        anomaly_MD[i] = cal_MD(anomaly_Z[:, l8_row], inv_C[l8_row][:,l8_row]) # 正常データのinv_Cを使う必要がある

    # SN比の算出
    sn = np.zeros(l8.shape[0])
    for idx, row in enumerate(anomaly_MD):
        sum_MD = 0
        for row_i in row:
            sum_MD += 1 / row_i
        sn[idx] = -10 * math.log10(sum_MD / len(row))
        
    # SN比を利用し，不要と思われる変数を削除する
    # 変数選択
    df_gain = pd.DataFrame(index=X.columns, columns=['効果ゲイン','残す'])
    for i, clm in enumerate(X.columns):
        gain = sum(sn[l8.T[i]]) - sum(sn[~l8.T[i]])
        df_gain.loc[df_gain.index == clm, '効果ゲイン'] = gain
        df_gain.loc[df_gain.index == clm, '残す'] = gain > 0
    # 選択された変数を保存
    select_columns = df_gain[df_gain['残す']].index
        
    # 選択された変数が1つ以下の場合の例外処理
    if len(select_columns) > 1:
        select_gain = df_gain[df_gain['残す']]['効果ゲイン'].values
        select_columns_weight = select_gain / select_gain.sum()
        # 縮小モデルでのスケーラーと共分散行列を計算
        reduced_model_scaler = StandardScaler()
        reduced_model_scaler.fit(X[select_columns][y == 0])
        reduced_model_normal_Z = reduced_model_scaler.transform(X[select_columns][y == 0])
        reduced_model_inv_C = inv_cov(reduced_model_normal_Z)
    # 選択された変数が一つ以下の場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
    else:
        select_columns = df_gain['効果ゲイン'].astype(float).idxmax()
        reduced_model_scaler = X[select_columns][y == 0].mean()
        reduced_model_inv_C = X[select_columns][y == 0].std()
        select_columns_weight = 1

    # 縮小モデルのスケーラーと共分散行列と選択した変数を出力
    return reduced_model_scaler, reduced_model_inv_C, select_columns, select_columns_weight


# 縮小モデルによってマハラノビス距離を計算する
def cal_WMD_by_reduced_model(X, reduced_model_scaler, reduced_model_inv_C, select_columns, select_columns_weight):
    # select_columnsがfloatになることがある？
    if type(reduced_model_scaler) == StandardScaler:
        Z = reduced_model_scaler.transform(X[select_columns])
        Weighted_Z = Z * select_columns_weight
        MD = cal_MD(Weighted_Z, reduced_model_inv_C)
    # 変数が一つしか選択されなかった場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
    else:
        MD = ((X[select_columns] - reduced_model_scaler) / reduced_model_inv_C) ** 2
    return MD

    
def predict_WMTS(X_test, reduced_model_scaler, reduced_model_inv_C, select_columns, select_columns_weight, threshold):
    proba = cal_WMD_by_reduced_model(X_test, reduced_model_scaler, reduced_model_inv_C, select_columns, select_columns_weight)
    pred = proba > threshold
    return proba, pred

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

# def predict_WeightedMD():
#     """新しいデータの重みづけマハラノビス距離を計算する"""
#     WMD = 0
#     return WMD

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

def predict_MTSBag(X, result_scaler, result_inv_C, select_columns, threshold, n_estimators):
    result = np.ndarray((n_estimators, len(X)), dtype=bool)
    for i in range(n_estimators):
        MD = predict_MD(X, result_scaler[i], result_inv_C[i], select_columns[i])
        result[i] = MD > threshold[i]
    # ここの計算方法を変えれば出力を異常度にできそう！！！（1/31）
    return result.sum(axis=0) / n_estimators, result.sum(axis=0) > (n_estimators/2)

def predict_MTSBag_ImpAgg(X, result_scaler, result_inv_C, select_columns, threshold, n_estimators):
    predict = np.ndarray((n_estimators, len(X)), dtype=bool)
    proba = np.ndarray((n_estimators, len(X)), dtype=float)
    for i in range(n_estimators):
        MD = predict_MD(X, result_scaler[i], result_inv_C[i], select_columns[i])
        predict[i] = MD > threshold[i]
        proba[i] = MD
    # 各分類器のMDの平均をpredict_probaとして保存（2/28）
    # 各分類器で閾値を決めてそれらで投票したものを分類の答えとしている
    return proba.mean(axis=0), predict.sum(axis=0) > (n_estimators/2)


def predict_WMTSBag_ImpAgg(X, result_scaler, result_inv_C, select_columns, select_columns_weight, threshold, n_estimators):
    predict = np.ndarray((n_estimators, len(X)), dtype=bool)
    proba = np.ndarray((n_estimators, len(X)), dtype=float)
    for i in range(n_estimators):
        MD = cal_WMD_by_reduced_model(X, result_scaler[i], result_inv_C[i], select_columns[i], select_columns_weight[i])
        predict[i] = MD > threshold[i]
        proba[i] = MD
    # 各分類器のMDの平均をpredict_probaとして保存（2/28）
    # 各分類器で閾値を決めてそれらで投票したものを分類の答えとしている
    return proba.mean(axis=0), predict.sum(axis=0) > (n_estimators/2)

####################################
# グラムシュミット直交化法を使ったMTS
####################################

#各説明変数ごとに標準化
def transform_standard(fit_X, transform_X):
    scaler = StandardScaler()
    scaler.fit(fit_X)
    return scaler.transform(transform_X)


# グラムシュミットの直交化
def fit_gram_schmidt(normal_Z):
    '''
    t: グラムシュミット係数 
    ips: normal_U の相関行列の対角成分
    を出力する関数

    入力: 
        normal_Z: 正常データの標準化したサンプルベクトル  
    '''
    normal_A = normal_Z.T
    # tの算出
    t = np.zeros((normal_A.shape[0], normal_A.shape[0]))
    normal_U = np.zeros(normal_A.shape)
    for l in range(normal_A.shape[0]):
        sigma = 0
        for q in range(l):
            t[l][q] = np.dot(normal_A[l], normal_U[q]) / (np.dot(normal_U[q], normal_U[q]))
            # normal_Uの二乗が0になってしまう問題
            sigma +=  t[l][q] * normal_U[q]
        normal_U[l] = normal_A[l] - sigma
    # ipsの算出
    ips = np.diag(np.cov(normal_U))

    return t, ips

def create_gram_vec_U(Z, t):
    '''
    U: gramschmidt特徴ベクトル 
    を出力する関数

    入力: 
        Z: 標準化したサンプルベクトル
        t; グラムシュミット係数
    '''
    A = Z.T
    U = np.zeros(A.shape)
    for l in range(A.shape[0]):
        sigma = 0 
        for q in range(l):
            sigma +=  t[l][q] * U[q]
        U[l] = A[l] - sigma
    return U

def gram_schmidt_cal_MD(U, ips, feature_weight=[1.0]*100):
    '''
    gramschmidt_MD を出力する関数

    入力:
        U.T: gramschmidtサンプルベクトル 
        ips: U の相関行列の対角成分 
        feature_weight: 変数の重み，デフォルトはすべて１
    '''
    
    k = U.shape[1]
    MD = np.zeros(U.shape[0])
    
    for i, one_U in enumerate(U):
        sigma_MD = 0
        for q, u in enumerate(one_U):
            sigma_MD += feature_weight[q] * (u**2 / ips[q])
            # ipsが0になってしまう問題
        sigma_MD = sigma_MD / k
        MD[i] = sigma_MD
    return MD


# MTGSを実行
def fit_MTGS(X, y):
    
    # 正常データのみを使用して標準化
    normal_Z = transform_standard(X[y == 0], X[y == 0])
    anomaly_Z = transform_standard(X[y == 0], X[y == 1])

    # 正常データのみを使用して t, ips を計算
    t, ips = fit_gram_schmidt(normal_Z)
    anomaly_U = create_gram_vec_U(anomaly_Z, t)

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
        result[i] = gram_schmidt_cal_MD(anomaly_U.T[:, l8_row], ips[l8_row])

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
        result_t, result_ips = fit_gram_schmidt(result_Z)
    # 選択された変数が一つ以下の場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
    else:
        select_columns = df_sn['SN比'].astype(float).idxmax()
        result_scaler = X[select_columns][y == 0].mean()
        result_ips = X[select_columns][y == 0].std()
        result_t = None

    # 単位空間のスケーラーと共分散行列と選択した変数を出力
    return result_scaler, result_t, result_ips, select_columns

# 新しいデータのgram_MDを計算する
def predict_gram_MD(X, result_scaler, result_t, result_ips, select_columns):
    # select_columnsがfloatになることがある？
    if type(result_scaler) == StandardScaler:
        Z = result_scaler.transform(X[select_columns])
        U = create_gram_vec_U(Z, result_t)
        MD = gram_schmidt_cal_MD(U.T, result_ips)
    # 変数が一つしか選択されなかった場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
    else:
        MD = ((X[select_columns] - result_scaler) / result_ips) ** 2
    return MD


# MTGSを実行し，変数選択をするのではなく，効果ゲインによって変数重みづけを行う
def fit_WMTGS(X, y):
    
    # 正常データのみを使用して標準化
    normal_Z = transform_standard(X[y == 0], X[y == 0])
    anomaly_Z = transform_standard(X[y == 0], X[y == 1])

    # 正常データのみを使用して t, ips を計算
    t, ips = fit_gram_schmidt(normal_Z)
    anomaly_U = create_gram_vec_U(anomaly_Z, t)

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

    # 異常データのマハラノビス距離
    anomaly_MD = np.zeros((l8.shape[0], anomaly_Z.shape[0]))
    for i, l8_row in enumerate(l8):
        anomaly_MD[i] = gram_schmidt_cal_MD(anomaly_U.T[:, l8_row], ips[l8_row]) # 正常データのinv_Cを使う必要がある

    # SN比の算出
    sn = np.zeros(l8.shape[0])
    for idx, row in enumerate(anomaly_MD):
        sum_MD = 0
        for row_i in row:
            sum_MD += 1 / row_i
        sn[idx] = -10 * math.log10(sum_MD / len(row))
        
    # SN比を利用し，不要と思われる変数を削除する
    # 変数選択
    df_gain = pd.DataFrame(index=X.columns, columns=['効果ゲイン','残す'])
    for i, clm in enumerate(X.columns):
        gain = sum(sn[l8.T[i]]) - sum(sn[~l8.T[i]])
        df_gain.loc[df_gain.index == clm, '効果ゲイン'] = gain
        df_gain.loc[df_gain.index == clm, '残す'] = gain > 0
    # 選択された変数を保存
    select_columns = df_gain[df_gain['残す']].index
        
    # 選択された変数が1つ以下の場合の例外処理
    if len(select_columns) > 1:
        select_gain = df_gain[df_gain['残す']]['効果ゲイン'].values
        select_columns_weight = select_gain / select_gain.sum()
        # 縮小モデルでのスケーラーと共分散行列を計算
        reduced_model_scaler = StandardScaler()
        reduced_model_scaler.fit(X[select_columns][y == 0])
        reduced_model_normal_Z = reduced_model_scaler.transform(X[select_columns][y == 0])
        reduced_model_t, reduced_model_ips = fit_gram_schmidt(reduced_model_normal_Z)
    # 選択された変数が一つ以下の場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
    else:
        select_columns = df_gain['効果ゲイン'].astype(float).idxmax()
        reduced_model_scaler = X[select_columns][y == 0].mean()
        reduced_model_ips = X[select_columns][y == 0].std()
        reduced_model_t = None
        select_columns_weight = 1

    # 縮小モデルのスケーラーと共分散行列と選択した変数を出力
    return reduced_model_scaler, reduced_model_t, reduced_model_ips, select_columns, select_columns_weight


# 縮小モデルによってマハラノビス距離を計算する
def cal_gram_WMD_by_reduced_model(X, reduced_model_scaler, reduced_model_t, reduced_model_ips, select_columns, select_columns_weight):
    # select_columnsがfloatになることがある？
    if type(reduced_model_scaler) == StandardScaler:
        Z = reduced_model_scaler.transform(X[select_columns])
        U = create_gram_vec_U(Z, reduced_model_t)
        MD = gram_schmidt_cal_MD(U.T, reduced_model_ips, select_columns_weight)
    # 変数が一つしか選択されなかった場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
    else:
        MD = ((X[select_columns] - reduced_model_scaler) / reduced_model_ips) ** 2
    return MD

    
def predict_WMTGS(X_test, reduced_model_scaler,  reduced_model_t, reduced_model_ips, select_columns, select_columns_weight, threshold):
    proba = cal_gram_WMD_by_reduced_model(X_test, reduced_model_scaler, reduced_model_t, reduced_model_ips, select_columns, select_columns_weight)
    pred = proba > threshold
    return proba, pred



def predict_MTGSBag_ImpAgg(X, result_scaler, result_t, result_ips, select_columns, threshold, n_estimators):
    predict = np.ndarray((n_estimators, len(X)), dtype=bool)
    proba = np.ndarray((n_estimators, len(X)), dtype=float)
    for i in range(n_estimators):
        MD = predict_gram_MD(X, result_scaler[i], result_t[i], result_ips[i], select_columns[i])
        predict[i] = MD > threshold[i]
        proba[i] = MD
    # 各分類器のMDの平均をpredict_probaとして保存（2/28）
    # 各分類器で閾値を決めてそれらで投票したものを分類の答えとしている
    return proba.mean(axis=0), predict.sum(axis=0) > (n_estimators/2)


def predict_WMTGSBag_ImpAgg(X, reduced_model_scaler, reduced_model_t, redused_model_ips, select_columns, select_columns_weight, threshold, n_estimators):
    predict = np.ndarray((n_estimators, len(X)), dtype=bool)
    proba = np.ndarray((n_estimators, len(X)), dtype=float)
    for i in range(n_estimators):
        MD = cal_gram_WMD_by_reduced_model(X, reduced_model_scaler[i], reduced_model_t[i], redused_model_ips[i], select_columns[i], select_columns_weight[i])
        predict[i] = MD > threshold[i]
        proba[i] = MD
    # 各分類器のMDの平均をpredict_probaとして保存（2/28）
    # 各分類器で閾値を決めてそれらで投票したものを分類の答えとしている
    return proba.mean(axis=0), predict.sum(axis=0) > (n_estimators/2)





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


