import pandas as pd
import numpy as np

import math
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

select_data = str(input('どのデータ？'))

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

    import tensorflow as tf

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

else:
    print('そのデータはありません')

# バギング側の話
# ブートストラップサンプリングの個数
n = 10
seed = random.randint(0, n)

# 使用する7つの変数をランダムに取得する
# バギングをする際はそれぞれのサブサンプルで7つの変数を選択する．
random.seed(seed)
random_s = random.sample(list(X.columns), 7)
X = X[random_s]

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
    MD = []
    for i in range(len(Z)):
        _a = np.dot(Z[i], inv_C)
        _MD = np.dot(_a, Z[i].T)
        _MD = _MD / Z.shape[1]
        MD.append(_MD)
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
    df_l8 = pd.DataFrame([[1,1,1,1,1,1,1],[1,1,1,2,2,2,2],[1,2,2,1,1,2,2],[1,2,2,2,2,1,1],[2,1,2,1,2,1,2],[2,1,2,2,1,2,1],[2,2,1,1,2,2,1],[2,2,1,2,1,1,2]])
    l8 = (df_l8==1).values

    #異常データのマハラノビス距離
    result = np.zeros((l8.shape[0], anomaly_Z.shape[0]))
    for x, l8_row in enumerate(l8):
        result[x] = cal_MD(anomaly_Z[:, l8_row], inv_C[l8_row][:,l8_row])

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

def show_result(y_test, y_pred):
    df_pred = pd.DataFrame(y_test)
    df_pred['pred'] = y_pred
    df_pred = df_pred.sort_values('pred').reset_index(drop=True)


    min_gini = np.inf
    threshold = 0
    for i in range(len(df_pred)):
        
        neg = df_pred.iloc[:i+1]
        pos = df_pred.iloc[i:]

        p_neg = sum(neg[y_test.name]) / len(neg)
        gini_neg = 1 - ( p_neg ** 2 + ( 1 - p_neg ) ** 2 )

        p_pos = sum(pos[y_test.name]) / len(pos)
        gini_pos = 1 - ( p_pos ** 2 + ( 1 - p_pos ) ** 2 )

        gini_split = (len(neg) / len(df_pred) * gini_neg) + (len(pos) / len(df_pred) * gini_pos)

        if min_gini > gini_split:
            min_gini = gini_split
            threshold = df_pred.iloc[i]['pred']
            threshold_idx = i
        print(i, gini_split, df_pred.iloc[i]['pred'])

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Best paramater')
    print(threshold_idx, min_gini, threshold)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    print('AUC : ', roc_auc_score(y_test.values, y_pred))

    recall = df_pred.iloc[threshold_idx + 1:][y_test.name].sum() / df_pred[y_test.name].sum()
    print('recall : ', recall)

    precision = df_pred.iloc[threshold_idx + 1:][y_test.name].mean()
    print('precision :', precision)

    g_mean = np.sqrt(recall * precision)
    print('g_mean : ', g_mean)

    RS = recall / precision
    print('RS : ', RS)

result_scaler, result_inv_C, select_columns = fit_MTS(X_train, y_train)

if result_scaler != 0:
    y_pred = predict_MTS(X_test, result_scaler, result_inv_C, select_columns)
else:
    y_pred = X_test[select_columns]
    
show_result(y_test, y_pred)


