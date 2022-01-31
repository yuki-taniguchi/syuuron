import time
start = time.time()
print('ライブラリインポート中')

import pandas as pd
import numpy as np

import math
import random
from tqdm import tqdm
from sklearn.utils import resample

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf

print('実験開始')

n_experiment = 10
data_list = [
    # 'yeast', 
    # 'wine', 
    # 'abalone', 
    # 'car',
    'cancer', 
    # 'letter'
    ]

def data_load(select_data):
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

        # print(dataset_path)

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

        # print(dataset_path)

        raw_data = pd.read_csv(dataset_path, names=range(8)).reset_index(drop=True)

        raw_data[7] = raw_data[7].apply(lambda x: 1 if x > 4 else 0)


        X = raw_data.drop(7, axis=1)
        y = raw_data[7]
        
    elif select_data == 'car':
        dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

        # ファイルのダウンロード
        dataset_path = tf.keras.utils.get_file('car.data', dataset_url)

        # print(dataset_path)

        raw_data = pd.read_csv(dataset_path, names=range(7))
        # .reset_index(drop=True)

        trans_dict1 = {'vhigh':3, 'high':2, 'med':1, 'low':0}
        trans_dict2 = {'big':2, 'med':1, 'small':0}
        trans_dict3 = {'high':2, 'med':1, 'low':0}


        raw_data[0] = raw_data[0].apply(lambda x: trans_dict1[x])
        raw_data[1] = raw_data[1].apply(lambda x: trans_dict1[x])
        raw_data[2] = raw_data[2].apply(lambda x: 5 if x == '5more' else int(x))
        raw_data[3] = raw_data[3].apply(lambda x: 6 if x == 'more' else int(x))
        raw_data[4] = raw_data[4].apply(lambda x: trans_dict2[x])
        raw_data[5] = raw_data[5].apply(lambda x: trans_dict3[x])
        raw_data[6] = raw_data[6].apply(lambda x: 0 if x == 'good' else 1)
        raw_data[7] = np.random.randint(0, 10, len(raw_data))


        X = raw_data.drop(6, axis=1)
        y = raw_data[6]

    elif select_data == 'yeast':
        raw_data = pd.read_csv('../data/yeast.csv', names=range(9)).reset_index(drop=True)

        raw_data[8] = raw_data[8].apply(lambda x: 0 if x == 'ME2' else 1)

        X = raw_data.drop(8, axis=1)
        y = raw_data[8]

    elif select_data == 'cancer':
        dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

        # ファイルのダウンロード
        dataset_path = tf.keras.utils.get_file('breast-cancer-wisconsin.data', dataset_url)

        # print(dataset_path)

        raw_data = pd.read_csv(dataset_path,
        names=range(10)
        ).reset_index(drop=True)

        raw_data[5] = raw_data[5].apply(lambda x: 0 if x == '?' else int(x))
        raw_data[9] = raw_data[9].apply(lambda x: 0 if x == 4 else 1)


        X = raw_data.drop(9, axis=1)
        y = raw_data[9]

    else:
        print('そのデータはありません')
    
    return X, y


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
def predict_MD(X, scaler, inv_C, select_columns):
    if type(select_columns) != np.int64:
        Z = scaler.transform(X[select_columns])
        MD = cal_MD(Z, inv_C)
    # 変数が一つしか選択されなかった場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
    else:
        MD = ((X[select_columns] - scaler) / inv_C) ** 2
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

def predict_MTSBag(X, scaler, inv_C, select_columns, threshold):
    result = np.ndarray((K, len(X_test)), dtype=bool)
    for i in range(K):
        MD = predict_MD(X, scaler[i], inv_C[i], select_columns[i])
        result[i] = MD > threshold[i]
    # 個々の計算方法を変えれば出力を異常度にできそう！！！（1/31）
    return result.sum(axis=0) / K, result.sum(axis=0) > (K/2)

def make_result_df(result_df, y_test, y_pred, y_proba):
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


print('MTS開始！')

for data in data_list:
    print(data)
    result_df = pd.DataFrame(
                        columns=['AUC', 'accuracy', 'recall', 'Specificity', 'precision', 'gmeans', 'RS'],
                        index=range(n_experiment))
    for m in tqdm(range(n_experiment)):
        
        X, y = data_load(data)

        # バギング側の話
        # ブートストラップサンプリングの個数
        n = 10
        seed = random.randint(0, n)

        # 使用する7つの変数をランダムに取得する
        # バギングをする際はそれぞれのサブサンプルで7つの変数を選択する．
        random.seed(seed)
        random_s = random.sample(list(X.columns), len(X.columns) if len(X.columns) < 7 else 7)
        X = X[random_s]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        result_scaler, result_inv_C, select_columns = fit_MTS(X_train, y_train)

        y_train_pred = predict_MD(X_train, result_scaler, result_inv_C, select_columns)

        threshold = determine_threshold(y_train, y_train_pred)

        y_proba = predict_MD(X_test, result_scaler, result_inv_C, select_columns)
        y_pred = y_proba > threshold

        result_df = make_result_df(result_df, y_test, y_pred, y_proba)
        
    result_df.to_csv(f'../data/output/MTS_{data}_result.csv')

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

        # K:再標本化の回数 SIZE:再標本化されたもののサンプルサイズ
        K = n_estimators
        SIZE = int(len(X) * max_samples)

        # 予測に必要なパラメータ
        select_columns = [0] * K
        result_scaler = [0] * K
        result_inv_C = [0] * K
        threshold = [0] * K

        for i in range(K):
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
            
        y_proba, y_pred = predict_MTSBag(X_test, result_scaler, result_inv_C, select_columns, threshold)

        result_df = make_result_df(result_df, y_test, y_pred, y_proba)
        
    result_df.to_csv(f'../data/output/MTSBag_{data}_result.csv')


from imblearn.over_sampling import SMOTE

print('SMOTEMTSBag開始！')

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

        # 実行するところ

        # K:再標本化の回数 SIZE:再標本化されたもののサンプルサイズ
        K = n_estimators
        SIZE = int(len(X) * max_samples)

        # 予測に必要なパラメータ
        select_columns = [0] * K
        result_scaler = [0] * K
        result_inv_C = [0] * K
        threshold = [0] * K

        # SMOTEを実行
        sampler = SMOTE()
        SMOTE_X, SMOTE_y = sampler.fit_resample(X=X_train, y=y_train)
        for i in range(K):
            # bootstrap sampling
            resampled_data_x, resampled_data_y = resample(SMOTE_X, SMOTE_y, n_samples = SIZE)
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

        y_proba, y_pred = predict_MTSBag(X_test, result_scaler, result_inv_C, select_columns, threshold)

        result_df = make_result_df(result_df, y_test, y_pred, y_proba)
        
    result_df.to_csv(f'../data/output/SMOTEMTSBag_{data}_result.csv')

process_time = time.time() - start

print('実行時間は', process_time)