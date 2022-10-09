import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("../")
from func.my_func import inv_cov, cal_MD

class MTSClassifier():
    """
    MTS 分類器
    """

    def __init__(self, add_weight=False):

        self.add_weight = add_weight
        
        self.reduced_model_scaler = StandardScaler()

    def fit(self, X, y):
        # 正常データのみを使用して標準化
        scaler = StandardScaler()
        scaler.fit(X[y == 0])
        normal_Z = scaler.transform(X[y == 0])
        anomaly_Z = scaler.transform(X[y == 1])

        # 正常データのみを使用して共分散行列を計算
        self.inv_C = inv_cov(normal_Z)

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
            anomaly_MD[i] = cal_MD(anomaly_Z[:, l8_row], self.inv_C[l8_row][:,l8_row]) # 正常データのinv_Cを使う必要がある

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
        self.select_columns = df_gain[df_gain['残す']].index
        
        # 選択された変数が1つ以下の場合の例外処理
        if len(self.select_columns) > 1:
            select_gain = df_gain[df_gain['残す']]['効果ゲイン'].values
            self.select_columns_weight = select_gain / select_gain.sum()
            # 縮小モデルでのスケーラーと共分散行列を計算
            self.reduced_model_scaler.fit(X[self.select_columns][y == 0])
            self.reduced_model_normal_Z = self.reduced_model_scaler.transform(X[self.select_columns][y == 0])
            self.reduced_model_inv_C = inv_cov(self.reduced_model_normal_Z)
        # 選択された変数が一つ以下の場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
        else:
            self.select_columns = df_gain['効果ゲイン'].astype(float).idxmax()
            self.reduced_model_mean = X[self.select_columns][y == 0].mean()
            self.reduced_model_std = X[self.select_columns][y == 0].std()

    def predict_proba(self, X):
        # select_columnsがfloatになることがある？
        if len(self.select_columns) > 1:
            Z = self.reduced_model_scaler.transform(X[self.select_columns])
            if self.add_weight:
                Weighted_Z = Z * self.select_columns_weight
                MD = cal_MD(Weighted_Z, self.reduced_model_inv_C)
            else:
                MD = cal_MD(Z, self.reduced_model_inv_C)
        # 変数が一つしか選択されなかった場合はその変数を正常データの平均と標準偏差で標準化してそれの二乗を異常値とする
        else:
            MD = ((X[self.select_columns] - self.reduced_model_mean) / self.reduced_model_std) ** 2
        return MD

    def determine_threshold(self, X, y):
        proba = self.predict_proba(X)
        _df = pd.DataFrame(y)
        _df['proba'] = proba
        _df = _df.sort_values('proba').reset_index(drop=True)

        min_gini = np.inf
        self.threshold = 0
        for i in range(len(_df)):
            
            neg = _df.iloc[:i+1]
            pos = _df.iloc[i:]

            p_neg = sum(neg[y.name]) / len(neg)
            gini_neg = 1 - ( p_neg ** 2 + ( 1 - p_neg ) ** 2 )

            p_pos = sum(pos[y.name]) / len(pos)
            gini_pos = 1 - ( p_pos ** 2 + ( 1 - p_pos ) ** 2 )

            gini_split = (len(neg) / len(_df) * gini_neg) + (len(pos) / len(_df) * gini_pos)

            if min_gini > gini_split:
                min_gini = gini_split
                self.threshold = _df.iloc[i]['proba']

    def predict(self, X):
        """
        determine_thresholdを実行してから実行する
        """
        proba = self.predict_proba(X)
        
        return proba > self.threshold


