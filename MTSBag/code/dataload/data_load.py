import pandas as pd
import numpy as np
import tensorflow as tf

INPUT_DIR = '../data/'

def data_load(select_data):
    """
    引数に特定のデータの名前を入れるとそのデータをロードしてX,yを返す関数
    input :     'yeast', 
                'wine', 
                'abalone', 
                'car',
                'cancer', 
                'letter'
    return: X, y
    """

    if select_data == 'letter':
        # データの取得
        df = pd.read_csv(INPUT_DIR + 'letter_recognition.csv', header=None)

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
        
        raw_data = pd.read_csv(dataset_path, names=range(13))
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
        raw_data = pd.read_csv(INPUT_DIR + 'yeast.csv', names=range(9)).reset_index(drop=True)

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

    elif select_data == '1504':
        df = pd.read_csv(INPUT_DIR + '1504_player_df.csv')
        X = df.drop([
            'TargetDate',
            'y_bin',
            'y_num'
            ], axis=1)
        y = df['y_bin']
        
    else:
        print('そのデータはありません')
    
    return X, y