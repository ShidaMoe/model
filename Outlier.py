# 外れ値処理を行う
import pandas as pd

def Processingoutliers(method, csv_path):
    # csvファイルの読み込み
    df = pd.read_csv(csv_path)
    # methodが1ならスミルノフ・グラブズ法を行う
    if method == 1:
        pass
    # スミルノフ・グラブズ法
    elif method == 2:
        # 四分位範囲法
        pass
    elif method == 3:
        # ロバスト回帰
        pass
    return
