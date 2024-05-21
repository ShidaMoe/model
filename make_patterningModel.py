import pandas as pd

def label_data(df, window_size, threshold):
    # window_sizeごとにデータを区切る
    windows = [df[i:i+window_size] for i in range(0, df.shape[0], window_size)]
    
    labeled_data = pd.DataFrame()
    
    for window in windows:
        # window内のデータの平均を計算
        mean = window.mean()
        
        # 平均が閾値を超えているかどうかでラベルを付ける
        label = 1 if mean > threshold else 0
        
        # ラベルを付けたデータを保存
        window['label'] = label
        labeled_data = labeled_data.append(window, ignore_index=True)
    
    return labeled_data

# この後の手順は？
# 1. ラベルづけしたデータをCSVファイルに保存
# 2. ラベルづけしたデータを使ってモデルを作成
# 3. モデルを使ってラベルづけしたデータを予測
# 4. 予測したデータをCSVファイルに保存