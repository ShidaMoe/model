# 特徴量計算とデータ整形(目的変数の追加)

import pandas as pd
import numpy as np
import os
import csv
import glob
import math
from sklearn.preprocessing import StandardScaler
import logging
from scipy.signal import welch

import paramerter
ALL_MODEL_NUM = paramerter.ALL_MODEL_NUM


# データの種類(acc_x,acc_y...)ごとに6つのcsvを作成
def CalcurateFeatures(csv_path, window_size, overlap, subject_number,task_name,features_list,data_type,feature_naming,objective_flag,delete_data,mental_choice):
    # csvファイルの読み込み
    csv_paths = []
    csv_path_original = csv_path.replace("processed","features")
    csv_path_original = csv_path_original.replace("\\kioku_"+task_name+"\\","\\kioku_"+task_name+"\\features\\")
    objective_name = None
    if objective_flag == 1:
        objective_name = "nasatlx"
    elif objective_flag == 2:
        objective_name = "mental"
        mental_name = objective_name + "-" + str(mental_choice)
    # print(objective_flag)
    # print(objective_name)

    # logging.info(csv_path_original)
    print(csv_path)
    df = pd.read_csv(csv_path)
    # 1列目のtimestampのみを取得
    timestamps = df["timestamp"]
    # 1列目のtimestampを削除
    df = df.drop("timestamp",axis=1)

    if feature_naming == "acc" :
        #gyrの列を削除する
        # df = df.drop("gyrx",axis=1)
        # df = df.drop("gyry",axis=1)
        # df = df.drop("gyrz",axis=1)
        # ヘッダーに"acc"を含むデータフレームを保存するリスト
        dfs_with_acc = pd.DataFrame()
        headers = []
        if objective_flag == 1:
            csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_all_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+")_nasatlx.csv"
        elif objective_flag == 2:
            csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_all_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+")__"+mental_name+".csv"
        csv_accfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_acc_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+").csv"
        # csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_"+feature_naming+"_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+").csv"
        df = pd.read_csv(csv_allfeatures_path)
        # ヘッダーに"acc"または"target_data"を含む列のヘッダーを取得
        headers = [col for col in df.columns if "acc" in col or "target_data" in col]

        # ヘッダーに"acc"または"target_data"を含む列のデータを取得
        for header in headers:
            dfs_with_acc = pd.concat([dfs_with_acc, df[header]], axis=1)
        # headersをヘッダーに設定
        dfs_with_acc.columns = headers
        
        return csv_accfeatures_path, dfs_with_acc

    elif feature_naming == "gyro" :
        dfs_with_gyr = pd.DataFrame()
        headers = []
        if objective_flag == 1:
            csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_all_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+")_nasatlx.csv"
        elif objective_flag == 2:
            csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_all_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+")__"+mental_name+".csv"
        csv_gyrfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_gyro_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+").csv"
        # csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_"+feature_naming+"_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+").csv"
        df = pd.read_csv(csv_allfeatures_path)
        # ヘッダーに"acc"または"target_data"を含む列のヘッダーを取得
        headers = [col for col in df.columns if "gyr" in col or "target_data" in col]

        # ヘッダーに"acc"または"target_data"を含む列のデータを取得
        for header in headers:
            dfs_with_gyr = pd.concat([dfs_with_gyr, df[header]], axis=1)
        # headersをヘッダーに設定
        dfs_with_gyr.columns = headers
        
        return csv_gyrfeatures_path, dfs_with_gyr
    
    elif feature_naming == "time" :
        dfs_with_time = pd.DataFrame()
        headers = []
        if objective_flag == 1:
            csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_all_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+")_nasatlx.csv"
        elif objective_flag == 2:
            csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_all_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+")__"+mental_name+".csv"
        csv_timefeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_time_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+").csv"
        # csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_"+feature_naming+"_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+").csv"
        df = pd.read_csv(csv_allfeatures_path)
        # features_listの各要素を含むヘッダーを取得
        headers = [col for col in df.columns if any(feature in col for feature in features_list) or "target_data" in col]
        dfs_with_time = df[headers]
        # headersをヘッダーに設定
        dfs_with_time.columns = headers
        
        return csv_timefeatures_path, dfs_with_time
    elif feature_naming == "freq" :
        dfs_with_freq = pd.DataFrame()
        headers = []
        if objective_flag == 1:
            csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_all_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+")_nasatlx.csv"
        elif objective_flag == 2:
            csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_all_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+")__"+mental_name+".csv"
        csv_freqfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_freq_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+").csv"
        # csv_allfeatures_path="E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_"+feature_naming+"_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+").csv"
        df = pd.read_csv(csv_allfeatures_path)
        # features_listの各要素を含むヘッダーを取得
        headers = [col for col in df.columns if any(feature in col for feature in features_list) or "target_data" in col]
        dfs_with_freq = df[headers]
        # headersをヘッダーに設定
        dfs_with_freq.columns = headers
        
        return csv_freqfeatures_path, dfs_with_freq
    
    #all かつ mentalならば
    elif objective_flag == 2:
        csv_allfeatures_path = "E:\\データ処理\\モデル構築\\data\\被験者" + \
            str(subject_number)+"\\allfeatures_"+task_name+"_all_"+data_type+"_delete_" + \
            delete_data+"_window("+str(window_size)+")_nasatlx.csv"
        df = pd.read_csv(csv_allfeatures_path)
        df = df.drop('target_data', axis=1)
        
        return csv_allfeatures_path, df
    
        # # features_listの各要素を含むヘッダーを取得
        # headers = [col for col in df.columns if any(feature in col for feature in features_list) or "target_data" in col]
        # dfs_with_freq = df[headers]
        # # headersをヘッダーに設定
        # dfs_with_freq.columns = headers
        
        # return csv_freqfeatures_path, dfs_with_freq
        


    # 標準化
    scaler = StandardScaler()
    # Features.pyのCalcurateFeatures関数内
    # if not df.empty:
    #     scaler.fit(df)
    # else:
    #     print("Warning: No data to scale.")
    #     return None, None
    print(df)
    if df.empty:
        print("Warning: No data to scale.")
        return None, None
    scaler.fit(df)
    df_std = scaler.transform(df)
    df_std = pd.DataFrame(df_std, columns=df.columns)

    # 特徴量の算出
    # データ数
    N = window_size
    # サンプリング周期
    head_dt = 1/5
    other_dt = 1/50
    if data_type == "頭部":
        dt = head_dt
    else:
        dt = other_dt

    features = pd.DataFrame()
    all_features = pd.DataFrame()  # 全ての特徴量を格納するデータフレームを初期化
    all_part_features = pd.DataFrame()
    step_size = window_size - overlap
    count=0
    all_headers=[]
    all_windowed_data = pd.DataFrame()
    for column in df_std.columns:
        logging.info(column+" calcurating...")
        # print(column+" calcurating...")
        headers = []
        all_features = pd.DataFrame()
        for i in range(0, len(df_std) - window_size + 1, step_size):
            # ウィンドウ内の最初と最後のタイムスタンプを取得
            start_timestamp = timestamps.iloc[i]
            end_timestamp = timestamps.iloc[i + window_size - 1]

            # タイムスタンプの範囲がdt*window_sizeを超えているかチェック
            if end_timestamp - start_timestamp > dt * (window_size+1):
                # タイムスタンプの範囲がdt*window_sizeを超えている場合、インデックスを1ずらす
                i += 1
                # print("Error: Timestamp range is over dt*window_size.")
                continue
            windowed_timestamps = timestamps.iloc[i:i+window_size]
            windowed_data = pd.Series(windowed_timestamps)
            all_windowed_data = pd.concat([all_windowed_data, windowed_data], axis=0, ignore_index=True)

            # N個のデータを出力
            # print(df_std[column].iloc[i:i+N])
            # 周波数領域の計算
            fft_data = np.fft.fft(df_std[column].iloc[i:i+N])
            # FFT結果の複素数を絶対値に変換
            fft_data = np.abs(fft_data)
            # FFT結果の複素数を絶対値に変換し、正規化
            fft_data = fft_data / (window_size / 2)
            # ナイキスト定数のデータを除外
            fft_data = fft_data[range(int(window_size / 2))]
            # print(fft_data)
            # features_listのすべての特徴量について特徴量を算出
            feature = {}
            features = pd.DataFrame()
            windows = [df_std[column].iloc[i:i+N].values]
            
            for feature_name in features_list:
                if feature_name == "mean": # 平均
                    feature = {
                        column + "_mean": df_std[column].iloc[i:i+N].mean()
                    }
                elif feature_name == "std": # 標準偏差
                    feature = {
                        column + "_std": df_std[column].iloc[i:i+N].std()
                    }
                elif feature_name == "max": # 最大値
                    feature = {
                        column + "_max": df_std[column].iloc[i:i+N].max()
                    }
                elif feature_name == "min": # 最小値
                    feature = {
                        column + "_min": df_std[column].iloc[i:i+N].min()
                    }
                elif feature_name == "range": # 範囲
                    feature = {
                        column + "_range": df_std[column].iloc[i:i+N].max() - df_std[column].iloc[i:i+N].min()
                    }
                elif feature_name == "var": # 分散
                    feature = {
                        column + "_var": df_std[column].iloc[i:i+N].var()
                    }
                elif feature_name == "median": # 中央値
                    feature = {
                        column + "_median": df_std[column].iloc[i:i+N].median()
                    }
                elif feature_name == "skew": # 歪度
                    feature = {
                        column + "_skew": df_std[column].iloc[i:i+N].skew()
                    }
                elif feature_name == "kurt": # 尖度
                    feature = {
                        column + "_kurt": df_std[column].iloc[i:i+N].kurt()
                    }
                # # 中央絶対偏差
                # elif feature_name == "mad":
                #     logging.info(type(df_std[column]))
                #     logging.info(df_std[column])
                #     logging.info(pd.__version__)
                #     feature = {
                #         column + "_mad": df_std[column].mad()
                #     }
                # 平均絶対偏差
                elif feature_name == "mean_abs":
                    feature = {
                        column + "_mean_abs": df_std[column].iloc[i:i+N].abs().mean()
                    }
                # 平均交差数
                elif feature_name == "mean_crossing":
                    feature = {
                        column + "_mean_crossing": ((df_std[column].iloc[i:i+N] > df_std[column].iloc[i:i+N].mean()).astype(float).diff() != 0).sum()
                    }
                # 相関係数
                # elif feature_name == "corr":
                #     feature = {
                #         column + "_corr": df_std.corr()
                #     }
                # 周波数領域の特徴量
                # 主周波数
                elif feature_name == "main":
                    # ピーク周波数を算出
                    max_index = np.argmax(fft_data)
                    # ピーク周波数を格納
                    feature = {
                        column + "_main": max_index * dt
                    }
                # スペクトル密度
                elif feature_name == "sd":
                    # スペクトル密度を計算
                    # x(n)*exp(-2*pi*f*t)をN個分足したものを2乗して平均を取る
                    sd = np.mean(np.abs(np.sum(df_std[column].iloc[i:i+N].values * np.exp(-2 * np.pi * np.arange(N) * dt * np.arange(N)[:, np.newaxis]), axis=0)) ** 2)
                    # スペクトル密度を格納
                    feature = {
                        column + "_sd": sd
                    }
                    # # パワースペクトル密度
                    # # psd = (1 / (dt * N)) * (fft_data ** 2)
                    # # ウィンドウサイズごとにデータを分割
                    # windows = [df_std[column].iloc[i:i+N].values]
                    # # print(windows)
                    # # 各ウィンドウに対してパワースペクトル密度を計算
                    # psds = [welch(window, fs=1.0, nperseg=window_size)[1] for window in windows]
                    # # print(psds)
                    # # パワースペクトル密度を格納
                    # feature = {
                    #     column + "_psd": psds
                    # }
                # ピーク周波数のパワースペクトル密度
                elif feature_name == "peak_psd":
                    # 最大値を持つ周波数を算出
                    max_index = np.argmax(fft_data)*dt
                    peak_psd = (1 / (dt * N)) * (max_index ** 2)
                    # 最大値を持つ周波数のパワースペクトル密度を格納
                    feature = {
                        column + "_peak_psd": peak_psd
                    }
                # エネルギー
                elif feature_name == "energy":
                    # エネルギーを算出
                    energy = np.sum(fft_data ** 2)
                    # エネルギーを格納
                    feature = {
                        column + "_energy": energy
                    }
                # エントロピー
                elif feature_name == "entropy":
                    # エントロピーを算出
                    entropy = - np.sum(fft_data * np.log2(fft_data))
                    # エントロピーを格納
                    feature = {
                        column + "_entropy": entropy
                    }
                # ループ一回目の時は
                # print(feature)
                # エントロピー
                # elif feature_name == "entropy":
                #     # FFTデータを正規化して確率分布を作成
                #     prob_dist = fft_data / fft_data.sum()
                #     # エントロピーを算出
                #     entropy = -np.sum(prob_dist * np.log2(prob_dist + np.finfo(float).eps))
                #     # エントロピーを格納
                #     feature = {
                #         column + "_entropy": entropy
                #     }
                # 周波数バンドのエネルギー
                # elif feature_name == "energy_band":
                #     # バンドパスフィルタをかける
                #     # 0.5Hz以下をカット
                #     fft_data[0:2] = 0
                #     # 20Hz以上をカット
                #     fft_data[40:50] = 0
                #     # エネルギーを算出
                #     energy_band = np.sum(fft_data ** 2)
                #     # エネルギーを格納
                #     feature = {
                #         column + "_energy_band": energy_band
                #     }
                # for key, value in feature.items():
                #     logging.info(f"{key}: {np.array(value).shape}")
                    # accx_corrがfeature辞書に存在する場合、その値を1次元に変換
                # if "accx_corr" in feature:
                #     accx_corr = np.array(feature["accx_corr"])
                #     if accx_corr.shape == (6, 6):
                #         for i in range(6):
                #             for j in range(6):
                #                 feature[f"accx_corr_{i}_{j}"] = accx_corr[i][j]
                #         del feature["accx_corr"]
                # for key in list(feature.keys()):
                #     if isinstance(feature[key], np.ndarray):
                #         if feature[key].shape[0] != 1:
                #             for i in range(feature[key].shape[0]):
                #                 feature[f"{key}_{i}"] = feature[key][i]
                #             del feature[key]
                #     elif isinstance(feature[key], list):
                #         if len(feature[key]) != 1:
                #             for i in range(len(feature[key])):
                #                 feature[f"{key}_{i}"] = feature[key][i]
                #             del feature[key]
                
                # # feature = {k: [v] for k, v in feature.items()}
                # features = pd.concat([pd.DataFrame(f, index=[0])
                #                for f in feature], ignore_index=True)
                # # features = pd.DataFrame(feature, index=[0])
                # print(features)
                # 辞書のキーをheadersリストに追加
                # headersの長さがfeature_listの長さと一致したら何もしない
                if len(headers) != len(features_list):
                    headers.extend(feature.keys())
                
                df = pd.DataFrame(feature, index=[0])
                features = pd.concat([features, df], axis=1, ignore_index=True)
                

                # print(features)
            all_features = pd.concat(
                [all_features, features], axis=0, ignore_index=True)                
            # print(all_features)
            count+=1
            # if count == 4:
            #     exit()
        # all_features.columns = features.keys()
        all_headers.extend(headers)
        all_features.columns = headers
        # print(all_features)
        # all_features = pd.concat([all_features, features], axis=1)  # 列方向にデータを結合
        csv_path = csv_path_original.replace(".csv", "") +"_"+ column + ".csv"
        # print("Successfully calculated features of "+column)
        
        # csvファイルへの書き出し
        # logging.info(csv_path)
        all_features.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
        # 全ての特徴量を1つのcsvに出力
        all_part_features = pd.concat(
            [all_part_features, all_features], axis=1, ignore_index=True)
    
    # print(all_part_features)
    all_windowed_data.to_csv("E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\window_data\\"+data_type+"_"+feature_naming+"_"+delete_data+"_window("+str(window_size)+")_"+objective_name+".csv")
    all_part_features.columns = all_headers   
    allcsv_path = "E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_number)+"\\allfeatures_"+task_name+"_"+feature_naming+"_"+data_type+"_delete_"+delete_data+"_window("+str(window_size)+").csv"
    # logging.info("Successfully all features write to "+allcsv_path)
    # print("Successfully all features write to "+allcsv_path)
    return allcsv_path, all_part_features



def MakingOneCsv(csv_paths,subject_num,task_name_list,data_part,delete_data,window_size,obgective_variable,feature_naming,object_name):
    # csv_pathsの中身をすべて読み込んで1つのcsvファイルにまとめる
    df = pd.DataFrame()
    for csv_path in csv_paths:
        df_tmp = pd.read_csv(csv_path)
        # df_tmpの列数・行数を取得
        df = pd.concat([df, df_tmp], axis=0)
    # logging.info(df)
    # csvファイルを作成
    #pathの指定

    csv_path = "E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_num)+"\\alldata_"+data_part+"_"+feature_naming+"_"+delete_data+"_window("+str(window_size)+")_"+object_name+".csv"
    df.to_csv(csv_path, index=False)
    logging.info("Successfully made one csv file.")
    # print("Successfully made one csv file in "+csv_path)
    return csv_path

def Make_alldatacsv(subject_number, feature_name, part, obgective_variable, windowsize, delete_data, object_name):
    # すべてのsubjectの特徴量を1つのcsvにまとめる
    # csvファイルの読み込み

    # 空のデータフレームを作成します。
    all_data = pd.DataFrame()
    file=[]
    #  "E:\\データ処理\\モデル構築\\data\\被験者"+str(subject_num)+"\\alldata_"+data_part+"_"+feature_naming+"_"+delete_data+"_window("+str(window_size)+")_"+object_name+".csv"
    # 各subject_numについてループを回します。
    for subject_num in range(1, ALL_MODEL_NUM+1):
        # ファイル名が条件に一致するCSVファイルを探します。
        for file in glob.glob(f"E:\\データ処理\\モデル構築\\data\\被験者*{subject_num}*\\alldata*{part}*{feature_name}*{delete_data}*{str(windowsize)}*{object_name}*.csv"):
            # CSVファイルを読み込みます。
            df = pd.read_csv(file)
            # print(df)
            # 読み込んだデータをall_dataに追加します。
            all_data = pd.concat([all_data, df])
    all_data_path = "E:\\データ処理\\モデル構築\\data\\被験者0\\alldata_"+part+"_"+feature_name+"_"+delete_data+"_window("+str(windowsize)+")_"+object_name+".csv"

    # 最終的なデータフレームをCSVファイルに出力します。
    all_data.to_csv(all_data_path, index=False)
    # print(all_data)

    return all_data_path
        
    
