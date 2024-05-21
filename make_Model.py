# 機械学習のモデルを作成する
# 手順
# 1. データの読み込み
# 2. データの描画
# 3. ウィンドウサイズの設定
# 4. 特徴量の算出・csvファイルへの作成
# 5. モデルの作成

# 
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import Features
import Outlier
import MachiLearning
import logging
import Objective

# # グラフの描画
# def ShowGraph(csv_path):
#     # csvファイルの読み込み
#     df = pd.read_csv(csv_path)
#     # logging.info(df)
#     # accとgyroに分けて描画
#     df_acc = df.iloc[:, 1:4]
#     df_gyro = df.iloc[:, 4:7]
#     # df_accとdf_gyroを横軸timestampにして描画
#     #横軸をtimestampに設定
#     df_acc = df_acc.set_index("timestamp")
#     df_gyro = df_gyro.set_index("timestamp")

#     # グラフを表示
#     plt.show()

# main関数
def main(task_name_list,subject_num, features_list, feature_name,window_size_original,overlap_original,plot_flag,model,learn_flag,delete_part,nasatlx_list,mental_list,csv_paths,objective_flag,mental_choice):
    # データの読み込み
    subject_number = subject_num
    # outlier_flag = int(input("外れ値処置を行いますか？(0=no/1=yes):"))
    outlier_flag = 0
    learn_flag = 0
    if learn_flag == 0:
        # for task_name in task_names:
        #     tmp="_delete_"+delete_data
        #     for path in glob.glob(f"E:\\データ処理\\モデル構築\\data\\被験者{subject_num}\\allfeatures*{task_name}*{feature_naming}*{data_part}*{tmp}*{str(window_size)}*{object_name}*.csv"):
        #         # data_partのしゅるいごとに分けてcsv_pathsに格納
        #         csv_paths.append(path)
        #         task_name_list.append(task_name) 
        for i in range(len(csv_paths)):
            # csv_pathのprocessed_の後の文字列の内、次のアンダー-バーまでを取得
            data_part = csv_paths[i].split("processed_")[1].split("_")[0]
            # print("data_part : " + data_part)
            if data_part != "頭部":
                window_size = window_size_original*10
                overlap=overlap_original*10
            else:
                window_size = window_size_original
                overlap=overlap_original
            # print("---- window_size:"+str(window_size),"feature_name:"+str(feature_name)+" ----")
            # print(csv_paths[i])
            # if plot_flag == 0:
            #     # plotを行う
            #     ShowGraph(csv_paths[i])
            #     # inputが1なら特徴量の算出
            #     stop_flag = int(input("特徴量を算出しますか？(0=yes/1=no):"))
            #     if stop_flag == 1:
            #         # 次のデータへ
            #         continue
            allcsv_path, all_features = Features.CalcurateFeatures(csv_paths[i],window_size,overlap,subject_number,task_name_list[i],features_list,data_part,feature_name,objective_flag,delete_part,mental_choice)
            if all_features is None or all_features.empty:
                print(task_name_list[i] + " is empty.")
                continue
            # print("Successfully calculated features.")
            # 外れ値処置を行うかどうかを判定するフラグ
            if outlier_flag == 1:
                # 外れ値処置を行う
                Outlier.Processingoutliers()
            if objective_flag == 1:
                object_name="nasatlx"
                mental_choice = ""
                # nasatlx_listをobjective_listに設定
                Objective.AddingObjectiveVariables(allcsv_path, all_features,nasatlx_list,task_name_list[i],subject_number, object_name,feature_name,mental_choice)
            elif objective_flag == 2:
                # mental_listをobjective_listに設定
                object_name="mental"
                Objective.AddingObjectiveVariables(allcsv_path, all_features,mental_list,task_name_list[i],subject_number,object_name,feature_name,mental_choice)
            #AddingObjectiveVariables(Objective_list[i])
            logging.info("Successfully added objective variables.")
            # print("Successfully added objective variables.")
    elif learn_flag == 1:
        logging.info("making model...")
        # モデルの作成
        part_list = ["頭部","柄","手首"]
        for part in part_list:
            #partを含むcsvファイルのパスを取得
            csv_path = csv_paths[i]
            logging.info(f"Making model for {part}...")
            MachiLearning.ModelBuild(csv_path, model, subject_number, feature_name,part,objective_flag,window_size)

