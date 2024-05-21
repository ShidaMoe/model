# モデル構築のみ行うプログラム

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import glob
import logging


# make_model.pyを開始するためのプログラム
import make_Model
import datetime
import Features
import MachiLearning
import start_features_and_model

# data_parts=["頭部","手首","柄"]
def start(subject_num, objective_flag, delete_data, model, objective_choice):
    df = pd.DataFrame()
    df_all_error = pd.DataFrame()
    df_all_importance = pd.DataFrame()
    if objective_flag == 1:
        objective_choice = ""
        df, df_all_error, df_all_importance = start_features_and_model.main(delete_data,model,subject_num,objective_flag,objective_choice)
    else:
        # mental全てをモデル構築
        if objective_choice == 1:
            for i in range(len(obgective_choice_list)):
                objective_choice = obgective_choice_list[i]
                print("**** mental_choice : " + str(objective_choice) + "****")
                df, df_all_error, df_all_importance = start_features_and_model.main(delete_data,model,subject_num,objective_flag,objective_choice)
        # mental or nasatlxを選択してモデル構築
        else:
            objective_choice = obgective_choice_list[objective_choice-2]
            df, df_all_error, df_all_importance = start_features_and_model.main(delete_data,model,subject_num,objective_flag,objective_choice)
    
    return df, df_all_error, df_all_importance

# # 削除データの指定
delete_list=["nonedata","corner-like","corner-like_others"]
delete_data = input("削除した部位を指定してください(1=削除なし/2=corner-likeのみ/3=corner-likeとothers_not_deleteとothers_delete(純粋な直視のみ)):")
model = int(input("モデルを選択してください(1=RandomForest/2=SVM/3=KNeighbors/4=NaiveBayes):"))
if delete_data == "1":
    delete_data = delete_list[0]
elif delete_data == "2":
    delete_data = delete_list[1]
elif delete_data == "3":
    delete_data = delete_list[2]

# task_names=["none_1","none_2","none_3","none_4","none_5","low_1","low_2","low_3","low_4","high_1","high_2","high_3","high_4.","high_5"] 
# window_size_list = [24] #6,12,24,60,120,240

# # 全ての被験者データを処理するかどうか  
all_data_flag = input("処理する被験者データは？(全員=1/1人=0/複数人=2): ")
objective_flag = int(input("目的変数(1=nasatlx/2=mental):"))
obgective_choice_list = ["satisfied"] #"mean","tired","repeat","concentration","satisfied"
objective_choice = int(input(
            "目的変数の種類を選択してください(0=nasa/1=all/2=mean/3=tired/4=repeat/5=concentration/6=satisfied):"))
#### start.pyにコピー ####
if all_data_flag == "1":
    df = pd.DataFrame()
    df_all_error = pd.DataFrame()
    df_all_importance = pd.DataFrame()
    all_df = pd.DataFrame()
    all_error_df = pd.DataFrame()
    all_importance_df = pd.DataFrame()
    for subject_num in range(1, 13):
        df, df_all_error, df_all_importance = start(subject_num, objective_flag, delete_data, model, objective_choice)
        all_df=pd.concat([all_df,df],axis=0)
        all_error_df=pd.concat([all_error_df,df_all_error],axis=1)
        all_importance_df=pd.concat([all_importance_df,df_all_importance],axis=1)
    import datetime
    now = datetime.datetime.now()
    # 時間を変換
    now = now.strftime('%Y%m%d%H%M%S') 
    # Excelファイルに保存
    filename = 'E:\\データ処理\\モデル構築\\model\\個人特化\\delete_'+delete_data+'-results('+now+').xlsx'
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        all_df.to_excel(writer, sheet_name='all', index=False)
        all_error_df.to_excel(writer, sheet_name='error', index=False)
        all_importance_df.to_excel(writer, sheet_name='importance', index=False)
elif all_data_flag == "0" :
    subject_num = input("被験者番号: ")
    start(subject_num, objective_flag, delete_data, model, objective_choice)
    # if objective_flag == 1:
    #     objective_choice = ""
    #     start_features_and_model.main(delete_data,model,subject_num,objective_flag,objective_choice)
    # else:
    #     objective_choice = int(input(
    #         "目的変数の種類を選択してください(1=all/2=mean/3=tired/4=repeat/5=concentration/6=satisfied):"))
    #     # mental全てをモデル構築
    #     if objective_choice == 1:
    #         for i in range(len(obgective_choice_list)):
    #             objective_choice = obgective_choice_list[i]
    #             print("**** mental_choice : " + str(objective_choice) + "****")
    #             start_features_and_model.main(delete_data,model,subject_num,objective_flag,objective_choice)
    #     # mental or nasatlxを選択してモデル構築
    #     else:
    #         objective_choice = obgective_choice_list[objective_choice-2]
    #         print("**** mental_choice : " + str(objective_choice) + "****")
    #         start_features_and_model.main(delete_data,model,subject_num,objective_flag,objective_choice)
#### start.pyにコピー ####
            


# if all_data_flag == "1":
#     for subject_num in range(1, 13):
#             print("*")
#             for feature_choice in range(1,6):
#                   print("*")
# elif all_data_flag == "0":
#     k=0
#     # DFの初期化
#     df = pd.DataFrame()
#     # 特定の被験者データを処理する
#     subject_num = input("被験者番号: ")
#     objective_flag = int(input("目的変数(1=nasatlx/2=mental):"))
#     if objective_flag == 1:
#         obgective_variable = "nasatlx"
#     elif objective_flag == 2:
#         obgective_variable = "mental"
#     csv_paths = []
#     task_name_list = []
#     feature_name_list = ["all","acc","gyro","time","freq"]
#     for feature_naming in feature_name_list:
#         for window_size in window_size_list:
#             for data_part in data_parts:
#                 if data_part != "頭部":
#                     window_size = window_size*10
#                 # task_nameかつdelete_partかつdata_partを含むcsvファイルのパスを取得し、data_partごとに1つのcsvファイルにまとめる
#                 for task_name in task_names:
#                     tmp="_delete_"+delete_data
#                     for path in glob.glob(f"E:\\データ処理\\モデル構築\\data\\被験者{subject_num}\\allfeatures*{task_name}*{feature_naming}*{data_part}*{tmp}*{str(window_size)}*.csv"):
#                         # data_partのしゅるいごとに分けてcsv_pathsに格納
#                         csv_paths.append(path)
#                         task_name_list.append(task_name)       
#                     # リストが空ならエラーを出力して終了
#                 if len(csv_paths) == 0:
#                     print("Error: Failed to get csv data.")
#                     logging.info("Error: Failed to get csv data.")
#                 else:
#                     print(csv_paths)
#                         # すべてのcsvファイルを読み込んで1つのファイルにまとめる
#                     allfeature_csvpath = Features.MakingOneCsv(csv_paths,subject_num,task_name_list,data_part,delete_data,window_size,obgective_variable,feature_naming)
#                     logging.info("Successfully made one csv file.")
#                     # csv_pathsの初期化
#                     csv_paths = []
#                     task_name_list = []
#                     print("start model")
#                     new_df = MachiLearning.ModelBuild(
#                         allfeature_csvpath, model, subject_num, feature_naming, data_part, obgective_variable, window_size,delete_data)
#                     # dfを列方向に連結
#                     df = pd.concat([df,new_df], axis=0)

#     import datetime
#     now = datetime.datetime.now()
#     # 時間を変換
#     now = now.strftime('%Y%m%d%H%M%S')                
#     # CSVファイルに保存
#     filename = 'E:\\データ処理\\モデル構築\\model\\被験者'+str(subject_num)+'\\'+delete_data+'results('+now+').xlsx'
#     df.to_excel(filename, index=False)

# elif all_data_flag == "2":
#     # 複数の被験者データを処理する
#     k=0
#     subject_num = input("被験者番号(例:1,2,3): ")
#     subject_num_list = subject_num.split(",")
#     for subject_num in subject_num_list:
#         objective_flag = int(input("目的変数(1=nasatlx/2=mental):"))
#     if objective_flag == 1:
#         obgective_variable = "nasatlx"
#     elif objective_flag == 2:
#         obgective_variable = "mental"
#     csv_paths = []
#     task_name_list = []
#     feature_name_list = ["all","acc","gyro","time","freq"]
#     for feature_naming in feature_name_list:
#         for window_size in window_size_list:
#             for data_part in data_parts:
#                 # task_nameかつdelete_partかつdata_partを含むcsvファイルのパスを取得し、data_partごとに1つのcsvファイルにまとめる
#                 for task_name in task_names:
#                     tmp=task_name+"_delete_"+delete_data
#                     for path in glob.glob(f"E:\\データ処理\\モデル構築\\data\\被験者{subject_num}\\allfeatures*{feature_naming}*{data_part}*{tmp}*{str(window_size)}*.csv"):
#                         # data_partのしゅるいごとに分けてcsv_pathsに格納
#                         csv_paths.append(path)
#                         task_name_list.append(task_name)       
#                     # リストが空ならエラーを出力して終了
#                 if len(csv_paths) == 0:
#                     print("Error: Failed to get csv data.")
#                     logging.info("Error: Failed to get csv data.")
#                     exit()
#                 else:
#                         # すべてのcsvファイルを読み込んで1つのファイルにまとめる
#                     allfeature_csvpath = Features.MakingOneCsv(csv_paths,subject_num,task_name_list,data_part,delete_data,window_size,obgective_variable,feature_naming)
#                     logging.info("Successfully made one csv file.")
#                     # csv_pathsの初期化
#                     csv_paths = []
#                     task_name_list = []
#                     print("start model")
#                     new_df = MachiLearning.ModelBuild(
#                         allfeature_csvpath, model, subject_num, feature_naming, data_part, obgective_variable, window_size)
#                     # dfを列方向に連結
#                     df = pd.concat([df,new_df], axis=0)


#         import datetime
#         now = datetime.datetime.now()
#         # 時間を変換
#         now = now.strftime('%Y%m%d%H%M%S')                
#         # CSVファイルに保存
#         filename = 'E:\\データ処理\\モデル構築\\model\\被験者'+str(subject_num)+'\\results('+now+')-'+obgective_variable+"-delete_"+delete_data+'.xlsx'
#         df.to_excel(filename, index=False)

            



        # print(csv_paths[1])

