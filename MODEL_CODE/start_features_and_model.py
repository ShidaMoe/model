# モデル構築と特徴量計算を行うプログラム

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
from openpyxl import load_workbook


# make_model.pyを開始するためのプログラム
import make_Model
import datetime
import Features
import glob
import logging
import MachiLearning
import paramerter

data_parts=paramerter.data_parts
task_names=paramerter.task_names
window_size_list_original = paramerter.window_size_list_original
window_size_list_head= paramerter.window_size_list_head
feature_name_list = paramerter.feature_name_list


def body(objective_flag,objective_choice,delete_data,model,subject_num,feature_name_list,data_parts,window_size_list_original,window_size_list_head,task_names):
        k=0
        # DFの初期化
        df = pd.DataFrame()
        df_all_error = pd.DataFrame()
        df_all_split_error = pd.DataFrame()
        df_all_importance = pd.DataFrame()
        error_df = pd.DataFrame()
        # 特定の被験者データを処理する
        # subject_num = input("被験者番号: ")
        # objective_flag = int(input("目的変数(1=nasatlx/2=mental):"))
        if objective_flag == 1:
                obgective_variable = "nasatlx"
                object_name = "nasatlx"
        elif objective_flag == 2:
                obgective_variable = "mental"
                object_name = "mental-" + objective_choice
        csv_paths = []
        task_name_list = []
        # 空のDataFrameを作成
        first_iteration = True
        i = 0
        df_dict = {}
        total_tasks = len(feature_name_list) * len(data_parts) * 3
        current_task = 0
        for feature_naming in feature_name_list:
            print(feature_naming)
            for data_part in data_parts:
                print(data_part)
                if data_part == "頭部":
                    window_size_list = window_size_list_head
                else:
                    window_size_list = window_size_list_original
                # print(window_size_list)
                for window_size in window_size_list:
                    print(window_size)
                    if subject_num == "0":
                        allfeature_csvpath = ""
                        if model == 1 or model == 4:
                            new_df,df_new_error,df_new_importance,first_iteration,i,model_name = MachiLearning.ModelBuild(
                                allfeature_csvpath, model, subject_num, feature_naming, data_part, obgective_variable, window_size,delete_data,object_name,first_iteration,i,df_dict)

                            df_all_importance = pd.concat([df_all_importance,df_new_importance], axis=1,ignore_index=True)
                        else:
                            new_df,df_new_error,first_iteration,i,model_name = MachiLearning.ModelBuild(
                                allfeature_csvpath, model, subject_num, feature_naming, data_part, obgective_variable, window_size,delete_data,object_name,first_iteration,i,df_dict)
                                #df_all_importance の初期化
                            df_all_importance = pd.DataFrame()
                            # dfを列方向に連結
                        df = pd.concat([df,new_df], axis=0)
                        df_all_error = pd.concat([df_all_error,df_new_error], axis= 1,ignore_index=True)
                        current_task += 1
                         
                    else:
                        # task_nameかつdelete_partかつdata_partを含むcsvファイルのパスを取得し、data_partごとに1つのcsvファイルにまとめる
                        for task_name in task_names:
                            tmp="_delete_"+delete_data
                            for path in glob.glob(f"E:\\データ処理\\モデル構築\\data\\被験者{subject_num}\\allfeatures*{task_name}*{feature_naming}*{data_part}*{tmp}*{str(window_size)}*{object_name}*.csv"):
                                # data_partのしゅるいごとに分けてcsv_pathsに格納
                                csv_paths.append(path)
                                task_name_list.append(task_name)       
                            # リストが空ならエラーを出力して終了
                        if len(csv_paths) == 0:
                            print("Error: Failed to get csv data.")
                            logging.info("Error: Failed to get csv data.")
                        else:
                            # print(csv_paths)
                                # すべてのcsvファイルを読み込んで1つのファイルにまとめる
                            # print(data_part)
                            # print(window_size)
                            allfeature_csvpath = Features.MakingOneCsv(csv_paths,subject_num,task_name_list,data_part,delete_data,window_size,obgective_variable,feature_naming,object_name)
                            # csv_pathsの初期化
                            csv_paths = []
                            task_name_list = []
                            # print("start model")
                            if model == 1 or model == 4:
                                new_df,df_new_error,df_new_importance,first_iteration,i,model_name = MachiLearning.ModelBuild(
                                allfeature_csvpath, model, subject_num, feature_naming, data_part, obgective_variable, window_size,delete_data,object_name,first_iteration,i,df_dict)
                                
                                df_all_importance = pd.concat([df_all_importance,df_new_importance], axis=1,ignore_index=True)
                            else:
                                new_df,df_new_error,first_iteration,i,model_name = MachiLearning.ModelBuild(
                                allfeature_csvpath, model, subject_num, feature_naming, data_part, obgective_variable, window_size,delete_data,object_name,first_iteration,i,df_dict)
                                #df_all_importance の初期化
                                df_all_importance = pd.DataFrame()
                            # dfを列方向に連結
                            df = pd.concat([df,new_df], axis=0)
                            df_all_error = pd.concat([df_all_error,df_new_error], axis= 1,ignore_index=True)
                            # print(error_df)
                            # df_all_split_error = pd.concat([df_all_split_error, error_df], axis=1, ignore_index=True)
                            # print(df_all_split_error)
                            # タスクが完了したので、現在のタスク数を増やす
                            current_task += 1

                            # ループの進行度を出力
                            print(f"Progress: {current_task}/{total_tasks} tasks completed.")
                    
        import datetime
        now = datetime.datetime.now()
        # 時間を変換
        now = now.strftime('%Y%m%d%H%M%S') 
        if subject_num == "0":
            # Excelファイルに保存
            filename = 'E:\\データ処理\\モデル構築\\model\\全員\\M1\\'+model_name+'\\被験者'+str(subject_num)+'-'+model_name+'-delete_'+delete_data+"-"+object_name+'-results('+now+').xlsx'
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                if len(df) > 1000000:
                    # DataFrameを分割してExcelに書き込む
                    chunk_size = 1000000  # 1 million rows per sheet
                    for i in range(0, len(df), chunk_size):
                        df_chunk = df[i:i+chunk_size]
                        df_chunk.to_excel(writer, index=False, sheet_name=f'data_{i // chunk_size + 1}')
                else:
                    df.to_excel(writer, index=False, sheet_name='data')

                df_all_error.to_excel(writer, sheet_name='error')
                df_all_importance.to_excel(writer, sheet_name='importance')
        else:
            # Excelファイルに保存
            filename = 'E:\\データ処理\\モデル構築\\model\\個人特化\\M1\\'+model_name+'\\被験者'+str(subject_num)+'-'+model_name+'-delete_'+delete_data+"-"+object_name+'-results('+now+').xlsx'#被験者'+str(subject_num)+'
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='data')
                df_all_error.to_excel(writer, sheet_name='error')
                # df_all_split_error.to_excel(writer, index=False,sheet_name='all_error')
                df_all_importance.to_excel(writer, sheet_name='importance')
            return df,df_all_error,df_all_importance
        print("Successfully made Excel file.")

def main(delete_data,model,subject_num,objective_flag,objective_choice):

    # 削除データの指定
    delete_list=["nonedata","corner-like","corner-like_others"]
    # delete_data = input("削除した部位を指定してください(1=削除なし/2=corner-likeのみ/3=corner-likeとothers_not_deleteとothers_delete(純粋な直視のみ)):")
    # model = int(input("モデルを選択してください(1=RandomForest/2=SVM/3=KNeighbors/4=NaiveBayes):"))
    # if delete_data == "1":
    #     delete_data = delete_list[0]
    # elif delete_data == "2":
    #     delete_data = delete_list[1]
    # elif delete_data == "3":
    #     delete_data = delete_list[2]
    # data_parts=["頭部","手首","柄"]#"頭部","手首","柄"
    # task_names=["none_1","none_2","none_3","none_4","none_5","low_1","low_2","low_3","low_4","low_5","high_1","high_2","high_3","high_4","high_5"] #,"none_2","none_3","none_4","none_5","low_1","low_2","low_3","low_4","low_5","high_1","high_2","high_3","high_4","high_5"
    # window_size_list_original = [60,120,240]#6,12,24,60,120,240
    # window_size_list_head= [6,12,24]
    # feature_name_list = ["all","acc","gyro","time","freq"]#"all","acc","gyro","time","freq"

    body(objective_flag,objective_choice,delete_data,model,subject_num,feature_name_list,data_parts,window_size_list_original,window_size_list_head,task_names)
            


       