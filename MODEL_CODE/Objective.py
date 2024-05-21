import glob
import logging
import os
import sys
import numpy as np
import pandas as pd
import math



def GettingObjectVarival(subject_number,obgective_choice):
    # ファイル名の冒頭がnasatlxのすべてのファイルを指定する
    csv_nasatlx_paths = glob.glob("e:\\データ処理\\被験者"+subject_number+"\\nasatlx*.csv")
    print(csv_nasatlx_paths)
    print("***")

    # ファイル名のアンダーバーの後ろのハイフンで区切られた3単語をtermsに格納
    terms = []
    for csv_nasatlx_path in csv_nasatlx_paths:
        filename_with_ext = os.path.basename(csv_nasatlx_path)  # ファイルパスからファイル名を取得
        filename, _ = os.path.splitext(filename_with_ext)  # ファイル名と拡張子を分割
        after_underscore = filename.split("_")[2]  # アンダーバーの後ろの文字を取得
        terms.append(after_underscore.split("-"))  # ハイフンで区切られた文字をリストとしてtermsに追加
    # logging.info(terms)
    # terms　を1次元配列に変換
    terms = sum(terms,[])
    # logging.info(terms)
    # terms_listにtemsの各要素を5個ずつterms[0]_1,terms2[0]_2,...,terms[0]_5,terms[1]_1,terms2[1]_2,...,terms[1]_5,...を格納
    terms_list = []
    for term in terms:
        for i in range(1,6):
            terms_list.append(term+"_"+str(i))
    # logging.info(terms_list)
    
    if obgective_choice == "nasa":
        nasa_list = GettingNasatlx(terms_list,csv_nasatlx_paths)
        ob_list = nasa_list
    else:
        mental_list = GettingMental(subject_number,obgective_choice,terms_list)
        ob_list = mental_list
    
    return ob_list

# 目的変数の取得
def GettingNasatlx(terms_list,csv_nasatlx_paths):
    # nasatlx_listとmental_listにそれぞれtems_listの要素数分の要素を追加
    nasatlx_list = [[] for _ in range(len(terms_list))]  # 2次元リストとして初期化
    for i in range(len(terms_list)):
        nasatlx_list[i].append(terms_list[i])  # terms_list[i]を二つ目の要素として追加
    # logging.info(nasatlx_list)
    # logging.info(mental_list)

    # ファイルの7列目を2行目から取得してnasatlx_list[i]に格納
    for csv_nasatlx_path in csv_nasatlx_paths:
        df = pd.read_csv(csv_nasatlx_path, encoding='iso-8859-1')
        for i in range(len(terms_list)):
            if 7 < len(df.columns):  # dfの列数を超えないことを確認
                nasatlx_list[i].append(float(df.iloc[i, 7]))  # 8列目の2行目の値を取得し、数値に変換して追加
    print(nasatlx_list)

    return nasatlx_list

def GettingMental(subject_number,mental_choice,terms_list):
    # ****精神疲労度の取得****
    # ファイル名の冒頭がmentalのすべてのファイルを指定する
    csv_mental_paths = glob.glob(
        "e:\\データ処理\\被験者"+subject_number+"\\mental\\*.csv")
    
    # print(csv_mental_paths)
    mental_list = []
    for i, csv_mental_path in enumerate(csv_mental_paths):
        filename = os.path.basename(csv_mental_path)  # ファイル名を取得
        df = pd.read_csv(csv_mental_path, header=None)  # ファイルを読み込む
        for term in terms_list:
            if term in filename:  # ファイル名にterms_listの要素が含まれているかチェック
                if len(df) > 0:  # dfが1行以上持っていることを確認
                    data_list = df.values.tolist()[0]
                    # 平均値を計算
                    mean_value = sum(data_list) / len(data_list)
                    if mental_choice == "mean":
                        selected_value = mean_value
                    elif mental_choice == "tired":
                        selected_value = df.iloc[0, 0]
                    elif mental_choice == "repeat":
                        selected_value = df.iloc[0, 8]
                    elif mental_choice == "concentration":
                        selected_value = df.iloc[0, 14]
                    elif mental_choice == "satisfied":
                        selected_value = df.iloc[0, 18]
                         
                    # 1, 9, 15, 19列目の要素を足す
                    # selected_sum = df.iloc[0, [0, 8, 14, 18]].sum()
                    # selected_sum += mean_value  # 平均値を足す
                    # selected_sum = selected_sum/5
                    # 1つ目の要素をterm、2つ目の要素をselected_sumに設定
                    print(f"term: {term}, mentalchoice:{mental_choice},selected_value: {selected_value}")
                    mental_list.append([term, selected_value])
                    break
                else:
                    print(f"Warning: DataFrame is empty. File: {csv_mental_path}")
    return mental_list

    

# 目的変数の追加
def AddingObjectiveVariables(allcsv_path, all_features,objective_list,task_name,subject_number,object_judge_name,feature_name,mental_choice):
    object_name = object_judge_name + "-" + str(mental_choice)
    # print(task_name)
    if feature_name == "all" :
        if object_judge_name == "nasatlx":
            for i in range(len(objective_list)):
                if objective_list[i][0] == task_name:
                    if not math.isnan(objective_list[i][1]):  # 最終列にヘッダーをtarget_dataとして追加
                        # 最終列にヘッダーをtarget_dataとして追加
                        new_column = pd.Series([objective_list[i][1]]* len(all_features), name="target_data")
                        all_features = pd.concat([all_features, new_column], axis=1)
                        # all_features.insert(len(all_features.columns), "target_data", objective_list[i][1])
                        break
            allcsv_path = allcsv_path.replace(
                ".csv", "") + "_" + object_judge_name + ".csv"
        elif object_judge_name == "mental":
            for i in range(len(objective_list)):
                if objective_list[i][0] == task_name:
                    if not math.isnan(objective_list[i][1]):
                        # 最終列にヘッダーをtarget_dataとして追加
                        # print(objective_list[i][1])
                        all_features['target_data'] = [objective_list[i][1]]* len(all_features)
                        break
            allcsv_path = allcsv_path.replace(
                "nasatlx.csv", "") + "_" + object_name + ".csv"
    else:
        if object_judge_name == "nasatlx":
            allcsv_path = allcsv_path.replace(
            ".csv", "") + "_" + object_judge_name + ".csv"
        else:
            allcsv_path = allcsv_path.replace(
            ".csv", "") + "_" + object_name + ".csv"
    # logging.info(all_features)
    # csvファイルへの書き出し
    # ファイル名の最後にobject_nameを追加
    
    all_features.to_csv(allcsv_path, index=False)
    return