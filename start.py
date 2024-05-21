# make_model.pyを開始するためのプログラム
import make_Model
import datetime
import Features
import glob
import logging
import start_features_and_model
import Objective
import sys
import GeneralModel

import paramerter

window_size_list = paramerter.window_size_list_original
overlap_list = paramerter.overlap_list_original
features = paramerter.features

def body_feature(objective_flag,delete_data,model,subject_num,task_names):
    plot_flag = 0
    learn_flag=0
    
    nasatlx_list = []
    mental_list = []
    object_choice_list = ["nasa","mean","tired","repeat","concentration","satisfied"]#"nasa",
    
    for object_choice in object_choice_list:
        if subject_num == "0":
            continue
        elif object_choice == "nasa" and objective_flag == 1:
            print("**** object_choice : " + str(object_choice) + "****")
            nasatlx_list = Objective.GettingObjectVarival(
                subject_num, object_choice)
            if len(nasatlx_list) == 0:
                print("Error: Failed to get nasa data.")
                exit()
            else:
                print("Successfully got objective variables.")
        elif objective_flag == 2 and object_choice != "nasa":
            print("**** object_choice : mental-" + str(object_choice) + "****")
            mental_list = Objective.GettingObjectVarival(subject_num,object_choice)
            if len(mental_list) == 0:
                    print("Error: Failed to get mental data.")
                    exit()
            else:
                    print("Successfully got objective variables.")
        else:
            continue
        
        # 全てのタスク名のフォルダを指定する
        csv_paths = []
        task_name_list=[]
        for task_name in task_names:
            # processedかつdelete_partを含むcsvファイルのパスを取得
            # print(task_name)
            tmp=task_name+"_delete_"+delete_data
            for path in glob.glob(f"e:\\データ処理\\被験者{subject_num}\\kioku_{task_name}\\processed*{tmp}*.csv"):
                csv_paths.append(path)
                task_name_list.append(task_name)

        for feature_choice in range(1,2): #for feature_choice in range(1,6):
            feature_list = []
            if feature_choice == 1 or feature_choice == 2 or feature_choice == 3:
                feature_list = features
                if feature_choice == 1:
                    feature_name = "all"
                    # print(feature_name)
                elif feature_choice == 2:
                    feature_name = "acc"
                    # print(feature_name)
                elif feature_choice == 3:
                    feature_name = "gyro"
                    # print(feature_name)
            elif feature_choice == 4:
                # featuresの"corr"までを格納
                feature_list = features[0:11]
                feature_name = "time"
                # print(feature_name)
            elif feature_choice == 5:
                # featuresの"main"以降を格納
                feature_list = features[11:]
                feature_name = "freq"
                # print(feature_name)
            for window_size, overlap in zip(window_size_list, overlap_list):
                # print(objective_flag)
                make_Model.main(task_name_list,subject_num, feature_list, feature_name,window_size,overlap,plot_flag,model,learn_flag,delete_data,nasatlx_list,mental_list,csv_paths,objective_flag,object_choice)  
                
                
                
def go_to_model(objective_flag,delete_data,model,subject_num):  
    objective_choice_list = ["tired","repeat","concentration","satisfied"]#"mean","tired","repeat","concentration","satisfied"
    if objective_flag == 1:
        objective_choice_index = ""
    elif objective_flag == 2:
        objective_choice_index = 1
    # print("objective_choice_index: ", objective_choice_index)
    
    
    #### モデルへGO ####
    if objective_flag == 1:
        objective_choice = ""
        start_features_and_model.main(delete_data,model,subject_num,objective_flag,objective_choice)
    else:
        print(objective_choice_index)
        # mental全てをモデル構築
        if objective_choice_index == 1:
            print("enter all mental")
            for i in range(len(objective_choice_list)):
                objective_choice = objective_choice_list[i]
                print("**** mental_choice : " + str(objective_choice) + "****")
                start_features_and_model.main(delete_data,model,subject_num,objective_flag,objective_choice)
        # mental or nasatlxを選択してモデル構築
        else:
            # print("Type of objective_choice: ", type(objective_choice))
            # print(objective_choice_index)
            object_index=int(objective_choice_index)-2
            objective_choice = objective_choice_list[object_index]
            print("**** mental_choice : " + str(objective_choice ) + "****")
            start_features_and_model.main(delete_data,model,subject_num,objective_flag,objective_choice)
        #### start_model.pyからコピー ####
def body_model(objective_flag,delete_data,model,subject_num,all_objective_flag):
    if all_objective_flag == "1":
        # nasa-tlxで実行
        objective_flag = 1
        go_to_model(objective_flag,delete_data,model,subject_num)
        # mentalで実行
        objective_flag = 2
        go_to_model(objective_flag,delete_data,model,subject_num)
    else:
        go_to_model(objective_flag,delete_data,model,subject_num)
        
        
    
        
        
        
def body_start():
    # 全ての被験者データを処理するかどうか
    all_data_flag = input("全ての被験者データを処理しますか？(one=0/all=1/multi=2/feature only=3/personalized model making only=4/all model making only=5/general model=6):")
    # feature_choice = input("特徴量リストを選択してください(1=all, 2=加速度領域, 3=角速度領域, 4=時間領域, 5=周波数領域): ")
    # plotを行うか行わないかを判定するフラグ
    # plot_flag = int(input("plotしますか？(0=no/1=yes):"))
    # plot_flag = 0
    # inputが1ならnasatlx_list、2ならmental_listを目的変数に設定
    objective_flag = int(input("目的変数を何に設定しますか？(1=nasatlx/2=mental):"))
    # objective_choice_list = ["mean","tired","repeat","concentration","satisfied"]#"mean","tired","repeat","concentration","satisfied"
    model = int(input("モデルを選択してください(1=RandomForest/2=MRM/3=SVR/4=GBM):"))
    # learn_flag=0
    # # learn_flag = int(input("特徴量計算/モデル構築を行いますか？(0=特徴量計算/1=モデル構築):"))
    # features=["mean","std","max","min","range","var","median","skew","kurt","mean_abs","mean_crossing","main","peak","sd","peak_psd","energy","entropy"]
    # feature_name = ""
    # featuresの日本語名を取得
    # features_japanese = ["平均","標準偏差","最大値","最小値","レンジ","分散","中央値","歪度","尖度","平均絶対偏差","平均交差点","メイン周波数","ピーク周波数","ピーク値","ピーク周波数の標準偏差","エネルギー","エントロピー"]


    # 削除データの指定
    delete_list=["nonedata","corner-like","corner-like_others"]
    # 削除データの指定
    delete_data = input("削除した部位を指定してください(1=削除なし/2=corner-likeのみ/3=corner-likeとothers_not_deleteとothers_delete):")
    if delete_data == "1":
        delete_data = delete_list[0]
    elif delete_data == "2":
        delete_data = delete_list[1]
    elif delete_data == "3":
        delete_data = delete_list[2]

    # ファイル名の冒頭がprocessedのすべてのファイルを指定する
    task_names=["none_1","none_2","none_3","none_4","none_5","low_1","low_2","low_3","low_4","low_5","high_1","high_3","high_4","high_5"]  #""none_1","none_2","none_3","none_4","none_5","low_1","low_2","low_3","low_4","low_5","high_1","high_3","high_4","high_5"
    
    main(all_data_flag,objective_flag,delete_data,model,*task_names)


def main(all_data_flag,objective_flag,delete_data,model,*task_names):
    objective_flag = int(objective_flag)
    # ロガーの設定
    # logging.basicConfig(filename='app.log('+str(now)+')', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    if all_data_flag == "1":
            for subject_num in range(1, 13):
                subject_num = str(subject_num)
                print("#### 被験者番号: "+subject_num+" ####")
                body_feature(objective_flag,delete_data,model,subject_num,task_names)
                all_objective_flag = "0"
                body_model(objective_flag,delete_data,model,subject_num,all_objective_flag)
    elif all_data_flag == "0":
        # 特定の被験者データを処理する
        subject_num = input("被験者番号: ")
        body_feature(objective_flag,delete_data,model,subject_num,task_names)
        all_objective_flag = "0"
        body_model(objective_flag,delete_data,model,subject_num,all_objective_flag)
        
    elif all_data_flag == "2":
        # 特定の被験者データを処理する
        subject_num = input("被験者番号(例:1,2,3): ")
        subject_num_list = subject_num.split(",")
        for subject_num in subject_num_list:   
            print("#### 被験者番号: "+subject_num+" ####") 
            body_feature(objective_flag,delete_data,model,subject_num,task_names)
            all_objective_flag = "0"
            body_model(objective_flag,delete_data,model,subject_num,all_objective_flag)


    # 特徴量計算のみ
    elif all_data_flag == "3":
        # 特徴量のみを計算
        print("start feature calculation")
        all_features_flag = input("全ての特徴量を計算しますか？(0=no/1=yes): ")
        if all_features_flag == "1":
            for subject_num in range(1, 13):
                subject_num = str(subject_num)
                print("###被験者番号: "+subject_num+"###")
                body_feature(objective_flag,delete_data,model,subject_num,task_names)
        else:
            subject_num = input("被験者番号: ")
            subject_num_list = subject_num.split(",")
            for subject_num in subject_num_list:   
                print("#### 被験者番号: "+subject_num+" ####") 
                body_feature(objective_flag,delete_data,model,subject_num,task_names)
    
    # モデル構築のみ  
    elif all_data_flag == "4":
        all_objective_flag = input("全ての目的変数に対してモデル構築しますか？(0=no/1=all): ")
        print("start personalized model")
        for subject_num in range(1, 13):
            subject_num = str(subject_num)
            print("#### 被験者番号: "+subject_num+" ####")
            body_model(objective_flag,delete_data,model,subject_num,all_objective_flag)
    elif all_data_flag == "5":
        all_objective_flag = input("全ての目的変数に対してモデル構築しますか？(0=no/1=all): ")
        print("start all model")
        subject_num = "0"
        print("#### 被験者番号: "+subject_num+" ####")
        body_model(objective_flag,delete_data,model,subject_num,all_objective_flag)
    elif all_data_flag == "6": #start general model
        # print("start general model")
        GeneralModel.main()
        
        
        
        

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "main":
            # print(sys.argv)
            main(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], *sys.argv[6:])
    else:
        body_start()