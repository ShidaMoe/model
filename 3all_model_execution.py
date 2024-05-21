import os
import threading
import subprocess
import datetime

# 現在のファイルのディレクトリを取得
current_dir = os.path.dirname(__file__)
# start.pyをインポートするために必要なパス操作
import sys
sys.path.append(os.path.dirname(current_dir))
# start.pyをインポート
import start

# print("start.py imported")

def run_general_model_main(path):
    # GeneralModel.mainを呼び出すためのラッパースクリプトを実行
    subprocess.run(["python", path], check=True)

def run_start_py_main_personalizedModel(start_py_path, all_data_flag, objective_flag, delete_data, model, task_names):
    args = ["python", start_py_path, "main","4", str(objective_flag), str(delete_data), str(model)] + list(task_names)
    try:
        subprocess.run(args, check=True)
        print("run_start_py_main_personalizedModel: 完了")
    except subprocess.CalledProcessError as e:
        print(f"run_start_py_main_personalizedModel: エラー {e}")
        
def run_start_py_main_allModel(start_py_path, all_data_flag, objective_flag, delete_data, model, task_names):
    # 引数を文字列に変換してリストに追加
    args = ["python", start_py_path, "main","5", str(objective_flag), str(delete_data), str(model)] + list(task_names)
    try:
        subprocess.run(args, check=True)
        print("run_start_py_main_allModel: 完了")
    except subprocess.CalledProcessError as e:
        print(f"run_start_py_main_allModel: エラー {e}")



# all_data_flag = "4"                               
# inputが1ならnasatlx_list、2ならmental_listを目的変数に設定
objective_flag = int(input("目的変数を何に設定しますか？(1=nasatlx/2=mental):"))
objective_choice_list = ["mean","tired","repeat","concentration","satisfied"]#"mean","tired","repeat","concentration","satisfied"
if objective_flag == 1:
    objective_choice_index = ""
elif objective_flag == 2:
    objective_choice_index = int(input(
        "目的変数の種類を選択してください(1=all/2=mean/3=tired/4=repeat/5=concentration/6=satisfied):"))
model = int(input("モデルを選択してください(1=RandomForest/2=SVM/3=KNeighbors/4=NaiveBayes):"))
learn_flag=0
features=["mean","std","max","min","range","var","median","skew","kurt","mean_abs","mean_crossing","main","peak","sd","peak_psd","energy","entropy"]
feature_name = ""

# 削除データの指定
delete_list=["nonedata","corner-like","corner-like_others"]
delete_data = input("削除した部位を指定してください(1=削除なし/2=corner-likeのみ/3=corner-likeとothers_not_deleteとothers_delete):")
if delete_data == "1":
    delete_data = delete_list[0]
elif delete_data == "2":
    delete_data = delete_list[1]
elif delete_data == "3":
    delete_data = delete_list[2]

# ファイル名の冒頭がprocessedのすべてのファイルを指定する
task_names=["none_1","none_2","none_3","none_4"]  #""none_1","none_2","none_3","none_4","none_5","low_1","low_2","low_3","low_4","low_5","high_1","high_3","high_4","high_5"
# 頭部の値でwindow_sizeのリストを作成
window_size_list = [6,12,24]#6,12,24
overlap_list = [3, 6,12]#3,6,12
# windoe_sizeとファイル名をメモ帳に保存
# hhmmssを取得
now = datetime.datetime.now().strftime("%H%M%S")
# windowsizeフォルダの下にpathを設定
path = "./windowsize/windowsize"+"("+str(now)+")/"



# GeneralModel.mainの実行用スレッド、相対パスを絶対パスに変換して渡す
general_model_path = os.path.join(current_dir, "GeneralModel.py")
thread1 = threading.Thread(target=run_general_model_main, args=(general_model_path,))

# start.pyの実行用スレッド、相対パスを絶対パスに変換して渡す
start_py_path = os.path.join(current_dir, "start.py")
thread2 = threading.Thread(target=run_start_py_main_personalizedModel, args=(start_py_path,"4",objective_flag,delete_data,model,task_names))

# start.pyの実行用スレッド、相対パスを絶対パスに変換して渡す
start_py_path = os.path.join(current_dir, "start.py")
# ここで引数を渡す
thread3 = threading.Thread(target=run_start_py_main_allModel, args=(start_py_path, "5",objective_flag,delete_data,model,task_names))

# 最初に特徴量計算
# start.main("3",objective_flag,delete_data,model,*task_names)
# スレッドの開始
thread1.start()
print("thread1 started")
thread2.start()
print("thread2 started")
thread3.start()
print("thread3 started")

# スレッドの終了を待つ
thread1.join()
print("thread1 joined")
thread2.join()  
print("thread2 joined")
thread3.join()
print("thread3 joined")