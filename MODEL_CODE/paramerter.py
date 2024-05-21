# パラメータを設定するファイル
#
# ============================ 変数宣言部 ============================== #

data_parts=["頭部","手首","柄"]
#"頭部","手首","柄"
task_names=["none_1","none_2","none_3","none_4","none_5","low_1","low_2","low_3","low_4","low_5","high_1","high_2","high_3","high_4","high_5"] 
#"none_1","none_2","none_3","none_4","none_5","low_1","low_2","low_3","low_4","low_5","high_1","high_2","high_3","high_4","high_5"

window_size_list_original = [60,120,240]
#1.2*50,2.4*50,4.8*50
#60,120,240
window_size_list_head= [6,12,24]
#1.2*5,2.4*5,4.8*5
#6,12,24
overlap_list_original = [30,60,120]
#0.6*50,1.2*50,2.4*50
#30,60,120
overlap_list_head = [3,6,12]
#0.6*5,1.2*5,2.4*5
#3,6,12

features=["mean","std","max","min","range","var","median","skew","kurt","mean_abs","mean_crossing","main","peak","sd","peak_psd","energy","entropy"]
feature_name_list = ["all","acc","gyro","time","freq"]
#"all","acc","gyro","time","freq"

object_name = ["nasatlx", "mental-tired", "mental-repeat", "mental-concentration", "mental-satisfied"] 
# "nasatlx", "mental-mean", "mental-tired", "mental-repeat", "mental-concentration", "mental-satisfied"

delete_data ="nonedata"

# ============================ 精度検証用定数 ============================== #
FOLD = 10  # 交差検証数 10

# ============================ 被験者人数設定 ============================== #
ALL_MODEL_NUM = 12
# 全員モデル 被験者数：12
GENERAL_MODEL_NUM = 12
# 汎用モデル 被験者数：12

