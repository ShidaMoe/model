# 汎用モデルを作成するためのプログラム
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import glob
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import r2_score

import paramerter

feature_name_list = paramerter.feature_name_list
data_parts = paramerter.data_parts
task_name = paramerter.task_names
delete_data = paramerter.delete_data
window_size_list_original = paramerter.window_size_list_original 
window_size_list_head = paramerter.window_size_list_head 
object_name = paramerter.object_name

GENERAL_MODEL_NUM = paramerter.GENERAL_MODEL_NUM



# def General_MachineLearning(X_train, X_test, y_train, y_test, model, subject_number, feature_name, part, obgective_variable, windowsize, deleted_rows, delete_data, object_name,first_iteration,i,df_dict):



# 1人抜きデータ集約
def OneOut_alldatacsv(subject_number, feature_name, part, windowsize, delete_data, object_name):
    # すべてのsubjectの特徴量を1つのcsvにまとめる
    # csvファイルの読み込み

    # 空のデータフレームを作成します。
    OneOut_alldata = pd.DataFrame()
    One_data_path = []  # One_data_pathのデフォルト値を空のリストとして設定
    file=[]
    for subject_num in range(1,GENERAL_MODEL_NUM+1): 
        if int(subject_num) == int(subject_number):
            # 1人抜きのデータを読み込まないように
            One_data_path = glob.glob(f"E:\\データ処理\\モデル構築\\data\\被験者{subject_number}\\alldata*{part}*{feature_name}*{delete_data}*{str(windowsize)}*{object_name}*.csv")
        else:
            # subject_numberを含むパスを生成します。
            file_path = f"E:\\データ処理\\モデル構築\\data\\被験者{subject_num}\\alldata*{part}*{feature_name}*{delete_data}*{str(windowsize)}*{object_name}*.csv"
            # file.append(file_path)
            for file in glob.glob(file_path):
                df = pd.read_csv(file)
                OneOut_alldata = pd.concat([OneOut_alldata, df])
                # print(df)
    # print(delete_data)
    OneOut_alldata_path = "E:\\データ処理\\モデル構築\\data\\汎化\\alldata_"+part+"_"+feature_name+"_"+delete_data+"_window("+str(windowsize)+")_"+object_name+".csv"
    # 最終的なデータフレームをCSVファイルに出力します。
    OneOut_alldata.to_csv(OneOut_alldata_path, index=False)
    if not One_data_path:
        print("Error  One_data_path is empty:機械学習用のデータ保存Excelファイルの中身がない")
        print(OneOut_alldata_path)
    # print(all_data)

    return OneOut_alldata_path, One_data_path


def General_MachineLearning(csv_path, pullout_path, subject_number, feature_name, part, object_name, windowsize, deleted_rows, delete_data,first_iteration,i,df_dict,model):

    # X = np.random.rand(12, 5)  # 12人分の5つの特徴量を持つデータ
    # y = np.random.rand(12)  # 12人分の目標変数
    print("csv_path:",csv_path)
    df = pd.read_csv(csv_path)
    df = df.dropna() # NaNを含む行を削除
    # dfの行数を取得
    num_rows = df.shape[0]
    # target_dataから必要な行数分だけデータを取得
    if 'target_data' in df.columns:
        y_train = df['target_data'][:num_rows]  
    else:
        print("'target_data' column is not in the dataframe")
        print("csv_path:",csv_path)
        exit()
    # y_train = df['target_data']
    if object_name == "nasatlx":
        y_train = y_train*100
    df = df.drop('target_data', axis=1)
    X_train = df

    df_test = pd.read_csv(pullout_path[0])
    df_test = df_test.dropna() # NaNを含む行を削除
    X_test = df_test.drop('target_data', axis=1)
    y_test = df_test['target_data']
    if object_name == "nasatlx":
        y_test = y_test*100

    # loo = LeaveOneOut()
    mae_scores = []

    # model = RandomForestRegressor()
    # model.fit(X_train, y_train)

    # # 特徴量の重要度を取得
    # importances = model.feature_importances_
    # # 特徴量の名前と重要度をデータフレームにまとめる
    # importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
    # # 重要度の降順でソート
    # importance_df = importance_df.sort_values('importance', ascending=False)


    # y_pred = model.predict(X_test)
    
    
    
    
        # モデルの作成
    if model == "1":
        model_name="RandomForest"
        print('RF')
        regressor = RandomForestRegressor()
        regressor.fit(X_train, y_train)
        # 予測
        y_pred = regressor.predict(X_test)
        
        # # 特徴量の重要度の計算
        # importances = regressor.feature_importances_
        # # 特徴量名と重要度を対応付ける
        # feature_importances = list(zip(X_train.columns, importances))

        # feature_importances = sorted(
        #     feature_importances, key=lambda x: x[1], reverse=True)

        # # y_predとy_test_foldを蓄積
        # y_pred_all = np.append(y_pred_all, y_pred)
        # # pandas.Seriesからnumpy.arrayに変換
        # y_test_all = np.append(y_test_all, y_test_fold.values)

        df_importance = regressor.feature_importances_

    elif model == "2":
        model_name="MRM"
        print('MRM')  # 重回帰分析
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)  # 訓練データにフィットさせる
        y_pred = regressor.predict(X_test)  # テストデータに対する予測

        
    elif model == "3":
        print('SVR')
        model_name="SVR"
        # SVRモデルのインスタンスを作成
        regressor = SVR(kernel='rbf')
        # 訓練データにフィットさせる
        regressor.fit(X_train, y_train)
        # テストデータに対する予測
        y_pred = regressor.predict(X_test)
        # # y_predとy_test_foldを蓄積
        # y_pred_all = np.append(y_pred_all, y_pred)
        # # pandas.Seriesからnumpy.arrayに変換
        # y_test_all = np.append(y_test_all, y_test_fold.values)
    
    elif model == "4":
        print('LightGBM')
        model_name="LGBM"

        # LightGBMのパラメータ設定
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'random_seed': 0,
            'verbose': -1
            # 'metric': {'l2', 'l1'},
            # 'num_leaves': 50,  # num_leavesの値を増やす
            # 'min_data_in_leaf': 20,  # min_data_in_leafの値を減らす
            # 'learning_rate': 0.05,
            # 'feature_fraction': 0.9,
            # 'bagging_fraction': 0.8,
            # 'bagging_freq': 5,
            # 'verbose': 0,
            # 'max_depth': -1,  # max_depthを増やす（-1は制限なしを意味する）
        }

        # 訓練データとテストデータのセットを作成
        lgb_train = lgb.Dataset(X_train, y_train)

        # モデルの学習
        gbm = lgb.train(params,
                        lgb_train
                        # num_boost_round=20,
                        # valid_sets=lgb_eval,
                        # early_stopping_rounds=5
                        )

        # テストデータに対する予測
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        # # y_predとy_test_foldを蓄積
        # y_pred_all = np.append(y_pred_all, y_pred)
        # # pandas.Seriesからnumpy.arrayに変換
        # y_test_all = np.append(y_test_all, y_test_fold.values)
        df_importance= gbm.feature_importance()



    # # y_pred_allとy_test_allをNumPy配列に変換
    # y_pred_all = np.append(y_pred_all, y_pred)
    # y_test_all = np.append(y_test_all, y_test_fold.values)
    # fold += 1

    
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    average_mae = np.mean(mae_scores)
    
    mse = mean_squared_error(y_test, y_pred)
    average_mse = np.mean(mse)
    
    rmse = np.sqrt(mse)
    average_rmse = np.mean(rmse)
    
    r2 = r2_score(y_test, y_pred)
    average_r2 = np.mean(r2)
    
    if (y_test == 0).any():
        # y_test_allの中の0を非常に小さい値に置き換える
        y_test_all_replaced = y_test.copy()
        y_test_all_replaced[y_test_all_replaced == 0] = 1e-10
        # mapeを計算
        mape = np.mean(np.abs((y_test - y_pred) / y_test_all_replaced)) * 100
        
    else:
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    error = y_pred -  y_test
    # df0_flattenedの下にerrorを連結
    
    # print("MAE:", average_mae)
    if model == "2" or model == "3":
        df_importance = None    
    return average_mae,model_name,average_mse,average_rmse,average_r2,mape,error,df_importance
# main
def main():
        print("start GeneralModel")
        model = input("モデルを選択してください(1=RandomForest/2=MRM/3=SVR/4=LightGBM):")
        # feature_name_list = ["all","acc"]#"all","acc","gyro","time","freq"
        # data_parts = ["頭部"] # "頭部", "手首", "柄"
        # object_name = ["nasatlx", "mental-mean", "mental-tired", "mental-repeat", "mental-concentration", "mental-satisfied"] # "nasatlx", "mental-mean", "mental-tired", "mental-repeat", "mental-concentration", "mental-satisfied"
        # delete_data = "nonedata"
        # window_size_list_original = [60,120,240] # 60,120,240
        # window_size_list_head = [6,12,24] # 6,12,24
        mae_list = []
        mse_list = []
        rmse_list = []
        r2_list = []
        mape_list = []
        df0 = pd.DataFrame()
        df1 = pd.DataFrame()
        all_df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        all_df2 = pd.DataFrame()
        error_df = pd.DataFrame()
        importance_df = pd.DataFrame()
        all_df_error = pd.DataFrame()
        all_df_importance = pd.DataFrame()
        mae_mean = 0
        mse_mean = 0
        rmse_mean = 0
        r2_mean = 0
        mape_mean = 0



        # 12人分の1人抜きのデータを作成
        for object in object_name:
            for feature_name in feature_name_list:
                print(feature_name)
                for part in data_parts:
                    print(part)
                    if part == "頭部":
                        window_size_list = window_size_list_head
                    else:
                        window_size_list = window_size_list_original
                    # print(window_size_list)
                    for windowsize in window_size_list:
                        print(windowsize)
                        for subject_num in range(1, GENERAL_MODEL_NUM+1):
                            OneOut_alldata_path, One_data_path = OneOut_alldatacsv(subject_num, feature_name, part, windowsize, delete_data, object)
                            # print(OneOut_alldata_path)
                            mae,model_name,mse,rmse,r2,mape,error,importance  = General_MachineLearning(OneOut_alldata_path, One_data_path, subject_num, feature_name, part, object, windowsize, None, delete_data, True,None, None,model)
                            
                            
                            df0 = pd.DataFrame({
                                'model_name': model_name,
                                'subject_num': subject_num,
                                'object_name': object,
                                'windowsize': windowsize,
                                'feature_name': feature_name,
                                'part': part,                         
                            }, index=[0]) 
                            df1 = pd.DataFrame({
                                'model_name': model_name,
                                'subject_num': subject_num,
                                'object_name': object,
                                'windowsize': windowsize,
                                'feature_name': feature_name,
                                'part': part,
                                'mae': mae,
                                'mse': mse,
                                'rmse': rmse,
                                'r2': r2,
                                'mape': mape,
                                
                            }, index=[0]) 
                            all_df1 = pd.concat([all_df1, df1])
                            # print(f"被験者{subject_num}のMAE: {mae}")
                        mae_mean=np.mean(mae)
                        mse_mean=np.mean(mse)
                        rmse_mean=np.mean(rmse)
                        r2_mean=np.mean(r2)
                        mape_mean=np.mean(mape)
                        print(f"MAE_mean: {mae_mean}")
                        
                        df2 = pd.DataFrame({
                            'object_name': object,
                            'windowsize': windowsize,
                            'feature_name': feature_name,
                            'part': part,
                            'mae_mean': mae_mean,
                            'mse_mean': mse_mean,
                            'rmse_mean': rmse_mean,
                            'r2_mean': r2_mean,
                            'mape_mean': mape_mean
                            
                        }, index=[0]) 
                        all_df2 = pd.concat([all_df2, df2])
                        df0 = df0.reset_index(drop=True)
                        # error_df = error_df.reset_index(drop=True)
                        error_df = pd.concat([error_df, error])
                        all_df_error = pd.concat([all_df_error, pd.concat([df0.T, error_df], axis=0,ignore_index=True)], axis=1,ignore_index=True)
                        # print(df_combined)
                        # df_temp = df0.apply(lambda x: pd.Series(x), axis=1).stack().reset_index(level=1, drop=True) # 基本情報のデータベースを列になるように準備
                        
                    
                        # df_dict[f'df0_flattened_{i}'].columns = error_df.columns
                        # df_temp_error = pd.concat([df_dict[f'df0_flattened_{i}'], error_df], axis=0,ignore_index=True) # 縦方向に連結
                        if model == "1" or model == "4":
                            # importance_mean_df = pd.DataFrame(importance, columns=[f'importance_mean_{i}'])
                            # df_dict[f'df0_flattened_{i}'].columns = importance_mean_df.columns
                            # df_temp_importance = pd.concat([df_dict[f'df0_flattened_{i}'], importance_mean_df], axis=0,ignore_index=True)
                            importance_df = importance_df.reset_index(drop=True)
                            # importance_df = pd.concat([importance_df, pd.DataFrame(importance)], axis=1)
                            all_df_importance = pd.concat([df0, importance_df],  axis=0,ignore_index=True)
                            # print(df_importance)
                        
        # print("MAE_list:", mae_list)
        # mae_mean=np.mean(mae_list)
        # print("MAE_average:", mae_mean)
        # df2 = pd.DataFrame({
        #     'mae_mean': mae_mean
        # })
        # df = pd.concat([all_df1, df2], axis=1)
        # Excelファイルに書き出し
        import datetime
        now = datetime.datetime.now()
        # 時間を変換
        now = now.strftime('%Y%m%d%H%M%S') 
        OneOut_output_path = "E:\\データ処理\\モデル構築\\model\\汎化\\M1\\"+model_name+'\\'+model_name+"-delete_"+delete_data+"-resultss("+now+").xlsx"
        
        if model == "1" or model == "4":
            with pd.ExcelWriter(OneOut_output_path, engine='xlsxwriter') as writer:
                all_df2.to_excel(writer, index=False, sheet_name='mean')
                all_df_error.to_excel(writer, sheet_name='error')
                all_df_importance.to_excel(writer, sheet_name='importance')# all_df1.to_excel(writer, sheet_name='detail')
        else:
            with pd.ExcelWriter(OneOut_output_path, engine='xlsxwriter') as writer:
                all_df2.to_excel(writer, index=False, sheet_name='mean')
                all_df_error.to_excel(writer, sheet_name='error')# all_df1.to_excel(writer, sheet_name='detail')

if __name__ == '__main__':
     main()
