# モデルの作成・構築を行う
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import learning_curve
import Features
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

import paramerter


# モデルの作成
def ModelBuild(features_csv_path, model, subject_number, feature_name, part, obgective_variable, windowsize, delete_data, object_name,first_iteration,i,df_dict):
    # 回帰問題を機械学習にかけるためのモデルを作成する
    if subject_number == "0":
        features_csv_path = Features.Make_alldatacsv(subject_number, feature_name, part, obgective_variable, windowsize, delete_data, object_name)
        # features_csv_pathが空でない場合
        if features_csv_path:
            print("Success to make all-data csv file.")
    # csvファイルの読み込み
    csv_path = features_csv_path
    df = pd.read_csv(csv_path)
    # NaNを含む行を削除
    # dropna()を使用する前の行数
    row_count_before = df.shape[0]

    # NaNを含む行を削除
    df = df.dropna()

    # dropna()を使用した後の行数
    row_count_after = df.shape[0]

    # 削除された行数を計算
    deleted_rows = row_count_before - row_count_after
    # print(df)
    # logging.info(df)
    # dfの列数を取得
    column_count = len(df.columns)
    # logging.info(column_count)
    # ヘッダー名がtarget_dataの列をyに格納
    y = df['target_data']
    if obgective_variable == "nasatlx":
        y = y*100
    df = df.drop('target_data', axis=1)
    # dfを説明変数としてXに格納
    X = df
    # logging.info(X)
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)  # 定数で固定すると実行ごとに分割が変わらない
    # X_trainとX_testが不一致かの検証
    # # ダミーのデータ
    # X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    # y_train = np.random.rand(100)
    # X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f'feature_{i}' for i in range(5)])
    # y_test = np.random.rand(20)
    return MachineLearning(X_train, X_test, y_train, y_test, model, subject_number, feature_name, part, obgective_variable, windowsize, deleted_rows, delete_data, object_name,first_iteration,i,df_dict)

# 機械学習


def MachineLearning(X_train, X_test, y_train, y_test, model, subject_number, feature_name, part, obgective_variable, windowsize, deleted_rows, delete_data, object_name,first_iteration,i,df_dict):
    # 回帰問題を機械学習にかける
    model = str(model)
    y_pred_all = np.array([])
    y_test_all = np.array([])
    df_importance = pd.DataFrame()
    df_with_error = pd.DataFrame()
    df_with_split_error = pd.DataFrame()
    error_list = []
    fold = 0
    # 10分割交差検証の設定
    FOLD = paramerter.FOLD
    kf = KFold(n_splits=FOLD, shuffle=True, random_state=0)
    

    # 各分割に対して学習とテストを行う
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
         # 訓練データとテストデータの重複を確認
        common = pd.merge(X_train_fold, X_test_fold, how='inner')
        # print(f"Number of common rows: {len(common)}")

        # モデルの作成
        if model == "1":
            model_name="RF"
            logging.info('RF')
            # print('RandomForest')
            regressor = RandomForestRegressor(random_state=0, n_estimators=100)
            count = fold+1
            # print("start learning : " + str(count) + "/10")
            
            regressor.fit(X_train_fold, y_train_fold)
            # 予測
            y_pred = regressor.predict(X_test_fold)
            
            # 特徴量の重要度の計算
            importances = regressor.feature_importances_
            # 特徴量名と重要度を対応付ける
            feature_importances = list(zip(X_train.columns, importances))

            feature_importances = sorted(
                feature_importances, key=lambda x: x[1], reverse=True)

            # y_predとy_test_foldを蓄積
            y_pred_all = np.append(y_pred_all, y_pred)
            # pandas.Seriesからnumpy.arrayに変換
            y_test_all = np.append(y_test_all, y_test_fold.values)

            df_importance[fold] = regressor.feature_importances_
            # print(X_train.columns)
            # パラメータチューニング
            # candidate_params = {'n_estimators': [100]}#10,1000
            # kf_ = KFold(n_splits=4, shuffle=True, random_state=0)
            # gs = GridSearchCV(estimator=reg, param_grid=candidate_params, cv=kf_, n_jobs=4)
            # gs.fit(X_train, y_train)
            # regressor = Pipeline([('estimator', gs.best_estimator_)])
            # 層化10分割交差検証の設定

        elif model == "2":
            model_name="MRM"
            print('MRM')  # 重回帰分析
            regressor = LinearRegression()
            regressor.fit(X_train_fold, y_train_fold)  # 訓練データにフィットさせる
            y_pred = regressor.predict(X_test_fold)  # テストデータに対する予測

            
        elif model == "3":
            print('SVR')
            model_name="SVR"
            # SVRモデルのインスタンスを作成
            regressor = SVR(kernel='rbf')
            # 訓練データにフィットさせる
            regressor.fit(X_train_fold, y_train_fold)
            # テストデータに対する予測
            y_pred = regressor.predict(X_test_fold)
            # y_predとy_test_foldを蓄積
            y_pred_all = np.append(y_pred_all, y_pred)
            # pandas.Seriesからnumpy.arrayに変換
            y_test_all = np.append(y_test_all, y_test_fold.values)
        
        elif model == "4":
            print('LightGBM')
            model_name="LGBM"

            # LightGBMのパラメータ設定
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
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
            lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
            lgb_eval = lgb.Dataset(X_test_fold, y_test_fold, reference=lgb_train)

            # モデルの学習
            gbm = lgb.train(params,
                            lgb_train,
                            # num_boost_round=20,
                            # valid_sets=lgb_eval,
                            # early_stopping_rounds=5
                            )
            # importances = gbm.feature_importance()
            df_importance[fold] = gbm.feature_importance()


            # テストデータに対する予測
            y_pred = gbm.predict(X_test_fold, num_iteration=gbm.best_iteration)

            # y_predとy_test_foldを蓄積
            y_pred_all = np.append(y_pred_all, y_pred)
            # pandas.Seriesからnumpy.arrayに変換
            y_test_all = np.append(y_test_all, y_test_fold.values)
        

        # y_pred_allとy_test_allをNumPy配列に変換
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test_fold.values)
        fold += 1
        # 学習曲線のプロット
    
    mae = mean_absolute_error(y_test_all, y_pred_all)
    # print('MAE = {:.3f}'.format(mae))

    # MSEの計算
    mse = mean_squared_error(y_test_all, y_pred_all)
    # print('MSE = {:.3f}'.format(mse))

    # RMSEの計算
    rmse = np.sqrt(mse)
    # print('RMSE = {:.3f}'.format(rmse))

    # R^2の計算
    r2 = r2_score(y_test_all, y_pred_all)
    # print('R^2 = {:.3f}'.format(r2))

    # 平均絶対パーセンテージ誤差(MAPE)の計算
    if (y_test_all == 0).any():
        # y_test_allの中の0を非常に小さい値に置き換える
        y_test_all_replaced = y_test_all.copy()
        y_test_all_replaced[y_test_all_replaced == 0] = 1e-10
        # mapeを計算
        mape = np.mean(np.abs((y_test_all - y_pred_all) / y_test_all_replaced)) * 100
        
    else:
        mape = np.mean(np.abs((y_test_all - y_pred_all) / y_test_all)) * 100
        # print('MAPE = {:.3f}'.format(mape))

    error = y_pred_all -  y_test_all
    if first_iteration:
        first_iteration = False  # フラグを更新
    else:
        i += 1  # iを更新
    

    # headerを作成
    header = ['model','obgective_variable','subject_number', 'data_part', 'windowsize', 'feature_name',
              'MAE', 'MSE', 'RMSE', 'R^2', 'MAPE', 'deleted_rows','Duplication']#'model','subject_number', 'data_part', 'obgective_variable', 'windowsize', 'feature_name','Actual', 'Predicted', 'error', 'Feature_name', 'Importance_1','Importance_2','Importance_3','Importance_4','Importance_5','Importance_6','Importance_7','Importance_8','Importance_9','Importance_10','Importance_mean','MAE', 'MSE', 'RMSE', 'R^2', 'MAPE', 'deleted_rows','Duplication'
    df0 = pd.DataFrame({
        'model': [model],
        'obgective_variable': [object_name],
        'subject_number': [str(subject_number)],
        'data_part': [part],
        'windowsize': [str(windowsize)],
        'feature_name': [feature_name]
    })

    # 予測値と実際の値をDataFrameにまとめる
    df1 = pd.DataFrame({
        'Actual': y_test_all,
        'Predicted': y_pred_all,
        'error': error.flatten()  # 1次元の配列に変換
    })
    # print(df_importance)
    # feature_names = [feature for feature, importance in feature_importances]

    # # 空のデータフレームを初期化
    # df_importance_mean = df_importance.mean(axis=1)
    # df2 = pd.DataFrame(feature_names, columns=['Feature_Names'])
    # df2 = pd.concat([df2, df_importance], axis=1)
    # df2 = pd.concat([df2, df_importance_mean], axis=1)
    # print(df2)


    # MSE, RMSE, R^2をDataFrameにまとめる
    df3 = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'R^2': [r2],
        'MAPE': [mape],
        'deleted_rows': [deleted_rows],
        'Duplication': [len(common)]
    })

    # すべての結果を1つのDataFrameに結合
    df = pd.concat([df0, df3], axis=1, ignore_index=True) #df0, df1, df2, df3
    # print(df)
    # headerを設定
    df.columns = header
    # print(df)
    
    # plot用のデータフレーム作成
    # df0_flattened = df0.apply(lambda x: pd.Series(x), axis=1).stack().reset_index(level=1, drop=True).to_frame('df0')
    df_temp = df0.apply(lambda x: pd.Series(x), axis=1).stack().reset_index(level=1, drop=True).to_frame(f'df0_{i}')
    df_dict[f'df0_flattened_{i}'] = df_temp

    # df0_flattenedの下にerrorを連結
    error_df = pd.DataFrame(error, columns=[f'error_{i}'])
    df_dict[f'df0_flattened_{i}'].columns = error_df.columns
    df_temp_error = pd.concat([df_dict[f'df0_flattened_{i}'], error_df], axis=0,ignore_index=True) # 縦方向に連結
    # df_with_error = pd.concat([df_with_error, df_temp_error], axis=1, ignore_index=True)
    
    # split_error_df = pd.DataFrame(error_list)
    # print(split_error_df)
    # df_dict[f'df0_flattened_{i}'].columns = split_error_df.columns #列名の設定
    # df_with_split_error = pd.concat([df_dict[f'df0_flattened_{i}'], split_error_df], axis=0,ignore_index=True)
    # error_df = pd.DataFrame(error_list).T
    # df_with_split_error = pd.concat([df_with_error, error_df], axis=1, ignore_index=True)
    # print(df_with_error)
            # df_with_split_error = pd.concat([df_with_split_error, df_temp_split_error], axis=1, ignore_index=True)
    if model == "1" or model == "4":
        df_importance_mean = df_importance.mean(axis=1)
        importance_mean_df = pd.DataFrame(df_importance_mean, columns=[f'importance_mean_{i}'])
        df_dict[f'df0_flattened_{i}'].columns = importance_mean_df.columns
        df_temp_importance = pd.concat([df_dict[f'df0_flattened_{i}'], importance_mean_df], axis=0,ignore_index=True)

    # df_temp_errorとdf_temp_importanceをそれぞれdf_with_errorとdf_with_importanceに横に連結
    
    # df_with_importance = pd.concat([df_with_importance, df_temp_importance], axis=1, ignore_index=True)
    # print(df_with_error)
    # print(df_with_importance)

    # 時間を取得
    import datetime
    now = datetime.datetime.now()
    # 時間を変換
    now = now.strftime('%Y%m%d%H%M%S')

    ###########################各回ごとの結果を出力#################################################
    # dfをcsvファイルに出力 
    # filename = 'E:\\データ処理\\モデル構築\\model\\被験者'+str(subject_number)+'\\cross\\log\\'+str(subject_number)+"_" + part+'_result('+now+')_' + \
    #     obgective_variable+"-" + \
    #     str(windowsize)+"-"+feature_name+"-delete_" + \
    #     delete_data+"_"+object_name+'.xlsx'
    # df.to_excel(filename, index=False)
    #############################################################################################

    # # 誤差の計算
    # errors = y_test - y_pred

    # # バイオリンプロットの作成
    # plt.figure(figsize=(8, 6))
    # sns.violinplot(y=errors)
    # plt.title('Error distribution')

    # # 図を閉じる
    # # 図の保存
    # plt.savefig('E:\\データ処理\\モデル構築\\model\\被験者'+str(subject_number)+'\\'+part+'_violin('+now+')_'+obgective_variable+"-"+str(windowsize)+"-"+feature_name+'.png')
    # plt.close()
    # モデルの保存
    # filename = 'E:\\データ処理\\モデル構築\\model\\被験者'+str(subject_number)+'\\'+part+'_model('+now+')_'+obgective_variable+"-"+str(
        # windowsize)+"-"+feature_name+"-delete_"+delete_data+"_"+object_name+'.sav'
    # pickle.dump(regressor, open(filename, 'wb'))
    # print("regressor saved successfully.")
    if model == "1" or model == "4":
        return df,df_temp_error,df_temp_importance,first_iteration,i,model_name
    else:
        return df,df_temp_error,first_iteration,i,model_name
