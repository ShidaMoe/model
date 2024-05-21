import pandas as pd
import numpy as np
import statistics
import scipy.stats as stats
from scipy.spatial import distance
from numpy import linalg as la
import math
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.pipeline import Pipeline as pipe
from imblearn.over_sampling import SMOTE
from copy import deepcopy
from imblearn.under_sampling import RandomUnderSampler
import pickle


#### データセット作成用 ####
def triming(input_path, output_path, key_num, end_num):
    del_index = []

    print(input_path)
    input_book = pd.ExcelFile(input_path)
    input_df = input_book.parse('Data')

    # textstartまでを削除
    text_start_index = input_df.index[(input_df['Event'] == 'Eye tracker Calibration end')].tolist()
    text_df = input_df.drop(range(text_start_index[0] + 2))

    # 必要な列を抽出
    df = text_df.loc[:, ['Recording timestamp', 'Computer timestamp', 'Eyetracker timestamp',
                         'Event', 'Gaze point X', 'Gaze point Y', 'Pupil diameter left',
                         'Pupil diameter right', 'Validity left', 'Validity right',
                         'Eye movement type', 'Gaze event duration', 'Fixation point X',
                         'Fixation point Y']]
    df = df.reset_index(drop=True)

    # keyboard操作の前後と作業開始後、終了前の10sを削除
    # 全てのdel_indexを1列のリストにして、ループ後に一括削除する

    # 作業開始直後の3sをdel_indexに追加
    del_index.extend(df.index[df['Recording timestamp'] <= df.iat[0, 0] + key_num * 1000000].tolist())

    # keyboardEventの前後1sをdel_indexに追加
    key_index = df.index[df['Event'] == 'KeyboardEvent'].tolist()
    for i in key_index[0:-1]:
        key_time = df.iat[i, 0]
        del_index.extend(df.index[(df['Recording timestamp'] >= key_time - key_num * 1000000) & (
                df['Recording timestamp'] <= key_time + key_num * 1000000)].tolist())

    # 作業終了時のkeyboardEventの前1sをdei_indexに追加
    key_time = df.iat[key_index[-1], 0]
    del_index.extend(df.index[(df['Recording timestamp'] >= key_time - end_num * 1000000) & (
            df['Recording timestamp'] <= key_time + end_num * 1000000)].tolist())

    # del_indexの行を一括削除
    df = df.drop(index=df.index[list(dict.fromkeys(del_index))])  # list(dict.fromkeys())で，del_index内の重複を削除
    # 欠損データを含む行を削除
    # df = df.dropna(subset=['Gaze point X', 'Gaze point Y', 'Pupil diameter left', 'Pupil diameter right']) # ここを実行する場合：delete invalid
    df = df.reset_index(drop=True)

    # ファイル書き出し
    df.to_excel(output_path, sheet_name='Data', index=False)
    df = pd.DataFrame()


# 瞳孔径の標準化用
def calc_zscore(input_df_path, output_df_path):
    # 仮想的なデータフレームを作成
    input_book = pd.ExcelFile(input_df_path)
    input_df = input_book.parse('Data')

    # 'Pupil diameter left' の非Nan値で標準化
    non_nan_values_left = input_df['Pupil diameter left'].dropna()
    input_df.loc[~input_df['Pupil diameter left'].isnull(), 'Pupil diameter left'] = (input_df[
                                                                                          'Pupil diameter left'] - non_nan_values_left.mean()) / non_nan_values_left.std()
    # 'Pupil diameter right' の非Nan値で標準化
    non_nan_values_right = input_df['Pupil diameter right'].dropna()
    input_df.loc[~input_df['Pupil diameter right'].isnull(), 'Pupil diameter right'] = (input_df[
                                                                                            'Pupil diameter right'] - non_nan_values_right.mean()) / non_nan_values_right.std()

    input_df.to_excel(output_df_path,
                      sheet_name='Data', index=False)


# オーバーラップ用のデータセット作成
def sliced_dataset(input_path, output_path, n):
    input_book = pd.ExcelFile(input_path)
    input_df = input_book.parse('Data')
    if not input_df.empty:
        T_START = input_df.iat[0, 0] + n * 0.5 * 1000000
        input_df_sliced = input_df[input_df['Recording timestamp'] >= T_START]

        input_df_sliced.to_excel(
            output_path,
            sheet_name='Data', index=False)
        print(output_path)


# n(sec)未満のウィンドウを削除
def del_windows(t_start, input_df_, n_):
    df = pd.DataFrame()
    del_index = []  # 削除する行番号を格納
    # input_df_ = input_df_[(input_df_['Validity left'] == 'Valid') & (input_df_['Validity right'] == 'Valid')]  # delete invalid の場合
    # input_df_.reset_index(drop=True)
    # ウィンドウ単位のループ
    while True:
        a = 0
        start_idx = 10 ** 9
        t_end = t_start + n_ * 1000000
        # ウィンドウ内（t_start~t_endの間）のタイムスタンプのindexを格納
        # print(len(input_df_))
        del_list = input_df_.index[(input_df_['Recording timestamp'] >= t_start) & (
                input_df_['Recording timestamp'] <= t_end)].tolist()
        del_list = [i for i in del_list if i < len(input_df_)]

        if len(del_list) == 1:  # ウィンドウ内にサンプルが一つしかない時
            # print('one sample', input_df_.iat[del_list[0], 0])
            del_index.extend(del_list)
        else:  # ウィンドウ内にサンプルが複数ある場合
            if input_df_.iat[del_list[-1], 0] - input_df_.iat[del_list[0], 0] < (n_ - 1) * 1000000 - 200000:
                # print('end', input_df_.iat[del_list[-1], 0], input_df_.iat[del_list[0], 0], input_df_.iat[del_list[-1], 0] - input_df_.iat[del_list[0], 0])
                # print('start end', input_df_.iat[del_list[0]-1, 0], input_df_.iat[del_list[-1], 0])
                del_index.extend(del_list)  # ウィンドウがn-1秒になってしまっている場合，除外
            else:
                for i in range(1, len(del_list)):  # ウィンドウ内で1秒以上の空白があれば除外
                    if input_df_.iat[del_list[i], 0] - input_df_.iat[del_list[i - 1], 0] > 100000:
                        # print('middle', input_df_.iat[del_list[i], 0], input_df_.iat[del_list[i - 1], 0], input_df_.iat[del_list[i], 0] - input_df_.iat[del_list[i - 1], 0])
                        # print('start end', input_df_.iat[del_list[0]-1, 0], input_df_.iat[del_list[-1], 0])
                        del_index.extend(del_list[0:i])  # ウィンドウの先頭から空白の部分まで除外
                        a = 1
                        start_idx = i
                        break
        # 更新
        if input_df_.iat[del_list[-1], 0] == input_df_.iat[-1, 0]:  # 末尾のとき
            if a == 1:
                del_index.extend(del_list[start_idx:])
            break
        else:  # 更新
            if a == 1:
                t_start = input_df_.iat[del_list[start_idx], 0]
            else:
                t_start = input_df_.iat[del_list[-1] + 1, 0]
        del_list.clear()
    df = input_df_.drop(index=input_df_.index[list(dict.fromkeys(del_index))])
    df = df.reset_index(drop=True)

    return df


# 削除後のデータを保存
def make_files(input_path, output_path, n):
    input_book = pd.ExcelFile(input_path)
    input_df = input_book.parse('Data')

    if not input_df.empty:
        T_START = input_df.iat[0, 0]
        output_df = del_windows(T_START, input_df, n)

        output_df.to_excel(output_path,
                           sheet_name='Data', index=False)
        print(output_path)


## 特徴量計算
# 記述統計量の計算
def calc_disc_stat(lst__):
    lst = np.array(lst__)[~np.isnan(np.array(lst__))]
    Mean = statistics.mean(lst)
    Max = max(lst)
    Min = min(lst)
    Med = statistics.median(lst)
    if len(lst) == 1:
        Std = 0
    else:
        Std = statistics.stdev(lst)
    Skew = stats.skew(lst)
    Kurtosis = stats.kurtosis(lst)
    Range = Max - Min
    lst_ = [Mean, Max, Min, Med, Std, Skew, Kurtosis, Range]
    return lst_


# 固視分散
def calc_fvar(lst_v):
    var = np.array([])
    lst_v_ = [np.array(i)[~np.isnan(np.array(i))].tolist() for i in lst_v if np.array(i)[~np.isnan(np.array(i))].any()]
    p_mean = [sum(column) for column in zip(*lst_v_)]
    p_mean = np.array([p_mean[0] / len(lst_v_), p_mean[1] / len(lst_v_)])  # 各固視点の重心

    for p in lst_v:
        # 重心と各固視点のユークリッド距離の2乗を計算してvarに追加
        var = np.append(var, (distance.euclidean(np.array(p), p_mean)) ** 2)
    f_di = (np.mean(var)) ** (1 / 2)  # varの平均のルート
    return f_di


# 特徴量計算の本体
def calc_features(input_path, output_path, n, f_num):
    input_book = pd.ExcelFile(input_path)
    print(input_path)
    input_df = input_book.parse('Data')
    features = pd.DataFrame()  # 全特徴量を格納

    if not input_df.empty:
        t_start = input_df.iat[0, 0]

        while True:
            drt_list_f = []  # ウィンドウ内の固視時間を格納
            drt_list_fp = []  # ウィンドウ内の固視座標を格納
            drt_list_px = []  # ウィンドウ内の瞳孔径_x
            drt_list_py = []  # ウィンドウ内の瞳孔径_y
            drt_list_px_ = []  # 標準化後のウィンドウ内の瞳孔径_x
            drt_list_py_ = []  # 標準化後のウィンドウ内の瞳孔径_y
            drt_list_s = []  # ウィンドウ内のサッカード時間を格納
            amp_list = []  # サッカード振幅を格納
            sp_list = []  # サッカード速度を格納
            agl_list_a = []  # 絶対サッカード角度を格納
            agl_list_r = []  # 相対サッカード角度を格納

            fsd_f = []  # 固視時間/サッカード時間（合計）を格納
            fv_f = []  # ウィンドウ内の固視分散を格納
            sd_f = []  # ウィンドウ内のサッカード時間の特徴量を格納
            sl_f = []  # ウィンドウ内のサッカード振幅の特徴量を格納
            sp_f = []  # ウィンドウ内のサッカード速度の特徴量を格納
            asa_f = []
            rsa_f = []
            pd_fx = []  # ウィンドウ内の左目の瞳孔径の特徴量を格納
            pd_fy = []  # ウィンドウ内の右目の瞳孔径の特徴量を格納

            P_START = []  # サッカードの開始点を全て格納（相対サッカード角度の計算で使う）
            P_END = []  # サッカードの終了点を全て格納（相対サッカード角度の計算で使う）
            p_start = []  # サッカードの開始点を格納（サッカード振幅）
            p_end = []  # サッカードの終了点を格納（サッカード振幅）
            SAC_START = []  # サッカードの開始時刻を全て格納
            SAC_END = []  # サッカードの終了時刻を全て格納

            # 初期化
            fix_period = []
            sac_period = []

            horiz_num = 0

            t_end = t_start + n * 1000000
            # ウィンドウ内のindexをすべて格納
            win_index = input_df.index[(input_df['Recording timestamp'] >= t_start) & (
                    input_df['Recording timestamp'] <= t_end)].tolist()

            ## 瞳孔径の計算
            for r in range(0, len(win_index)):
                # 瞳孔径を格納
                drt_list_px.append(input_df.at[input_df.index[win_index[r]], 'Pupil diameter left'])
                drt_list_py.append(input_df.at[input_df.index[win_index[r]], 'Pupil diameter right'])
            if not (np.isnan(drt_list_px).all() or np.isnan(drt_list_py).all()):
                drt_list_px_ = np.array(drt_list_px)[~np.isnan(np.array(drt_list_px))].tolist()
                drt_list_py_ = np.array(drt_list_py)[~np.isnan(np.array(drt_list_py))].tolist()

            # ウィンドウ内のfixationに該当するindexを格納
            fix_index = input_df.index[(input_df['Recording timestamp'] >= t_start) & (
                    input_df['Recording timestamp'] <= t_end) & (
                                               input_df['Eye movement type'] == 'Fixation')].tolist()
            # ウィンドウ内のsaccadeに該当するindexを格納
            sac_index = input_df.index[(input_df['Recording timestamp'] >= t_start) & (
                    input_df['Recording timestamp'] <= t_end) & ((input_df['Eye movement type'] == 'Saccade') | (
                    input_df['Eye movement type'] == 'Unclassified'))].tolist()

            ## 固視
            if fix_index:
                start = None
                for i, val in enumerate(fix_index):
                    if i == 0:
                        start = val
                    elif fix_index[i] - fix_index[i - 1] != 1:
                        fix_period.append([start, fix_index[i - 1]])
                        start = val

                # 最後の連続した区間を追加
                if start is not None:
                    fix_period.append([start, fix_index[i - 1]])
                # ウィンドウで端にfixationがある場合
                if fix_period[0][0] == win_index[0]:
                    fix_period.pop(0)
                if fix_period and fix_period[-1][-1] == win_index[-1]:
                    fix_period.pop()

            if fix_period:
                for fp in fix_period:
                    drt_list_f.append(input_df.at[input_df.index[fp[0]], 'Gaze event duration'])  # 固視時間
                    # fix_indexに該当する固視座標を格納
                    drt_list_fp.append([input_df.at[input_df.index[fp[0]], 'Fixation point X'],
                                        input_df.at[input_df.index[fp[0]], 'Fixation point Y']])  # 固視座標
                    # 固視分散
                    fv_f = calc_fvar(drt_list_fp)
            else:
                print('no fixation')
                # print('file: ' + sm + fn + dy)
                print('win_start_rec: ', input_df.at[input_df.index[win_index[0]], 'Recording timestamp'])
                print('================================================')

            ## サッカード
            # サッカード振幅 サッカード時間
            if sac_index:
                start = None
                for i, val in enumerate(sac_index):
                    if i == 0:
                        start = val
                    elif sac_index[i] - sac_index[i - 1] != 1:
                        sac_period.append([start, sac_index[i - 1]])
                        start = val

                # 最後の連続した区間を追加
                if start is not None:
                    sac_period.append([start, sac_index[-1]])
                # ウィンドウで端にsaccadeがある場合
                if sac_period[0][0] == win_index[0]:
                    sac_period.pop(0)
                if sac_period and sac_period[-1][-1] == win_index[-1]:
                    sac_period.pop()

                for sp in sac_period:
                    sac_start = input_df.at[input_df.index[sp[0] - 1], 'Recording timestamp']  # サッカード開始時刻
                    x = input_df.at[input_df.index[sp[0] - 1], 'Gaze point X']  # 視点のx座標
                    y = input_df.at[input_df.index[sp[0] - 1], 'Gaze point Y']  # 視点のy座標
                    SAC_START.append(sac_start)
                    p_start = [x, y]
                    P_START.append(p_start)
                    sac_end = input_df.at[input_df.index[sp[1] + 1], 'Recording timestamp']  # サッカード終了時刻
                    x = input_df.at[input_df.index[sp[1] + 1], 'Gaze point X']  # 視点のx座標
                    y = input_df.at[input_df.index[sp[1] + 1], 'Gaze point Y']  # 視点のy座標
                    SAC_END.append(sac_end)
                    p_end = [x, y]
                    P_END.append(p_end)

                    drt_list_s.append((sac_end - sac_start) / 1000)  # サッカード時間
                    if not (np.isnan(p_start).any() or np.isnan(p_end).any()):
                        amp_list.append(distance.euclidean(p_start, p_end))  # サッカード振幅を計算
                        # 絶対サッカード角度を計算
                        x_ = p_end[0] - p_start[0]
                        y_ = p_end[1] - p_start[1]
                        if x_ == 0:
                            angle = 90.0
                        else:
                            tan = y_ / x_
                            angle = np.arctan(tan) * 180 / math.pi
                        agl_list_a.append(angle)

                        # 水平サッカードの個数をカウント
                        if -30 <= angle <= 30:
                            horiz_num += 1
                    else:
                        amp_list.append(np.nan)
                # 相対サッカード角度
                for b in range(1, len(drt_list_fp) - 1):
                    p0 = np.array(drt_list_fp[b - 1])
                    p1 = np.array(drt_list_fp[b])
                    p2 = np.array(drt_list_fp[b + 1])
                    p0_ = p0 - p1
                    p2_ = p2 - p1
                    inn = np.inner(p0_, p2_)
                    nor = la.norm(p0_) * la.norm(p2_)
                    if nor != 0:
                        c = inn / nor
                        arcC = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
                        agl_list_r.append(arcC)

            if not agl_list_r:
                print('angle list')
                print('p0, p1, p2: ', p0, p1, p2)
                # print('file: ' + sm + fn + dy)
                print('win_start_rec: ', input_df.at[input_df.index[win_index[0]], 'Recording timestamp'])
                print('================================================')
            elif not sac_index:
                print('no saccade')
                # print('file: ' + sm + fn + dy)
                print('win_start_rec: ', input_df.at[input_df.index[win_index[0]], 'Recording timestamp'])
                print('================================================')

            if fix_period and sac_period:
                # 固視時間/サッカード時間
                fsd_f.append(sum(drt_list_f) / sum(drt_list_s))
            # 特徴量を一つのリストにまとめる（必要なものは記述統計量を計算する）
            # 固視
            if fix_period:
                fd_f = calc_disc_stat(drt_list_f)  # 固視時間の記述統計量を計算
                if len(drt_list_f) > 4:
                    fd_f.append(len(drt_list_f))  # 固視回数を末尾に追加
                if fv_f:
                    fd_f.append(fv_f)  # 固視分散を末尾に追加
                if sac_index:
                    fd_f.extend(fsd_f)  # 固視時間/サッカード時間を末尾に追加
            # 瞳孔径
            if (not (np.isnan(drt_list_px_).all() or np.isnan(drt_list_py_).all())) and drt_list_px_ and drt_list_py_:
                pd_fx = calc_disc_stat(drt_list_px_)
                pd_fy = calc_disc_stat(drt_list_py_)
                fd_f.extend(pd_fx)  # 左の瞳孔径を末尾に追加
                fd_f.extend(pd_fy)  # 右の瞳孔径を末尾に追加
            # サッカード
            if sac_period:
                sd_f = calc_disc_stat(drt_list_s)  # サッカード時間の記述統計量を計算
                if (not np.isnan(amp_list).all()) and amp_list:
                    sl_f = calc_disc_stat(amp_list)  # サッカード振幅の記述統計量を計算
                sp_list = [m / n for (m, n) in zip(amp_list, drt_list_s)
                           if not (np.isnan(m) or np.isnan(n))]  # サッカード速度の記述統計量を計算
                if sp_list:
                    sp_f = calc_disc_stat(sp_list)
                if agl_list_a:
                    asa_f = calc_disc_stat(agl_list_a)  # 絶対サッカード角度の記述統計量を計算
                if agl_list_r:
                    rsa_f = calc_disc_stat(agl_list_r)  # 相対サッカード角度の記述統計量を計算
                if (not np.isnan(amp_list).all()) and amp_list:
                    sd_f.extend(sl_f)  # サッカード振幅を末尾に追加
                    sd_f.extend(sp_f)  # サッカード速度を末尾に追加
                sd_f.extend(asa_f)  # 絶対サッカード角度を末尾に追加
                if agl_list_r:
                    sd_f.extend(rsa_f)  # 相対サッカード角度を末尾に追加
                sd_f.append(horiz_num / len(drt_list_s))  # 水平サッカード割合を末尾に追加
                sd_f.append(len(drt_list_s))  # サッカード回数を末尾に追加
                fd_f.extend(sd_f)  # 固視の特徴量とサッカードの特徴量を結合
            fd_f.insert(0, input_df.loc[win_index[0], 'Recording timestamp'])
            if len(np.array(fd_f)[~np.isnan(np.array(fd_f))].tolist()) == (f_num + 1):
                features = pd.concat([features, pd.Series(fd_f)], axis=1)  # 特徴量をデータフレームにする
            # a += 1  # aはfeaturesの行数（=ウィンドウの数）
            # すべてのウィンドウで計算を終えたらループを抜ける，そうでなければウィンドウの開始時刻を更新
            if input_df.iat[win_index[-1], 0] == input_df.iat[-1, 0]:
                win_index.clear()
                break
            else:
                t_start = input_df.iat[win_index[-1] + 1, 0]
            win_index.clear()

    features.to_excel(output_path,
                      sheet_name='Data', index=False)
    print(output_path)


# データセットにする
def conbine_files(input_path, fn, num, dy, df_sm):
    input_book = pd.ExcelFile(input_path)
    print(input_path)
    input_df = input_book.parse('Data')
    if not input_df.empty:
        # 転置して行：データの件，列：特徴量の種類とする
        input_df_t = input_df.T
        # 列名として特徴量の名前を記入
        input_df_t = input_df_t.set_axis(
            ['Timestamp', 'fixation dur mean', 'fixation dur max', 'fixation dur min ', 'fixation dur med',
             'fixation dur std', 'fixation dur skew', 'fixation dur kurtosis', 'fixation dur range',
             'fixation count', 'fixation variance', 'fixation saccade dur ratio',
             'pupil left dia mean', 'pupil left dia max', 'pupil left dia min', 'pupil left dia med',
             'pupil left dia std', 'pupil left dia skew', 'pupil left dia kurtosis', 'pupil left dia range',
             'pupil right dia mean', 'pupil right dia max', 'pupil right dia min', 'pupil right dia med',
             'pupil right dia std', 'pupil right dia skew', 'pupil right dia kurtosis', 'pupil right dia range',
             'saccade dur mean', 'saccade dur max', 'saccade dur min', 'saccade dur med', 'saccade dur std',
             'saccade dur skew', 'saccade dur kurtosis', 'saccade dur range',
             'saccade amp mean', 'saccade amp max', 'saccade amp min', 'saccade amp med', 'saccade amp std',
             'saccade amp skew', 'saccade amp kurtosis', 'saccade amp range',
             'saccade vel mean', 'saccade vel max', 'saccade vel min', 'saccade vel med', 'saccade vel std',
             'saccade vel skew', 'saccade vel kurtosis', 'saccade vel range',
             'saccade abs ang mean', 'saccade abs ang max', 'saccade abs ang min', 'saccade abs ang med',
             'saccade abs ang std', 'saccade abs ang skew', 'saccade abs ang kurtosis', 'saccade abs ang range',
             'saccade rel ang mean', 'saccade rel ang max', 'saccade rel ang min', 'saccade rel ang med',
             'saccade rel ang std', 'saccade rel ang skew', 'saccade rel ang kurtosis', 'saccade rel ang range',
             'holizontal saccade prop', 'saccade count'], axis=1)

        tar = [fn for i in range(len(input_df_t))]  # 目的変数
        input_df_t['target'] = tar
        input_df_t['participant'] = num
        input_df_t['day'] = dy
        input_df_t['Timestamp'] = [i - input_df_t.loc[0, 'Timestamp'] for i in input_df_t['Timestamp']]
        df_sm = pd.concat([df_sm, input_df_t], axis=0)  # dfの下の行に追加
        df_sm.loc[df_sm['fixation count'] < 5, 'fixation count'] = np.nan

        return df_sm


#### 分類モデルの構築と評価 #####
# ウィンドウサイズごとのデータ件数の統一
def undersampling(file_name, n_, fold_, under1_, under2_, train_X_, train_y_, test_X_, test_y_, target_names_, train_class_num_2_, train_class_num_3_):
    # 非pickle化
    with open(file_name, 'rb') as f:
        data_obj = pickle.load(f)
        train_num_lst = data_obj['train_num_lst']
        test_num_lst = data_obj['test_num_lst']
    if n_ != 12:  # n!=12のときは，手動でウィンドウサイズ間，クラス間のデータを合わせる
        if under1_ and (not under2_):
            if len(target_names_) == 3:
                strategy = {0: train_num_lst[fold_][0], 1: train_num_lst[fold_][1], 2: train_num_lst[fold_][2]}
                strategy02 = {0: test_num_lst[fold_][0], 1: test_num_lst[fold_][1], 2: test_num_lst[fold_][2]}
                rus = RandomUnderSampler(random_state=0, sampling_strategy=strategy)
                train_X_, train_y_ = rus.fit_resample(train_X_, train_y_)
                rus02 = RandomUnderSampler(random_state=0, sampling_strategy=strategy02)
                test_X_, test_y_ = rus02.fit_resample(test_X_, test_y_)
            elif len(target_names_) == 2:
                strategy = {0: train_num_lst[fold_][0], 1: train_num_lst[fold_][1]}
                strategy02 = {0: test_num_lst[fold_][0], 1: test_num_lst[fold_][1]}
                rus = RandomUnderSampler(random_state=0, sampling_strategy=strategy)
                train_X_, train_y_ = rus.fit_resample(train_X_, train_y_)
                rus02 = RandomUnderSampler(random_state=0, sampling_strategy=strategy02)
                test_X_, test_y_ = rus02.fit_resample(test_X_, test_y_)

        elif under1_ and under2_:
            if len(target_names_) == 3:
                strategy = {0: min(train_num_lst[fold_]), 1: min(train_num_lst[fold_]), 2: min(train_num_lst[fold_])}
                strategy02 = {0: test_num_lst[fold_][0], 1: test_num_lst[fold_][1], 2: test_num_lst[fold_][2]}
            elif len(target_names_) == 2:
                strategy = {0: min(train_num_lst[fold_]), 1: min(train_num_lst[fold_])}
                strategy02 = {0: test_num_lst[fold_][0], 1: test_num_lst[fold_][1]}
            rus = RandomUnderSampler(random_state=0, sampling_strategy=strategy)
            train_X_, train_y_ = rus.fit_resample(train_X_, train_y_)
            rus02 = RandomUnderSampler(random_state=0, sampling_strategy=strategy02)
            test_X_, test_y_ = rus02.fit_resample(test_X_, test_y_)
        elif (not under1_) and under2_:
            if len(target_names_) == 3:
                # ここは自動計算
                strategy = {0: min(train_class_num_3_), 1: min(train_class_num_3_), 2: min(train_class_num_3_)}
            elif len(target_names_) == 2:
                strategy = {0: min(train_class_num_2_), 1: min(train_class_num_2_)}
            rus = RandomUnderSampler(random_state=0, sampling_strategy=strategy)
            train_X_, train_y_ = rus.fit_resample(train_X_, train_y_)
    else:  # n=12のときは，クラス間のデータは一番少ないのにあわせる（自動計算）
        if under2_:
            if len(target_names_) == 3:
                strategy = {0: min(train_class_num_3_), 1: min(train_class_num_3_), 2: min(train_class_num_3_)}
            elif len(target_names_) == 2:
                strategy = {0: min(train_class_num_2_), 1: min(train_class_num_2_)}
            rus = RandomUnderSampler(random_state=0, sampling_strategy=strategy)
            train_X_, train_y_ = rus.fit_resample(train_X_, train_y_)
    print('train_y_ 0 1 2: ', len(np.where(train_y_ == 0)[0]), len(np.where(train_y_ == 1)[0]),
          len(np.where(train_y_ == 2)[0]))
    print('test_y_ 0 1 2: ', len(np.where(test_y_ == 0)[0]), len(np.where(test_y_ == 1)[0]),
          len(np.where(test_y_ == 2)[0]))

    return train_X_, train_y_, test_X_, test_y_


# 外れ値の閾値計算
def calc_outlier(X_, f_num):
    below = []
    above = []
    X = deepcopy(X_)

    for i in range(0, f_num):
        n = 0
        calc_sd_list = X[:, i]
        SD = statistics.stdev(calc_sd_list)
        MEAN = statistics.mean(calc_sd_list)
        below.append(MEAN - 3*SD)
        above.append(MEAN + 3*SD)

    return below, above


# 外れ値の変換
def trans_outlier(below, above, X_, f_num):
    X = deepcopy(X_)
    O = [[0] * f_num for k in range(len(X))]
    for i in range(0, f_num):
        n = 0
        check_sd_list = X[:, i]
        if len(check_sd_list):
            for j in range(0, len(X)):  # len(X)はサンプル数のこと
                if X[j][i] < below[i]:
                    n += 1
                    X[j][i] = below[i]
                    O[j][i] = 1
                elif above[i] < X[j][i]:
                    n += 1
                    X[j][i] = above[i]
                    O[j][i] = 1
    sample_num = 0
    for j in range(0, len(X)):
        if sum(O[j]) >= 1:
            sample_num += 1

    return X


# 分類モデル作成
def classifier_build(model, train_X, train_y, over):
    if model == 1:
        print('RandomForest')
        clf = RandomForestClassifier(random_state=0)
        # パラメータチューニング
        candidate_params = {'n_estimators': [10, 100, 1000]}
        kf_ = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        gs = GridSearchCV(estimator=clf, param_grid=candidate_params, cv=kf_, n_jobs=4)
        gs.fit(train_X, train_y)
        if over:
            classifier = pipe([('sm', SMOTE(k_neighbors=5, random_state=0)),
                               ('estimator', gs.best_estimator_)])  # SMOTEしている
        else:
            classifier = pipe([('estimator', gs.best_estimator_)])
    elif model == 2:
        print('SVM')  # ここも本来はチューニング可能
        clf = SVC(kernel='linear', probability=True, random_state=0)
    elif model == 3:
        print('KNeighbors')
        clf = KNeighborsClassifier()
        std_scl = StandardScaler()
        # パラメータチューニング
        candidate_params = {'n_neighbors': [1, 2, 3, 4, 5]}
        kf_ = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        gs = GridSearchCV(estimator=clf, param_grid=candidate_params, cv=kf_, n_jobs=4)
        gs.fit(train_X, train_y)
        if over:
            classifier = pipe([('transformer', std_scl),
                               ('sm', SMOTE(k_neighbors=5, random_state=0)),
                               ('estimator', gs.best_estimator_)])
        else:
            classifier = pipe([('transformer', std_scl),
                               ('estimator', gs.best_estimator_)])
    elif model == 4:
        print('Naive Bayes')
        clf = GaussianNB()
    if model == 2 or 4:
        std_scl = StandardScaler()
        if over:
            classifier = pipe([('transformer', std_scl),
                               ('sm', SMOTE(k_neighbors=5, random_state=0)),  # SMOTEしている
                               ('estimator', clf)])
        else:
            classifier = pipe([('transformer', std_scl),
                               ('estimator', clf)])

    return classifier


# 事後確率を得る
def get_proba_list(pred_l, proba, y):
    max_list = []
    proba_list = []
    predict = []
    answer = []

    for pl, pred in zip(proba, pred_l):
        for p, pr in zip(pl, pred):
            max_list.append(max(p))
            proba_list.append(p)
            predict.append(pr)
    for yi in y:
        answer.append(yi.tolist())
    answer = sum(answer, [])

    return max_list, proba_list, predict, answer


# 混同行列とclassification_reportを表示させる
def cm_classification_report(answer, predict, model, val, target_names, cm_name):
    # ConfusionMatrix
    cm = confusion_matrix(answer, predict)
    cm = pd.DataFrame(data=cm, index=target_names, columns=target_names)
    sns.set(font_scale=5)
    fig = plt.figure(figsize=(15, 12))
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues', fmt="d")
    plt.yticks(rotation=0)
    plt.ylabel("Grand Truth", fontsize=40)
    plt.xlabel("Prediction", fontsize=40)
    # fig.set_ylim(len(cm), 0)
    # plt.savefig(cm_name + str(model) + '_' + val + '_normed' + '.png')
    now = datetime.datetime.now()
    plt.savefig(cm_name + str(model) + '_' + val + '_' + now.strftime('%Y%m%d_%H%M%S') + '_' + '.png')
    print(classification_report(answer, predict, target_names=target_names, digits=3))


#### 個人特化 ####
# 平準化を行う
def standardization_p(l, f_num):
    l_mean = np.zeros(f_num)
    l_pstdev = np.zeros(f_num)
    for j in range(0, f_num):
        l_mean[j] = statistics.mean(l[:, j])
        l_pstdev[j] = statistics.pstdev(l[:, j])
        l[:, j] = [(l_i - l_mean[j]) / l_pstdev[j] for l_i in l[:, j]]
    return l, l_mean, l_pstdev


def normalization(l):
    norm_result = np.array([(l_i - statistics.mean(l)) / statistics.pstdev(l) for l_i in l])
    return norm_result


# 相関係数として類似度を算出
def calc_similarity(sim_file_name, target_names, normalize, sub_, sub_num, train_df, test_df):
    sub_ = [element for element in sub_ if element != sub_num]
    for state in target_names:
        if normalize:
            lst_bins = [round(-4 + i / 100, 2) for i in range(0, 12 * 100 + 1)]
        else:
            lst_bins = [round(0 + i / 100, 2) for i in range(0, 8 * 100 + 1)]
        res_df = pd.DataFrame()
        res_lst = []
        # テストユーザのヒストグラム
        lst_test = np.array(test_df.loc[test_df['target'] == state, 'saccade vel mean'].tolist())
        lst_test = [round(lst_test[ni], 2) for ni in range(len(lst_test))]
        print('lst_test: ')
        print(lst_test)
        count_test = [lst_test.count(i) for i in lst_bins]
        hist_test = pd.Series([i / sum(count_test) for i in count_test])

        # トレーニングユーザのヒストグラム
        for a in sub_:  # ここは訓練データセットの人（元の人数＋ループ学習で増えたデータセット）
            lst_ = np.array(train_df.loc[train_df['participant'] == a, 'saccade vel mean'].tolist())  # a番目の人の全部のデータ
            if normalize:
                lst_ = normalization(lst_)
                train_df.loc[train_df['participant'] == a, 'saccade vel mean'] = lst_
            else:
                pass
            lst_train = np.array(train_df.loc[(train_df['participant'] == a) & (train_df['target'] == state), 'saccade vel mean'].tolist())
            # print(m, a, min(lst), max(lst))
            lst_train = [round(lst_train[ni], 2) for ni in range(len(lst_train))]
            count_train = [lst_train.count(i) for i in lst_bins]
            hist_train = pd.Series([i / sum(count_train) for i in count_train])  # これがヒストグラムを表すリスト
            # 類似度算出
            res = hist_train.corr(hist_test)
            res_lst.append(res)
        res_df[str(state)] = res_lst
        # ここで類似度を１つにまとめる
        res_df['final'] = res_df.sum(axis=1) / 3 * (1 + res_df.std(axis=1))
    if not res_df.empty:
        res_df.to_excel(sim_file_name, sheet_name='sub'+str(sub_num), index=False)
