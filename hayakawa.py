import pandas as pd
from decimal import *
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

#========================アンケート結果読み込み=====================#
task02=pd.read_excel('task02.xlsx',index_col=None)#ファイルの読み込みtask02
#=================================================================#
pre_se=[]
post_se=[]
def se_list(someone, file_name):
    for sm in someone:
        for fn in file_name:
        #=================アンケートの結果=================
            index=(task02.index[task02['被験者']==sm])[0]
            pre=fn+'_pre'
            post=fn+'_post'
            pre_se.append(task02.loc[index,pre])
            post_se.append(task02.loc[index,post])
        #===============================================
def data_list(someone, file_name):
    data=[]
    for sm in someone:
        for fn in file_name:
            x=input_data.loc[fn,sm]
            data.append(x)
    return data  

day=['02']
someone = ['hikensya']#02
file_name = ['n_iraira1','n_iraira2','n_iraira3','n_iraira4','n_iraira5','iraira1-1','iraira1-2','iraira2-1','iraira2-2','iraira3-1','iraira3-2','iraira4-1','iraira4-2','iraira5-1','iraira5-2']
se_list(someone,file_name)
#=======================データをデータフレームに==============================#
print("model(1:LR,2:SVR,3:RF):")
model=int(input())
print("sse(1:pre,2:post):")
sse=int(input())

df=pd.DataFrame()
input_data=pd.read_excel('./data/1mean_sacdis.xlsx',index_col=0)#データ
df.loc[:,'sac_dis_mean']=data_list(someone, file_name)
input_data=pd.read_excel('./data/1max_sacdis.xlsx',index_col=0)#データ
df.loc[:,'sac_dis_max']=data_list(someone, file_name)
input_data=pd.read_excel('./data/1min_sacdis.xlsx',index_col=0)#データ
df.loc[:,'sac_dis_min']=data_list(someone, file_name)
input_data=pd.read_excel('./data/1med_sacdis.xlsx',index_col=0)#データ
df.loc[:,'sac_dis_med']=data_list(someone, file_name)
input_data=pd.read_excel('./data/1hensa_sacdis.xlsx',index_col=0)#データ
df.loc[:,'sac_dis_hensa']=data_list(someone, file_name)
input_data=pd.read_excel('./data/1range_sacdis.xlsx',index_col=0)#データ
df.loc[:,'sac_dis_range']=data_list(someone, file_name)

# input_data=pd.read_excel('./data/2lookper_02.xlsx',index_col=0)#データ
# df.loc[:,'lookper']=data_list(someone,file_name)

input_data=pd.read_excel('./data/3mean_itime.xlsx',index_col=0)#データ
df.loc[:,'timegap_mean']=data_list(someone, file_name)
input_data=pd.read_excel('./data/3max_itime.xlsx',index_col=0)#データ
df.loc[:,'timegap_max']=data_list(someone, file_name)
input_data=pd.read_excel('./data/3min_itime.xlsx',index_col=0)#データ
df.loc[:,'timegap_min']=data_list(someone, file_name)
input_data=pd.read_excel('./data/3med_itime.xlsx',index_col=0)#データ
df.loc[:,'timegap_med']=data_list(someone, file_name)
input_data=pd.read_excel('./data/3hensa_itime.xlsx',index_col=0)#データ
df.loc[:,'timegap_hensa']=data_list(someone, file_name)
input_data=pd.read_excel('./data/3range_itime.xlsx',index_col=0)#データ
df.loc[:,'timegap_range']=data_list(someone, file_name)

input_data=pd.read_excel('./data/4mean_kakudo.xlsx',index_col=0)#データ
df.loc[:,'kakudo_mean']=data_list(someone,file_name)
input_data=pd.read_excel('./data/4max_kakudo.xlsx',index_col=0)#データ
df.loc[:,'kakudo_max']=data_list(someone,file_name)
input_data=pd.read_excel('./data/4min_kakudo.xlsx',index_col=0)#データ
df.loc[:,'kakudo_min']=data_list(someone,file_name)
input_data=pd.read_excel('./data/4med_kakudo.xlsx',index_col=0)#データ
df.loc[:,'kakudo_med']=data_list(someone,file_name)
input_data=pd.read_excel('./data/4hensa_kakudo.xlsx',index_col=0)#データ
df.loc[:,'kakudo_hensa']=data_list(someone,file_name)
input_data=pd.read_excel('./data/4range_kakudo.xlsx',index_col=0)#データ
df.loc[:,'kakudo_range']=data_list(someone,file_name)
if sse==1:
    df.loc[:,'sse']=pre_se
else:
    df.loc[:,'sse']=post_se

# (df.corr()).to_excel("./i_corr.xlsx")
K=5
features=18 
scores_r2=[]
scores_mae=[]
x=df.iloc[:,0:features]
y=df.iloc[:,features]
kf = KFold(n_splits= K,shuffle=True,random_state=1)

if model==1:
    importances_l=[0]*(features-1)
else:
    importances_l=[0]*features
print(len(x),len(y))
scaler = StandardScaler()

for train_index,test_index, in kf.split(x):
    train_x = x.iloc[train_index] 
    test_x  = x.iloc[test_index]
    train_y = y.iloc[train_index]
    test_y  = y.iloc[test_index]
    # print(train_x)
    if model==1:
        train_x=train_x.drop('sac_dis_range', axis=1)
        test_x=test_x.drop('sac_dis_range', axis=1)
        train_x_std = scaler.fit_transform(train_x)
        test_x_std = scaler.transform(test_x)
        lr=LinearRegression()   
        lr.fit(train_x_std,train_y)
        pred = lr.predict(test_x_std)
        fti = lr.coef_ #重要度算出
        print(fti)
    elif model==2:
        train_x_std = scaler.fit_transform(train_x)
        test_x_std = scaler.transform(test_x)
        svr=SVR(kernel='linear', C=1, epsilon=0.1, gamma='auto')#'rbf'
        svr.fit(train_x_std,train_y)
        pred = svr.predict(test_x_std)
        fti = abs(svr.coef_[0]) #重要度算出
    else:  
        rf=RandomForestRegressor(random_state=1)
        rf.fit(train_x,train_y)
        pred = rf.predict(test_x)
        fti = rf.feature_importances_   
    
    score = r2_score(test_y, pred)
    scores_r2.append(score)
    score = mean_absolute_error(test_y, pred)
    scores_mae.append(score)
    # print("fti",fti)
    importances_l = [x+y for (x, y) in zip(fti, importances_l)]
importances_l = list(map(lambda x: x / 5, importances_l))
    # print(importances_l)
print("FTI")
for i, feat in enumerate(train_x):
        print('\t{0:15s} : {1:>.10f}'.format(feat, importances_l[i]))

print(scores_r2)
print('平均スコア',np.mean(scores_r2))
print(scores_mae)
print('平均スコア',np.mean(scores_mae))

# print(importances_l)
# x = range(0, features)
# plt.bar(x,importances_l)
# plt.show()
# scores = cross_val_score(lr,x,y,cv=kf,scoring="neg_mean_absolute_error")
# print(scores)
# print ("Mean score: {} (+/-{})".format( np.mean (scores), sem(scores)))
# scores = cross_val_score(lr,x,y,cv=kf,scoring="r2")
# print(scores)
# print ("r2: {} (+/-{})".format( np.mean (scores), sem(scores)))
