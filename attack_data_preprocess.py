import pandas as pd
from imblearn.over_sampling import SMOTE


# 读取数据
df = pd.read_csv('./DoS_dataset.csv',names=['time_stamp','CAN_ID','DLC']+['DATA['+str(i)+']' for i in range(8)]+['Label'])

# 缺失值补0
for r in range(len(df.index)):
    if df.loc[r,'DLC'] < 8:
        for i in range(8):
            if df.loc[r,'DATA['+str(i)+']'] == 'R' or df.loc[r,'DATA['+str(i)+']'] == 'T':
                df.loc[r,'Label']=df.loc[r,'DATA['+str(i)+']']
                df.loc[r, 'DATA[' + str(i) + ']'] = '00'
                break

df = df.fillna('00')

print(df)




# 提取timestamp特征值做成dataframe
# timestamp_list = []
# for timestamp in time_stamp_series:
#     timestamp_list.append([(timestamp.split(': '))[1]])
#
# timestamp_df = pd.DataFrame(data=timestamp_list,columns=['time_stamp'])
#
# # 拼接得到最终处理后的数据集
# df = pd.concat([timestamp_df,other_data_df],axis=1)
# print(df.info())
# df['CAN_ID'] = df['CAN_ID'].fillna('0000')
# df['DLC'] = df['DLC'].fillna(0)
# df['DATA_FILED'] = df['DATA_FILED'].fillna('00'+' 00'*7)
# print(df)
#

# 处理timestamp强关联,顺便对CANID作进制转换
for r in reversed(range(len(df.index))):
    if r == 0:
        df.iloc[r,0] = 0.0
    else:
        df.iloc[r,0] = float(df.iloc[r,0]) - float(df.iloc[r-1,0])

    df.iloc[r,1] = int(df.iloc[r,1],16)
# df.to_csv('normal_run_data.csv')
# print(df)
print(df)
print(df.columns)

# 处理DATAFILED，便于处理成特征向量
# datafiled_list = [datafiled.split(' ') for datafiled in df['DATA_FILED']]
# datafiled_df = pd.DataFrame(data=datafiled_list,columns=['DATA'+'['+str(i)+']' for i in range(8)])
# df.drop(columns='DATA_FILED',inplace=True)
# df = pd.concat([df,datafiled_df],axis=1)
# print(df)

print(df.info())

# 对DATAFILE作进制转换
df.loc[:,['DATA['+str(i)+']' for i in range(8)]] = df.loc[:,['DATA['+str(i)+']' for i in range(8)]].applymap(lambda x:int(x,16))
print(df)
print(df.Label.value_counts())
oversampe = SMOTE(random_state=0)
x, y = oversampe.fit_resample(df.drop(['Label'],axis=1),df['Label'])
print(x)
print('-------------------------------------')
print(y)

df = pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis=1)
print(df)
print(df.Label.value_counts())

df.to_csv('processed_Dos_data.csv')