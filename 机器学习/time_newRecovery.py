import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('.\DATA1\data_without_province_update_20200226.csv',header=0)
df.head()

df1 = df.rename({"Unnamed: 0":"ds"},axis="columns")
df1['ds'] = pd.to_datetime(df1['ds'])
df1['num'] = df1.index
df1.tail()
print(df1)
degree = 2
df1['num'] = np.arange(len(df1))

x2=np.array(df1['total_recoveries'])
x1 = np.array(df1['num'])
y1 = np.array(df1['new_recoveries']).astype(np.int32)
#计算参数S
f1 = np.polyfit(x1,y1,degree)
#计算拟合值
xDate = df1['ds'].values
yvals = np.polyval(f1,x1)


plt.plot(x2,y1,'s',color='green')
#plt.plot(xDate,yvals,'r',color='red')
plt.title("Polynomial Regression with degree = {}".format(degree))
plt.ylabel('New recoveries cases')
plt.xlabel('Date')
plt.show()

hh=df1.shape[1]
x_test=df1.loc[len(df1) - 8:,['num']]
y_test=df1.loc[len(df1) - 8:,['new_recoveries']]
print(y_test)
y_pre=np.polyval(f1,x_test)
print(y_pre)
