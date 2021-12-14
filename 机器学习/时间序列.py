import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM, SimpleRNN,GRU,BatchNormalization
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot') # 将背景颜色改为。。

from datetime import datetime,date,timedelta
import numpy as np
import os
for dirname, _, filenames in os.walk('.\DATA1'):
    for filename in filenames:
        print(os.path.join(dirname,filename))

#数据可视化
df = pd.read_csv('.\DATA1\data_without_province_update_20200227.csv',header=0)
df.head()

df1 = df.rename({"Unnamed: 0":"ds"},axis="columns")
df1['ds'] = pd.to_datetime(df1['ds'])
df1['num'] = df1.index  #下标
df1.tail()

print(df1)

# 标识出时期内的new_confirmed，new_recoveries
fig,axes = plt.subplots(1,2,figsize=(16,8),dpi=80)
axes[0].scatter(df1['ds'],df1['new_confirmed'],s=15,c='green')
axes[1].scatter(df1['ds'],df1['new_recoveries'],s=15,c='red')
axes[0].set_ylabel('New Confirmed cases')
axes[1].set_ylabel('New Recoveried cases')
axes[0].set_title('Up to 2/29 new confirmed cases')
axes[1].set_title('Up to 2/29 new recoveried cases')
for i in range(0,2):
    axes[i].set_xticklabels(df1['ds'],rotation=60)
plt.show()

# 选取减掉最后8行的new_recoveries的数据
training_set = df1.loc[:len(df1) - 8,['new_recoveries']].values
print("****************************")
print(training_set)
print("****************************")
test_set = df1.loc[len(df1) - 8:, ['new_recoveries']].values
print("****************************")
print(test_set)
print("****************************")



# 归一化
sc = MinMaxScaler(feature_range=(0, 1)) # MinMaxScaler（将数据预处理为(0,1)上的数
# 注意训练数据要使用fit_transform，因为要找到转换规则
# fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
training_set_scaled = sc.fit_transform(training_set)
# 测试数据不要fit过程，与训练数据使用一套参数，保持一致性
test_set = sc.transform(test_set)



#LSTM时间序列分析
x_train = []
y_train = []

x_test = []
y_test = []

# 前3天的数据当做输入x,第3+1天数据当作目标y
for i in range(3, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 3:i, 0])
    y_train.append(training_set_scaled[i, 0])

#提取数据集
x_train, y_train = np.array(x_train), np.array(y_train)
print('x_train:', x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print('x_train:', x_train)
print('x_train:', x_train.shape)



for i in range(3, len(test_set)):
    x_test.append(test_set[i - 3:i, 0])
    y_test.append(test_set[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))



#模型构建
model = tf.keras.Sequential([
    LSTM(64, activation='relu', return_sequences=False, unroll=True),
    Dense(1)
])
model.compile(optimizer='adam',
              loss='mean_squared_error')


#模型训练
checkpoint_save_path = "./checkpoint/nCov.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 # save_best_only=True,
                                                 verbose=1)

history = model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback], verbose=1)
model.summary()


#损失可视化
file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#预测结果
predicted_train_data = model.predict(x_train)
predicted_train_data = sc.inverse_transform(predicted_train_data)
real_train_data = df.loc[3:3+len(predicted_train_data),['new_confirmed']].values
train_date = df.iloc[3:3+len(predicted_train_data),0:1].values

predicted_test_data = model.predict(x_test)
# 对预测数据还原。
predicted_test_data = sc.inverse_transform(predicted_test_data)

print(predicted_test_data)
print(df1['new_recoveries'][-len(predicted_test_data):])



'''#多项式回归分析
degree = 2
df1['num'] = np.arange(len(df1))
x1 = np.array(df1['num'])
y1 = np.array(df1['new_recoveries']).astype(np.int32)
#计算参数
f1 = np.polyfit(x1,y1,degree)
#计算拟合值
xDate = df1['ds'].values
yvals = np.polyval(f1,x1)
plt.plot(xDate,y1,'s',color='green')
plt.plot(xDate,yvals,'r',color='red')
plt.title("Polynomial Regression with degree = {}".format(degree))
plt.ylabel('New recoveries cases')
plt.xlabel('Date')
plt.show()
'''