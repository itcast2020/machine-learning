import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = '.\DATA1\data_without_province_update_20200226.csv'
data2 = pd.read_csv(path,header=1,names=['total_confirmed','total_recoveries','total_deaths','new_deaths','new_confirmed','new_recoveries'],usecols=[1,2,3,4,5,6])
data2.head()
print(data2)

# 特征归一化(为了使梯度下降得更方便)
data2 = (data2 - data2.mean()) / data2.std()  # 取平均，再除以m
data2.head()

data2.insert(0, 'Ones', 1)  # 这里是为了统一梯度下降函数的方便，一个表达式搞定，乘X0即*1

def hansh(X1,theta1):
    return  X1 @ theta1

# 定义代价函数
def computeCost(X, y, theta):
    inner = ((X @ theta) - y) ** 2
    return np.sum(inner) / (2 * len(X))

# 一般代价函数不会变，但是随着表达式的变化，梯度下降函数会变化，就不一定是乘X了
def gradientDescent(X, y, theta, alpha):
    costs = []
    for i in range(10000):
        theta = theta - (X.T @ (X @ theta - y)) * alpha / len(X)
        cost = computeCost(X, y, theta)
        costs.append(cost)
    return theta, costs


# 获取设置数据
cols = data2.shape[1]  # 读取矩阵的第二个维度
hh = data2.shape[0]
print(cols)
print("**********************X")
X = data2.iloc[:, :cols-1].values
print(X)
y = data2.iloc[:, cols-1]
print(y)
y = y.values.reshape(47, 1)
print(y.shape)
theta = np.zeros((6, 1))
print("**************************************")
# 选取不同的学习率
candidate_alpha = [0.0003, 0.003, 0.03, 0.0001, 0.001, 0.01]
#candidate_alpha = [0.01]
fig, ax = plt.subplots()


for alpha in candidate_alpha:
    theta1, costs = gradientDescent(X, y, theta, alpha)
    print("*******theta")
    print(theta1)
    theta2=theta1
    print("*******theta")
    print(theta2)
    print("________________________________")
    ax.plot(np.arange(10000), costs, label=alpha)
    ax.legend()
    ax.set(xlabel='iters',
           ylabel='costs',
           title='cost vs iters')

plt.show()




path1='.\DATA1\data_without_province_update_20200302.csv'
data = pd.read_csv(path1,header=1, names=['total_confirmed', 'total_recoveries','total_deaths','new_recoveries','new_deaths','new_confirmed'],usecols=[1,2,3,4,5,6])
data.head()

#data = (data - data.mean()) / data.std()  # 取平均，再除以m
data.insert(0, 'Ones', 1)
cols = data.shape[1]
ccc=data.shape[0]
X1 = data.iloc[ccc-3:ccc, :cols-1].values
print(X1)
y = data.iloc[ccc-3:ccc, cols-1]
print("y********")
print(y)
print("**********************")

y_pre=hansh(X1,theta2)

b=np.matrix([[2505.98],[2788.08], [2755.76]])
a=[[2505.98],
       [2788.08],
       [2755.76]]
print(b)











