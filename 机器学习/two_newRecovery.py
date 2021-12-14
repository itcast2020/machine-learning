import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = '.\DATA1\data_without_province_update_20200226.csv'

data2 = pd.read_csv(path,header=0, names=['new_deaths','new_confirmed','new_recoveries'],usecols=[4,5,6])
data2.head()
print(data2)

# 特征归一化(为了使梯度下降得更方便)
#data2 = (data2 - data2.mean()) / data2.std()  # 取平均，再除以m
data2.head()


data2.insert(0, 'Ones', 1)  # 这里是为了统一梯度下降函数的方便，一个表达式搞定，乘X0即*1

def data():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
            xs = data2["new_confirmed"]
            ys = data2["new_deaths"]
            zs = data2["new_recoveries"]
            ax.scatter(xs, ys, zs, c=c, marker=m)

        ax.set_xlabel('new_confirmed')
        ax.set_ylabel('new_deaths')
        ax.set_zlabel('new_recoveries')
        plt.show()

data()

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
y = y.values.reshape(48, 1)
print(y.shape)
theta = np.zeros((3, 1))
print("**************************************")
# 选取不同的学习率
candidate_alpha = [0.0003, 0.003, 0.03, 0.0001, 0.001, 0.01]
fig, ax = plt.subplots()

for alpha in candidate_alpha:
    theta, costs = gradientDescent(X, y, theta, alpha)
    print("*******theta")
    print(theta)
    print("________________________________")
    ax.plot(np.arange(10000), costs, label=alpha)
    ax.legend()
    ax.set(xlabel='iters',
           ylabel='costs',
           title='cost vs iters')

plt.show()

path1='.\DATA1\data_without_province_update_20200302.csv'
data1 = pd.read_csv(path1,header=0, names=['new_deaths','new_confirmed','new_recoveries'],usecols=[4,5,6])
data1.head()
print(data1)
data1.insert(0, 'Ones', 1)
cols = data1.shape[1]
ccc=data1.shape[0]
X1 = data1.iloc[ccc-3:ccc, :cols-1].values
print(X1)
y1 = data1.iloc[ccc-3:ccc, cols-1].values
print("************************")
print(y1)
print("**********************")
y_pre=hansh(X1,theta)

print(y_pre)














