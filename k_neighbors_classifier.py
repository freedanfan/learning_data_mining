import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 导入sklearn包中的函数
from sklearn.model_selection import train_test_split

income = pd.read_csv("datasets/adult.csv")
income.apply(lambda x:np.sum(x.isnull()))

# 查看数据的缺失情况
table_null = income.apply(lambda x:np.sum(x.isnull()))

# 离散型变量的统计描述
object_des = income.describe(include=['object']).to_string()
print(object_des)


for feature in income.columns:
    print("feature={}, dtype={}".format(feature, income[feature].dtype))
    if income[feature].dtype == "object":
        income[feature] = pd.Categorical(income[feature]).codes

income.drop(["education", "fnlwgt"], axis=1, inplace=True)
print("\n\n")
for feature in income.columns:
    print("feature={}, dtype={}".format(feature, income[feature].dtype))
print(income.tail())

x_train, x_test, y_train, y_test = train_test_split(income.drop('income', axis=1),
                                                    income['income'], train_size= 0.75,
                                                    random_state= 1234)
print("训练数据集共有{}条观测".format(x_train.shape[0]))
print("测试数据集共有{}条观测".format(x_test.shape[0]))

# # 导入K近邻模型的类
from sklearn.neighbors import KNeighborsClassifier
#
# 构建K近邻模型
kn = KNeighborsClassifier() # 默认参数构建模型
kn.fit(x_train, y_train)
print(kn)

# 导入网格搜索法的函数
from sklearn.model_selection import GridSearchCV
# k近邻模型的网格搜索法
# 选择不同的参数
k_options = list(range(1,12))
parameters = {"n_neighbors": k_options}
# 搜索不同的k值
grid_kn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=10,scoring='accuracy')
grid_kn.fit(x_train, y_train)
# 结果输出
print(grid_kn.best_score_, grid_kn.best_params_, grid_kn.best_score_)

# K近邻模型在测试集上的预测
kn_pred = kn.predict(x_test)
print(pd.crosstab(kn_pred, y_test))

# 模型得分
print("模型在训练集上的准确率%f", kn.score(x_train, y_train))
print("模型在测试集上的准确率%f", kn.score(x_test, y_test))

# 是绘制ROC曲线，并计算曲线下的面积AUC值
# 导入模型评估模块
from sklearn import metrics
# 计算ROC曲线的X轴和Y轴数据
fpr, tpr, _ = metrics.roc_curve(y_test, kn.predict_proba(x_test)[:, 1])
# 绘制ROC曲线
plt.plot(fpr, tpr, 'r-')
# 添加阴影
plt.stackplot(fpr, tpr, colors='r')
# 绘制参考线
plt.plot([0, 1], [0, 1], 'b--')
# 往图中添加文本
plt.text(0.6, 0.4, 'AUC=%.3F' % metrics.auc(fpr, tpr), fontdict=dict(size=18))
plt.savefig('kn_AUC.png')
plt.show()

# 预测测试集
grid_kn_pred = grid_kn.predict(x_test)
print(pd.crosstab(grid_kn_pred, y_test))

# 模型得分
print("网格搜素：模型在训练集上的准确率%f", grid_kn.score(x_train, y_train))
print("网格搜素：模型在测试集上的准确率%f", grid_kn.score(x_test, y_test))

# 绘制ROC曲线
fpr, tpr, _ = metrics.roc_curve(y_test, grid_kn.predict_proba(x_test)[:, 1])
plt.plot(fpr, tpr, 'r-')
# 添加阴影
plt.stackplot(fpr, tpr, colors='r')
# 绘制参考线
plt.plot([0, 1], [0, 1], 'b--')
# 往图中添加文本
plt.text(0.6, 0.4, 'AUC=%.3F' % metrics.auc(fpr, tpr), fontdict=dict(size=18))
plt.savefig('grid_kn_AUC.png')
plt.show()

