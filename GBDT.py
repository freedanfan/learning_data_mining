import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 导入sklearn包中的函数
from sklearn.model_selection import train_test_split
# 导入网格搜索法的函数
from sklearn.model_selection import GridSearchCV

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

# # 导入GBDT模型的类
from sklearn.ensemble import GradientBoostingClassifier
# 构建GBDT模型
gbdt = GradientBoostingClassifier()     # 默认参数构建模型
gbdt.fit(x_train, y_train)
print(gbdt)

# K近邻模型在测试集上的预测
gbdt_pred = gbdt.predict(x_test)
print(pd.crosstab(gbdt_pred, y_test))

# 模型得分
print("模型在训练集上的准确率%f", gbdt.score(x_train, y_train))
print("模型在测试集上的准确率%f", gbdt.score(x_test, y_test))

# 是绘制ROC曲线，并计算曲线下的面积AUC值
# 导入模型评估模块
from sklearn import metrics
import os
save_path = "./images"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 计算ROC曲线的X轴和Y轴数据
fpr, tpr, _ = metrics.roc_curve(y_test, gbdt.predict_proba(x_test)[:, 1])
# 绘制ROC曲线
plt.plot(fpr, tpr, 'r-')
# 添加阴影
plt.stackplot(fpr, tpr, colors='r')
# 绘制参考线
plt.plot([0, 1], [0, 1], 'b--')
# 往图中添加文本
plt.text(0.6, 0.4, 'AUC=%.3F' % metrics.auc(fpr, tpr), fontdict=dict(size=18))
plt.savefig('./images/gbdt_AUC.png')
plt.show()

# GBDT模型的网格搜索法
# 选择不同的参数
learning_rate_option = [0.01, 0.05, 0.1]
max_depth_options = [3, 5, 7, 9]
n_estimators_options = [100, 300, 500]
parameters = {'learning_rate': learning_rate_option, 'max_depth': max_depth_options,
              'n_estimators': n_estimators_options}
grid_gbdt = GridSearchCV(estimator= GradientBoostingClassifier(), param_grid= parameters,
                         cv=10, scoring='accuracy')
grid_gbdt.fit(x_train, y_train)

# 结果输出
print(grid_gbdt.best_score_, grid_gbdt.best_params_, grid_gbdt.best_score_)


# 预测测试集
grid_gbdt_pred = grid_gbdt.predict(x_test)
print(pd.crosstab(grid_gbdt_pred, y_test))

# 模型得分
print("网格搜素：模型在训练集上的准确率%f", grid_gbdt.score(x_train, y_train))
print("网格搜素：模型在测试集上的准确率%f", grid_gbdt.score(x_test, y_test))

# 绘制ROC曲线
fpr, tpr, _ = metrics.roc_curve(y_test, grid_gbdt.predict_proba(x_test)[:, 1])
plt.plot(fpr, tpr, 'r-')
# 添加阴影
plt.stackplot(fpr, tpr, colors='r')
# 绘制参考线
plt.plot([0, 1], [0, 1], 'b--')
# 往图中添加文本
plt.text(0.6, 0.4, 'AUC=%.3F' % metrics.auc(fpr, tpr), fontdict=dict(size=18))
plt.savefig('./images/grid_gbdt_AUC.png')
plt.show()

