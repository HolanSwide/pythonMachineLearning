import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 特征工程
def preProcess(pro_data):
    # 年龄划分
    pro_data['age'] = pro_data['age'].map(lambda x: 'young' if x < 30 else 'middle' if x < 60 else 'elder')

    # 获取特征名
    tags = pro_data.columns.values.tolist()

    # 用于保存数据原特征名
    data_features = {}

    # 筛选字符串类型
    for tag in tags.copy():
        if pro_data[tag].dtypes != object:
            tags.remove(tag)

    # 特征化
    for tag in tags:
        features = []
        val = np.array(pro_data[tag].values)
        for i in val:
            if i in features:
                continue
            else:
                features.append(i)
        for i in features:
            val[val == i] = features.index(i)
        pro_data[tag] = val
        data_features[tag] = features

    return pro_data, data_features


# 数据清洗
def clean(pre_data):
    """
    删除部分特征列：
        month day_of_week duration pdays lending_rate3m nr_employed
    """
    pre_data = pre_data.drop(['month', 'day_of_week', 'duration', 'pdays', 'lending_rate3m', 'nr_employed'], axis=1)
    """
    缺失值处理:
        查找存在缺失值的列，填充以上下的平均值
    """
    null_list = pre_data.isnull().sum()
    for i in null_list.index:
        if null_list[i] != 0:
            pre_data[i] = pre_data[i].interpolate()
    return pre_data


# 绘制图表
def drawPics(pic_data):
    plt.figure(dpi=400, figsize=(16, 9))

    # 客户职业与是否购买
    pic = sns.countplot(x='job', hue='subscribe', data=pic_data)
    pic.set(title='job - subscribe')
    plt.savefig("./pics/job-subscribe.png")

    # 客户年龄与是否购买
    pic = sns.countplot(x='age', hue='subscribe', data=pic_data)
    pic.set(title='age - subscribe')
    plt.savefig("./pics/age-subscribe.png")


# 加载、处理和划分数据集
def getDataset():
    # 加载数据
    train = pd.read_csv('./train.csv')

    # 数据处理
    data_res = preProcess(train)
    real_data = data_res[0]
    features = data_res[1]

    # 数据清洗
    real_data = clean(real_data)

    # 绘制图表
    # drawPics(data[0])

    # 数据分割
    y = real_data['subscribe']
    real_data.drop('subscribe', axis=1, inplace=True)
    x = real_data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    return x_train.astype('int'), x_test.astype('int'), y_train.astype('int'), y_test.astype('int'), features


def knnModel(model_data, n):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(model_data[0], model_data[2])
    # 预测结果
    # y_predict = knn.predict(data[1])
    # 准确率
    score = knn.score(model_data[1], model_data[3])
    return knn, score


def testData():
    """
    测试数据处理
    """
    # 加载数据
    test = pd.read_csv('./test.csv')

    # 数据处理
    id_col = test['id']
    test_res = preProcess(test)
    testset_data = test_res[0]
    features = test_res[1]

    # 数据清洗
    testset_data = clean(testset_data)

    # 标准化
    std = StandardScaler()
    testset_data = std.fit_transform(testset_data)

    return testset_data, features, id_col


if __name__ == '__main__':
    # 训练
    data = getDataset()
    res = knnModel(data, 20)
    print("> 准确率为：", res[1])
    # 测试
    model = res[0]
    test_data = testData()
    pre_y = model.predict(test_data[0])
    # 测试结果转化与保存
    index_0 = (pre_y == 0)
    index_1 = (pre_y == 1)
    pre_y = pre_y.astype('object')
    pre_y[index_0] = data[4]['subscribe'][0]
    pre_y[index_1] = data[4]['subscribe'][1]
    df = DataFrame({'id': test_data[2], 'subscribe': pre_y})
    df.to_csv("./res/submission.csv", index=False)

    print("============== end ==============")
