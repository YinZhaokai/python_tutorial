#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   tf2_test.py
@Time    :   2020/02/18 19:33:05
@Author  :   Zhaokai Yin
@Version :   0.1
@Contact :   yinzhaokai@foxmail.com
@License :   (C)Copyright 2017-2020
'''

# here put the import lib
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from configparser import ConfigParser
from collections import namedtuple
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib


def read_data():
    """[读取原始数据。将读取原始数据拆出来，这样在修改配置文件后再计算的时候就不需要重复读取数据了]
    """
    df = pd.read_csv('SanXia1Hour.csv', sep=',')
    return df

def df_split(df,flag = 0):
    """[将数据集划分为训练集和测试集]
    
    Arguments:
        df {[DataFrame]} -- [未划分的数据集]
        flag {[int]} -- [数据集划分方式，0为按照比例划分，1为按照下标划分]
    """
    if flag == 0:
        df_train, df_test = train_test_split(df, test_size=0.4, shuffle=False)

    if flag == 1:
        df_train = df[30688:92056]
        df_test = df[92056:135880]

    return df_train, df_test


def normalization(train_data, test_data, obs_i):
    """[训练及测试数据归一化]
    
    Arguments:
        train_data {[ndarray]} -- [训练数据集]
        test_data {[ndarray]} -- [测试数据集]
    
    Returns:
        nor_train_data [ndarray] -- [归一化后的训练数据集]
        nor_test_data [ndarray] -- [归一化后的测试数据集]
        test_mean [float] -- [训练集上出库流量均值]
        test_std [float] -- [训练集上出库流量标准差]
    """
    
    train_mean = np.mean(train_data[:, obs_i])  # 求出库均值
    train_std = np.std(train_data[:, obs_i])  # 求出库标准差
    nor_train_data = preprocessing.scale(train_data)  # 将数据归一化，均值为0，标准差为1

    test_mean = train_mean
    test_std = train_std
    #每一列数据在训练集上的标准差
    std_array = train_data.std(axis = 0)
    #因为训练集每列的标准差将要作为测试集标准化计算时的分母，因此先排除可能为0的项
    for i in range(std_array.size):
        if std_array[i] < 0.000001 and std_array[i] > -0.000001:
            std_array[i] = 0.01

    #使用训练集的均值和标准差来标准化测试集，未来预报的时候也使用训练集的均值和标准差来做，保证统一
    nor_test_data = (test_data - train_data.mean(axis=0)) / std_array

    return nor_train_data, nor_test_data, train_mean, train_std


def train_current_qout(hps, data, data_test, test_mean, test_std, config, obs_i):
    """[模拟当前时刻流量]
    
    Arguments:
        hps {[namedtuple]} -- [LSTM超参数]
        data {[ndarray]} -- [归一化训练集数据]
        data_test {[ndarray]} -- [归一化测试集数据]
        test_mean {[float]} -- [预报目标均值]
        test_std {[float]} -- [预报目标方差]
        config {[type]} -- [配置文件实例]
        obs_i {[int]} -- [预报目标在数据的第几列]
    """

    X = []  # 训练输入
    X_test = []  # 测试输入
    y = []  # 训练输出
    y_test = []  # 测试输出
    data_len = len(data)  # 训练集长度
    data_test_len = len(data_test)  # 测试集长度
    n_inputs = data.shape[1]  # 输入信息的维度

    #将训练输入、输出数据存入X,Y
    for i in range(data_len - hps.seq_size):
        end = i + hps.seq_size
        y.append(data[end-1, obs_i].tolist())
        y1 = data[end, obs_i]
        #将当前时刻的出库设定为0，作为网络输入
        temp = data[i: end].tolist()
        temp[-1][obs_i] = 0
        X.append(temp)

    #将测试输入、输出数据存入X_test、y_test
    for i in range(data_test_len - hps.seq_size - 1):
        end = i + hps.seq_size
        y_test.append(data_test[end-1, obs_i].tolist())
        temp = data_test[i: end].tolist()
        temp[-1][obs_i] = 0
        X_test.append(temp)


    y = np.array(y, dtype=np.float64)
    X = np.array(X, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)
    # #打乱训练集的顺序
    # X, y = shuffle(X, y)
    print(X.shape[-2:])
    print(y.shape)

    Datatuple = namedtuple(
        'Datatuple', 'X, y, X_test, y_test, data_len, data_test_len')
    datas = Datatuple(X, y, X_test, y_test, data_len, data_test_len)

    #构建并训练神经网络，返回最终训练集上的MSE及测试集上的MSE和结果
    test_result = LSTM_graph(hps, n_inputs, datas, 't_qOut', config)

    statTuple = namedtuple(
        'statTuple', 'test_result, test_mean, test_std, y_test')
    stats = statTuple(test_result, test_mean, test_std, y_test)
    #验证期结果演示
    result_stat(stats, 't_qOut', hps, config)
    return


def LSTM_graph(hps, n_inputs, datas, name, config):
    #重置神经网络并固定随机种子
    reset_graph(2019)

    #构建LSTM神经网络
    model = tf.keras.Sequential([
        # layers.Embedding(input_dim=n_inputs, output_dim=hps.hidden_size, input_length=hps.seq_size),
        layers.LSTM(hps.hidden_size, input_shape=[
                    hps.seq_size, n_inputs], return_sequences=False, dropout=0.1),
        # layers.LSTM(hps.hidden_size, return_sequences=False),
        layers.Dense(1)
    ])

    print(model.summary())
    model.compile(optimizer='adam', loss="mse")
    #神经网络训练
    history = model.fit(datas.X, datas.y, batch_size=hps.batch_size, epochs=hps.n_iterations, validation_split=0.1, shuffle=True,workers=128,use_multiprocessing=True)
    #绘制训练过程图并保存
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'valivation'], loc='upper left')
    plt.savefig('loss.png')
    plt.close()

    #进行验证期的计算
    predictions = model.predict(datas.X_test)
    #保存神经网络
    model.save('LSTM_TF2_TEST.h5')

    # new_model = tf.keras.models.load_model('LSTM_TF2_TEST.h5')
    # new _predictions = new_model.predict(datas.X_test)


    return predictions


def result_stat(stats, name, hps, config):
    """[验证期结果展示与评价]
    
    Arguments:
        stats {[type]} -- [description]
        name {[type]} -- [description]
        hps {[type]} -- [description]
        config {[type]} -- [description]
    """
    result = [] #还原后的预报结果
    tru = []    #还原后的实测过程
    delta = []  

    fenzi = []  #计算NSE的分子
    fenmu = []  #计算NSE的分母

    for i in range(len(stats.test_result)):
    
        ipre = stats.test_result[i]*stats.test_std+stats.test_mean if (
            stats.test_result[i]*stats.test_std+stats.test_mean) >= 0 else 0
        result.append(ipre[0])
        tru.append(stats.y_test[i]*stats.test_std+stats.test_mean)
        # f.write(str(ipre) + '\t' + str(stats.y_test[i][obs_i]*stats.std_test+stats.mean_test) + '\n')
        dt = abs(ipre-tru[i])
        fz = dt**2
        fm = (tru[i]-stats.test_mean)**2
        fenzi.append(fz)
        fenmu.append(fm)
        delta.append(dt)
    # f.close()
    NSE1 = 1-np.sum(fenzi)/np.sum(fenmu)

    tru1 = np.array(tru)
    result1 = np.array(result)
    diff = result1 - tru1
    #保存结果

    save_matrix = np.concatenate((tru1, result1, diff),axis=0).reshape(3,-1)
    np.savetxt('result.txt',save_matrix.T,fmt='%.8f')

    # write_context = ''
    # for i, i_tru in enumerate(tru1):
    #     write_context += str(i_tru) + '\t' + str(result1[i]) + '\t' + str(diff[i]) + '\n'
    # with open('result.txt', 'w', encoding='utf-8') as f:
    #     f.write(write_context)
    
    NSE = 1-np.sum((tru1-result1)**2)/np.sum((tru1-np.mean(tru1))**2)
    BIAS = (np.sum(result1 - tru1) / np.sum(tru1)) * 100  # /len(result1)
    print('Avg_error is %.3f m3/s, NSE is %.3f , BIAS is %.3f' %
          (np.mean(delta), NSE, BIAS))

    some_plots(tru1, result1)

#绘制验证期预报效果图
def some_plots(tru, result):
    xx = np.array(range(len(result)))

    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 24, }
    matplotlib.rc('font', **font1)
    fig = plt.figure(figsize=(16, 9))
    plt.subplot(111)
    plt.title('prediction skill')
    plt.plot(xx, tru, c='#4169E1', label='Observed', alpha=1, zorder=1)
    plt.scatter(xx, result, c='#CD5C5C', marker='+',
                label='Simulated', alpha=0.8, zorder=2)

    plt.legend(loc="upper right", prop=font1)

    plt.xlabel("Hour", font1)
    plt.ylabel(r'Hourly Outflow  (m$^3$/s)', font1)

    plt.savefig('result.png')
    plt.close(fig)
    

#重置神经网络，并保持随机种子不变
def reset_graph(seed=42):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

def SingleResTF2(df):
    """[单个方案运行的起点，方便循环调用]
    
    Arguments:
        df {[type]} -- [description]
    """
    CONFIGFILE = './config.ini'
    config = ConfigParser()
    config.read(CONFIGFILE)

    #将数据划分为训练集和测试集
    df_train, df_test = df_split(df,1)
    train_data = np.array(
        df_train[['MONTH', 'HOUR', 'QIN', 'QOUT', 'LEVEL', 'TREND']], dtype='float64')
    test_data = np.array(
        df_test[['MONTH', 'HOUR', 'QIN', 'QOUT', 'LEVEL', 'TREND']], dtype = 'float64')
        
    #obs_i指代第几列是QOUT
    obs_i = 3
    #将训练集、测试集数据归一化
    nor_train_data, nor_test_data, test_mean, test_std = normalization(
        train_data, test_data,obs_i)

    # #从配置文件中读取超参数
    seq_size = config.getint('hps', 'seq_size')
    hidden_size = config.getint('hps', 'hidden_size')
    learning_rate = config.getfloat('hps', 'learning_rate')
    n_iterations = config.getint('hps', 'n_iterations')
    batch_size = config.getint('hps', 'batch_size')
    #将变量打包，减少函数传参的数量
    HParams = namedtuple(
        'HParams', 'seq_size, hidden_size, learning_rate, n_iterations, batch_size')
    hps = HParams(seq_size, hidden_size, learning_rate,
                  n_iterations, batch_size)

    #模拟当前时刻流量
    train_current_qout(hps, nor_train_data, nor_test_data, test_mean, test_std, config, obs_i)


def main():
    df = read_data()
    SingleResTF2(df)
    return
if __name__=='__main__':
    main()
