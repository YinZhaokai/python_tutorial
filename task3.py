#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   task3.py
@Time    :   2020/02/07 17:33:50
@Author  :   Zhaokai Yin
@Version :   0.1
@Contact :   yinzhaokai@foxmail.com
@License :   (C)Copyright 2017-2020
'''

# here put the import lib
import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdate

def interpolate(filename):
    """[从文件中读取数据并插值]
    
    Arguments:
        filename {[string]} -- [保存数据的文件路径]
    """
    #从txt文件中读取数据，并命名为a-f列，其中a,c,e列为日期，b,d,f列为流量值
    #skiprows读取时要跳过的行
    data1 = pd.read_table(filename, names=["a", "b", "c", "d", "e", "f"], skiprows=[0])

    #将三组数据分别保存在三个DataFrame内
    d1 = pd.DataFrame(data1[["a", "b"]].values, columns=['TM', 'Q'])
    d2 = pd.DataFrame(data1[["c", "d"]].values, columns=['TM', 'Q'])
    d3 = pd.DataFrame(data1[["e", "f"]].values, columns=['TM', 'Q'])
    #将三个DataFrame连接起来
    df1 = pd.concat([d1, d2, d3], ignore_index = True)
    #将时间列的数据转化为时间格式
    df1["TM"] = pd.to_datetime(df1["TM"])
    #将流量值列的数据转化为浮点数格式
    df1["Q"] = pd.to_numeric(df1["Q"])
    #新建一个DataFrame，保存从原始数据起始时间开始到终止时间为止的时间序列，时间间隔为1小时
    helper = pd.DataFrame({'TM': pd.date_range(
        df1['TM'].min(), df1['TM'].max(), freq='1H')})

    #将df1与helper使用“外连接”的方式联合查询（具体原理参考“SQL外连接”），并保存在df2中
    df2 = pd.merge(df1, helper, on = 'TM', how = 'outer').sort_values('TM')
    #将df2备份在df3中，保存的是未插值前的数据表
    #注意必须用dataframe.copy()方法。
    #直接使用df3=df2的话只是给df2起了个别名df3，所有针对df3的操作都会影响df2，反之亦然
    df3 = df2.copy()
    #调用dataframe.interpolate()命令执行插值，method参数代表插值方法，这里选择的是线性插值
    df2['Q'] = df2['Q'].interpolate(method='linear')
    #将插值前（df3）和插值后（df2）的dataframe返回
    return df2,df3




#绘制图线的函数，参数分别为
# Q_inter - 插值后的数据
# Q_origin - 插值前的数据
# Q_scale - 标准化后的数据
# save - 是否保存图像，默认值为False
# 其中前三个均为必填参数，最后一个为可选参数，如果调用时不填的话会按照默认值使用该参数
def plotting(Q_inter, Q_origin, Q_scale, save=False):
    """[绘制流量数据的图]
    
    Arguments:
        Q_inter {[DataFrame]} -- [插值后的数据]
        Q_origin {[DataFrame]} -- [插值前的数据]
        Q_scale {[DataFrame]} -- [标准化后的数据]
    
    Keyword Arguments:
        save {bool} -- [是否保存图像] (default: {False})
    """
    # 开始画图
    # 建立绘图对象，plt.subplots(2)中的2表示在一张图中绘制2个子图
    # plt 代表整幅图的对象，ax代表2个子图构成的list
    fig, ax = plt.subplots(2)
    #设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #根据第一组数据绘制图线，ax[0]代表针对图中的第一幅子图进行绘制
    #plot代表折线图，后面的参数中分别为（横坐标，纵坐标，label=数据标签，lw=线宽，color=颜色）
    ax[0].plot(Q_inter['TM'], Q_inter['Q'],
            label = u"插值后流量", lw = 2, color = 'violet')
    #scatter代表散点图，其余参数相同
    ax[0].scatter(Q_origin['TM'].values, Q_origin['Q'].values,
                  label=u"原始流量", color='blue')
    #给子图设置图例，loc代表图例在子图中的位置
    ax[0].legend(loc='upper right')
    # 给子图设置y轴的名称，可以在$$包围中使用LaTex语法书写数学符号
    ax[0].set_ylabel('$Q(m^3/s)$')
    # 给子图设置标题
    ax[0].set_title(u'流量插值')
    # 我们已经使用时间序列数据作为x轴，因此此处不需要再设置
    # ax[0].set_xticklabels(pd.date_range(
    #     Q_origin['TM'].min(), Q_origin['TM'].max()), rotation=00)
    # 设置x轴时间的显示格式
    ax[0].xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    # 设置x轴时间的显示间隔 mdate.WeekdayLocator(1) 代表每到周一时显示
    ax[0].xaxis.set_major_locator(mdate.WeekdayLocator(1))
    # 设置网格线
    ax[0].grid(color='gray', which='major', linestyle='dashed', linewidth=0.5)

    # 针对第二个子图进行设置
    ax[1].plot(Q_scale["TM"], Q_scale["Q"], label = u"归一化后流量", color = 'violet')
    ax[1].legend(loc = 'upper right')
    ax[1].set_ylabel('$Q(m^3/s)$')
    ax[1].set_title(u'流量归一化')

    # ax[1].set_xticklabels(pd.date_range(
    #     Q_scale['TM'].min(), Q_scale['TM'].max()), rotation=00)
    ax[1].xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax[1].xaxis.set_major_locator(mdate.WeekdayLocator(1))
    ax[1].grid(color = 'gray', which = 'major', linestyle = 'dashed', linewidth = 0.5)
    # 调整两个子图之间的间距
    plt.tight_layout()

    #根据save参数的取值决定是保存图像还是显示图像，两个命令不能同时执行，否则后执行的那个会没有图
    if save:
        plt.savefig('./插值前后比较.png')
    else:
        plt.show()
    plt.close(fig)
    return


def main():
    # 从interpolate()函数中获取插值前后的dataframe，注意插值前的dataframe的时间序列也是连续的
    Q_inter, Q_origin = interpolate("task3_in.txt")
    # 下面两行分别表示显示Q_inter、Q_origin的前10行，可以显著减少输出的数据
    # print(Q_inter.head(10))
    # print(Q_origin.head(10))
    # 从插值后的dataframe中复制一个出来，作为未来要标准化的数据
    Q_scale = Q_inter.copy()
    # 调用preprocessing.scale()命令进行z-score标准化
    Q_scale["Q"] = preprocessing.scale(Q_scale["Q"].values)
    # print(Q_scale.head(10))
    # 调用plotting()函数绘制图线，可选参数save决定是否保存图像
    plotting(Q_inter, Q_origin, Q_scale)
    # 将Q_inter、Q_scale两个dataframe保存在同一个Excel文件的不同Sheet中
    writer = pd.ExcelWriter('task3_out.xlsx')
    Q_inter.to_excel(writer, 'Sheet1')
    Q_scale.to_excel(writer, 'Sheet2')
    return
if __name__=='__main__':
    main()
