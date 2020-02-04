# Python教程

## 写在前面的话

学习深度学习免不了要学习Python，在学习Python的过程中我有如下几个体会，先和大家分享。

* Python在线学习资源丰富，各种教程和问题都可以通过搜索引擎获得，特别是各种博客(如[csdn](https://www.csdn.net/)、[cnblogs](https://www.cnblogs.com/))上面的经验，[github](https://github.com/)上面现成的代码，以及[stackoverflow](https://stackoverflow.com/)上面的问答，非常方便，一定要加以利用。**但python由于历史原因发展为了2.7和3.x两大分支，互不兼容，因此在查询问题的时候一定要注意python版本是否一致**；
* Python现成的“轮子”很多，有很多成熟的工具让我们来使用，很多情况下都不需要写复杂的逻辑；
* 要熟悉命令行的操作方式，在windows上就是cmd或powershell，在linux上就是shell，在线的各种python教程中都会有各种在命令行中输入命令的操作，大家一定要习惯。
* **自主在线查询问题的能力很重要！！** 查询时要注意python的错误信息，这是出bug时查询解决方案的最直接线索。

## 环境配置

接下来就是python及编辑器的安装，强烈推荐从anaconda下载python，里面内置了很多常用的工具包，使用的时候非常方便，[下载网址](https://www.anaconda.com/distribution/#download-section)。
编辑器推荐使用vscode，跟我的环境一致，比较容易解决你们的问题，这里推荐2个教程介绍如何安装和配置环境：[教程一](https://zhuanlan.zhihu.com/p/31417084)、[教程二](https://www.cnblogs.com/schut/p/10346755.html)。
安装好python和vscode之后，请按照[python入门一](https://www.w3school.com.cn/python/index.asp)、[python入门二](https://www.w3cschool.cn/python3/python3-tutorial.html)或[python入门三](https://www.runoob.com/python/python-tutorial.html)来学习python的基本语法。**三个入门教程不尽相同，建议都看一遍！！！** 另外就是如何使用vscode调试python，这里推荐一个[教程](https://zhuanlan.zhihu.com/p/41189402)，调试时的快捷键与Visual Studio相同，F5为调试、运行，F10为单步运行，F11为进入子函数内部。需要注意的是在python代码调试完毕没有bug后，建议直接使用`python ***.py`的命令运行，这样效率会明显高于调试状态。

## vscode界面简介

<img src="https://wx2.sinaimg.cn/mw1024/005YcoSuly1gbjwjnzj45j31hc0svq8g.jpg" referrerpolicy="no-referrer">

## python上的第一个任务-获取当前时间

python中获取系统当前时间的操作可以轻易搜索到，这里借助这个简单的任务来了解python代码的基本写法，[代码文件 task1.py](https://github.com/YinZhaokai/python_tutorial/blob/master/task1.py)。
<!-- 
```python
#在此处引用“包”
import time

#定义函数
def main():
    #直接将信息输出在屏幕/控制台
    print(time.strftime('%Y-%m-%d %H:%M:%S'))

    #将信息保存在“task1_out.txt”文件中，‘w’代表写文件（‘r’代表读文件，‘a’代表向已存在的文件中追加）
    # ‘encoding='utf-8'’指代写入文件时的编码格式，python对文件编码格式非常敏感（主要影响中文），需要注意
    # windows系统默认的编码格式是gbk（ANSI）
    # with结构 代表系统在系统打开文件，并执行完结构内所有命令后，会自动关闭文件
    with open('task1_out.txt', 'w', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    return

#代码运行起点！！
#在这种写法的python代码中，此处应当为程序运行起点，此处之前都应当是函数定义
#例外是某些代码在此处之前“顶格”写，不包含在任何函数定义中，且与def关键字左对齐，通常是定义全局变量，这些变量可以在任意位置被调用
if __name__=='__main__':
    main()
``` -->
样例代码展示了如何用“代码”而非“脚本”的形式写python，这种写法在写较大型的工程时更加合适。

python代码通常包含包引用、函数定义、主函数等部分。

* 在包引用部分，通常使用`import time`这样的写法，但当包的名字比较长，而在正文中又需要多次调用时，通常会起别名，例如`import tensorflow as tf`，这样在下文中调用tensorflow的函数时只需要写`tf.variable()`就可以了。
* 在函数定义部分，需要注意的是python对于不同类型的数据是在传参时是传*值*（在子函数内修改变量值不影响外部函数变量值）还是传*引用*（在子函数修改变量值影响外部函数变量值）的规定是不同的，这个要去多查资料。另外就是返回函数值的方法是直接在`return`后面接要返回的变量，可以返回任意多个，像这样：

```python
def subroutine1(a,b):
    c = a+b
    d = a-b
    return c,d

def main():
    a = 2
    b = 3
    c, d = subroutine1(a,b)

    #如果只想要部分返回值的话可以这么写
    _, e = subroutine1(a,b)
```

* 主函数部分建议写在单独定义的主函数当中，因为这样可以在`if __name__=='__main__':`之后的部分单独调用部分函数进行单元测试。每个子函数都通过单元测试后再测试项目整体，会比较容易。

## python上的第二个任务-求矩阵转置

python中有关矩阵运算通常都调用numpy包来实现，这里借助这个任务来了解常用的与科研相关的python工具包。有了这些工具包我们就有了基于python进行数据计算与分析的有力工具，可以轻易利用复杂/高级的算法实现我们的目的，避免将大量时间花在如何用代码实现这些算法上。[代码文件 task2.py](https://github.com/YinZhaokai/python_tutorial/blob/master/task2.py)。

***

* **numpy** -- NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。NumPy 是一个运行速度非常快的数学库，主要用于数组计算，包含：

  * 一个强大的N维数组对象 ndarray；
  * 广播功能函数；
  * 整合 C/C++/Fortran 代码的工具；
  * 线性代数、傅里叶变换、随机数生成等功能。

    [numpy官方教程](https://www.numpy.org.cn/)、[numpy教程一](https://www.runoob.com/numpy/numpy-tutorial.html)、[numpy教程二](https://www.yiibai.com/numpy/)

***

* **pandas** -- Pandas 是 Python 的核心数据分析支持库，提供了快速、灵活、明确的数据结构，旨在简单、直观地处理关系型、标记型数据。Pandas 适用于处理以下类型的数据：

  * 与 SQL 或 Excel 表类似的，含异构列的表格数据;
  * 有序和无序（非固定频率）的时间序列数据;
  * 带行列标签的矩阵数据，包括同构或异构型数据;
  * 任意其它形式的观测、统计数据集, 数据转入 Pandas 数据结构时不必事先标记。
  
  Pandas 的主要数据结构是 Series（一维数据）与 DataFrame（二维数据），这两种数据结构足以处理金融、统计、社会科学、工程等领域里的大多数典型用例。Pandas 基于 NumPy 开发，可以与其它第三方科学计算支持库完美集成。[pandas官方教程](https://www.pypandas.cn/docs/getting_started/)、[pandas教程一](https://zhuanlan.zhihu.com/p/25630700)、[pandas教程二](https://www.cnblogs.com/misswangxing/p/7903595.html)

***

* **sklearn** -- Scikit learn 也简称 sklearn, 是机器学习领域当中最知名的 python 模块之一。Sklearn 包含了很多种机器学习的方式:

  * Classification 分类
  * Regression 回归
  * Clustering 聚类
  * Dimensionality reduction 数据降维
  * Model Selection 模型选择
  * Preprocessing 数据预处理
  
  [sklearn官方教程](https://sklearn.apachecn.org/)、[sklearn教程一](https://zhuanlan.zhihu.com/p/35708083)、[sklearn教程二](https://blog.csdn.net/lilianforever/article/details/53780613)

***

* **matplotlib** -- Matplotlib 是一个 Python 的 2D绘图库，它以各种硬拷贝格式和跨平台的交互式环境生成出版质量级别的图形。诸如：线图、散点图、等高线图、条形图、柱状图、3D 图形、甚至是图形动画等等。[matplotlib官方教程](https://www.matplotlib.org.cn/)、[matplotlib教程一](https://www.runoob.com/w3cnote/matplotlib-tutorial.html)、[matplotlib教程二](https://www.cnblogs.com/nxld/p/7435930.html)

***

上述提到的都是第三方的工具包，除此之外，python还内置了很多更泛用的工具模块（标准库），比如：

* **math** - 该模块提供了对C标准定义的数学函数的访问。
* **time** - 该模块提供了各种时间相关的函数。
* **datetime** - datetime 模块提供了可以通过多种方式操作日期和时间的类。在支持日期时间数学运算的同时，实现的关注点更着重于如何能够更有效地解析其属性用于格式化输出和数据操作。
* **random** - 该模块实现了各种分布的伪随机数生成器。在实数轴上，有计算均匀、正态（高斯）、对数正态、负指数、伽马和贝塔分布的函数。 为了生成角度分布，可以使用 von Mises 分布。
* **sys** - 该模块提供了一些变量和函数。这些变量可能被解释器使用，也可能由解释器提供。这些函数会影响解释器。
* **os** - 该模块提供了一些方便使用操作系统相关功能的函数。 如果你是想读写一个文件，请参阅 *open()*，如果你想操作路径，请参阅 *os.path* 模块，如果你想在命令行上读取所有文件中的所有行请参阅 *fileinput* 模块。 有关创建临时文件和目录的方法，请参阅 *tempfile* 模块，对于高级文件目录处理，请参阅 *shutil* 模块。
* **json** - JSON 编码和解码器。
* **re** - 这个模块提供了正则表达式匹配操作。
* **configparser** - 该类的作用是使用配置文件生效，配置文件的格式和windows的INI文件的格式相同。
* **subprocess** - subprocess 模块允许你生成新的进程（比如调用exe），连接它们的输入、输出、错误管道，并且获取它们的返回码。
* **multiprocessing** - 基于进程的并行。

[官方文档](https://docs.python.org/zh-cn/3.7/library/index.html)
[参考一](https://blog.csdn.net/ruanxingzi123/article/details/82787852)
[参考二](https://blog.csdn.net/qq_39407518/article/details/80065601)

## python上的第三个任务-水文整编计算

从excel文件中（提示：可以借助xlrd, xlwt包或pandas包）读取一段历史径流数据（[数据见此]()），按照线性插值的方式，从1998-7-13 8:00到1998-9-11 8:00 每隔一个小时都插上值，并保存在excel文件中（包括时间列），并将插值后的径流数据进行z-score标准化，并分别绘制标准化前、后的洪水过程线图。
