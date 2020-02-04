# Python教程

## 写在前面的话

学习深度学习免不了要学习Python，在学习Python的过程中我有如下几个体会，先和大家分享。

* Python在线学习资源丰富，各种教程和问题都可以通过搜索引擎获得，特别是各种博客(如[csdn](https://www.csdn.net/)、[cnblogs](https://www.cnblogs.com/))上面的经验，[github](https://github.com/)上面现成的代码，以及[stackoverflow](https://stackoverflow.com/)上面的问答，非常方便，一定要加以利用。**但python由于历史原因发展为了2.7和3.x两大分支，互不兼容，因此在查询问题的时候一定要注意python版本是否一致**；
* Python现成的“轮子”很多，有很多成熟的工具让我们来使用，很多情况下都不需要写复杂的逻辑；
* 要熟悉命令行的操作方式，在windows上就是cmd或powershell，在linux上就是shell，在线的各种python教程中都会有各种在命令行中输入命令的操作，大家一定要习惯。

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
