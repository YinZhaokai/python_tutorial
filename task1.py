#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   task1.py
@Time    :   2020/02/04 03:32:03
@Author  :   Zhaokai Yin
@Version :   0.1
@Contact :   yinzhaokai@foxmail.com
@License :   (C)Copyright 2017-2020
'''

# here put the import lib
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
