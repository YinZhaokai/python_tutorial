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
import time

def main():
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    with open('task1_out.txt', 'w', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    return
if __name__=='__main__':
    main()
