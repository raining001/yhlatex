#!/usr/bin/env python
"""Train models."""
import sys, os, argparse, logging, glob

k = ['(', ')', '<', '>', '[', ']', '\\langle', '\\rangle','|', '\}', '\{']
ss = ['\\left(', '\\right)', '\\left[', '\\right]','\\right>', '\\left<', '\\left\\langle', '\\right\\langle', '\\right|', '\\left|']
style = ['\\biggl', '\\biggr', '\\Biggl', '\\Biggr', '\\bigg', '\\Bigg', '\\bigr', '\\bigl', '\\Bigl', '\\Bigr', '\\Big', '\\big']


def process(line):
    result =""
    line = line.split(" ")
    for i in line:

        if i in style:
            result+=i + " "
            continue
        if i in ss:
            result+=i + " "
            continue
        if i in k:
            result += i + " "
            continue
    if len(result) == 0:
        result = "N"
    return result


def savetxt(ignoretxt, out_dir):
    fw = open(out_dir, 'w')  # 将要输出保存的文件地址
    for line in ignoretxt:  # 读取的文件
        fw.write(line)  # 将字符串写入文件中
        fw.write("\n")  # 换行


def main():
    # dir = "data/im2text/"
    # tgt_file = ['tgt-test', 'tgt-train', 'tgt-val']
    dir = "results/style_35/"
    # tgt_file = ['ref', 'pred']
    tgt_file = ['cor_pred']

    for i in tgt_file:
        fs = dir + i + '.txt'
        results = []
        with open(fs) as f:
            for line in f:
                results.append(process(line.strip()))
        print(i)
        savetxt(results, dir+i+'-st.txt')



if __name__ == "__main__":
    main()

