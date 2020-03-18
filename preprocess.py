#!/usr/bin/env python
from onmt.bin.preprocess import main
import torch


def load_txt(filename, filter):
    formulas = []
    with open(filename) as f:
        for line in f:
            if filter[0] in line:
                print(line)
                formulas.append(line)
    return formulas

if __name__ == "__main__":
    # filter = ["\left("]
    # # formulas = load_txt("results/0310/results/ref.txt", filter)
    # # print(len(formulas))
    # formulas = load_txt("results/0310/results/pred125_bz1.txt", filter)
    # print(len(formulas))
    a = [[[1,2,3,7],
         [4,5,6,7],
         [7,8,9,7]]
         ,[[-1,-2,-3,-7],
         [-4,-5,-6,-7],
         [-7,-8,-9,-7]]
         ]
    b = torch.Tensor(a)
    print(b)
    print(b.size())
    print(b.view(b.size(0), b.size(1)*b.size(2)))
    print(b.view(b.size(0), b.size(2)*b.size(1)))
