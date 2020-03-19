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
    # src (bz, c ,h, w)
    # a (bz, T, h, w)
    # v (L, bz, c)

    a = [[[[-1,-8,-9],[-2,-4,-6]],
         [[-2,-9,-1],[-5,-7,-3]]]]

    v = [[[[1,8,9],[2,4,6]],
         [[2,9,1],[5,7,3]]]]
    A = torch.Tensor(a)
    V = torch.Tensor(v)
    all_outputs = []
    for row in range(V.size(2)):
        inp = V[:, :, row, :].transpose(0, 2) \
            .transpose(1, 2)

        all_outputs.append(inp)  # outputs torch.Size([W, bz, c])
    out = torch.cat(all_outputs, 0)

    print("A", A.size())
    print(A)
    print("V", V.size())
    print(V)

    s_attns = A.view(A.size(0), A.size(1), -1)
    memory = out
    print("memory", memory, memory.size())
    print('s_atten', s_attns.size())
    print('s_atten', s_attns)
    nL, nB, nC = memory.size()
    nT = s_attns.size()[1]

    # Normalize
    print('s_attns.view(nB, nT, -1).sum(2)', s_attns.view(nB, nT, -1).sum(2))
    print('s_attns.view(nB, nT, -1).sum(2).view(nB, nT, 1)', s_attns.view(nB, nT, -1).sum(2).view(nB, nT, 1))
    s_attns = s_attns / s_attns.view(nB, nT, -1).sum(2).view(nB, nT, 1)
    print('s_attns_normal', s_attns.size())
    print('s_attn', s_attns)
    # weighted sum
    print("memory.view(nB, nL, nC)", memory.view(nB, nC, nL))
    print("memory.view(nB, nL, nC)", memory.view(nB, nC, nL).size())
    print("memory.transpose(1, 0)", memory.transpose(1, 0).transpose(2,1))
    print("memory.transpose(1, 0)", memory.transpose(1, 0).transpose(2,1).size())
    C = torch.bmm(s_attns.view(nB, nT, nL), memory.transpose(1, 0))
    C = C.view(C.size(1), C.size(0), -1)
    print('C', C)
    print("C", C.size())

    # exit(1)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")

    nB, nC, nH, nW= V.size()
    nT = A.size()[1]
    # Normalize

    A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)
    print("A", A)
    print("A", A.size())
    # # weighted sum
    C = V.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
    print("V", V.view(nB, 1, nC, nH, nW))
    print("A", A.view(nB, nT, 1, nH, nW))
    print("C",C.size())
    print("C", C)
    print("C",  C.view(nB, nT, nC, -1))
    C = C.view(nB, nT, nC, -1).sum(3).transpose(1, 0)

    print("C_", C)



    # C, _ = self.pre_lstm(C)
    # C = F.dropout(C, p=0.3, training=self.training)
