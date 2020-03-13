#!/usr/bin/env python
from onmt.bin.preprocess import main
import torch



if __name__ == "__main__":
    a = [[[1 for _ in range(3)] for _ in range(4)] for _ in range(5)]
    sa = torch.tensor(a, dtype=torch.long)
    b = [[[2 for _ in range(1)] for _ in range(4)] for _ in range(5)]
    sb = torch.tensor(b, dtype=torch.long)
    print(sb.size())
    c = torch.mul(sb, sa)
    print(c)