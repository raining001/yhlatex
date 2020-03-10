#!/usr/bin/env python
from onmt.bin.train import main
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

if __name__ == "__main__":

    main()
