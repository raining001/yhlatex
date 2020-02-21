#!/usr/bin/env python
from onmt.bin.translate import main
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

if __name__ == "__main__":
    main()


'''
onmt_translate -model py-model.pt

-data_type img -model models/baseline/baseline_nopos.pt -src_dir data/im2text/images -src error_lab/src-test.txt -output error_lab/pred.txt -max_length 150 -beam_size 5 -gpu 0 -verbose -attn_debug -attn_view


'''
