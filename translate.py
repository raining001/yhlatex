#!/usr/bin/env python
from onmt.bin.translate import main
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


if __name__ == "__main__":
    main()


'''
onmt_translate -model py-model.pt

-data_type img -model models/baseline/baseline_nopos.pt -src_dir data/im2text/images -src error_lab/src-test.txt -output error_lab/pred.txt -max_length 150 -beam_size 5 -gpu 0 -verbose -attn_debug -attn_view
-data_type img -model results/baseline/baseline_nopos.pt -src_dir error_lab/images/ -src error_lab/src-test.txt -output error_lab/pred.txt -max_length 150 -beam_size 5 -gpu 0 -verbose --batch_size 1
-data_type img -model results/0325/res_step_150000.pt -src_dir error_lab/images/ -src error_lab/src-test.txt -output error_lab/pred.txt -max_length 150 -beam_size 5 -gpu 0 -verbose --batch_size 1


-data_type img -model results/rowcol_baseline/colrow_step_100000.pt -src_dir data/images/ -src error_lab/src-test.txt -output error_lab/pred.txt -max_length 150 -beam_size 5 -gpu 0 -verbose --batch_size 1
'''
