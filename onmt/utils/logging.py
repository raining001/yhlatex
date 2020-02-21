# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
    filename='train_log.txt')
logger = logging.getLogger()

# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# logger = logging.getLogger()
# logging.getLogger('').addHandler(console)
def init_logger(log_file=None, log_file_level=logging.NOTSET):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename='train_log.txt')

    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    # logger.handlers = [console_handler]
    logging.getLogger('').addHandler(console_handler)


    return logger
