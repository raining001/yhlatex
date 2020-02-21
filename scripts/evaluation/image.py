import numpy as np
import distance
from imageio import imread
import os
import time

from utils.general import get_files
from .evaluate_image import img_edit_distance_file
import logging

def score_dirs(dir_ref, dir_hyp, prepro_img, ckpt):
    """Returns scores from a dir with images

    Args:
        dir_ref: (string)
        dir_hyp: (string)
        prepro_img: (lambda function)

    Returns:
        scores: (dict)

    """
    img_refs = [f for f in get_files(dir_ref) if f.split('.')[-1] == "png"]
    img_hyps = [f for f in get_files(dir_hyp) if f.split('.')[-1] == "png"]

    em_tot = l_dist_tot = length_tot = n_ex = 0
    result = []
    flag = ""

    for img_name in img_refs:
        print('img_name', img_name)
        img_ref = imread(dir_ref + img_name)
        img_ref = prepro_img(img_ref)

        if img_name in img_hyps:
            print('img_name', img_name)
            img_hyp = imread(dir_hyp + img_name)
            img_hyp = prepro_img(img_hyp)
            l_dist, length = img_edit_distance(img_ref, img_hyp)


        else:
            l_dist = length = img_ref.shape[1]

        l_dist_tot += l_dist
        length_tot += length
        # if l_dist < 1:
        if l_dist < 1:
            em_tot += 1
            flag = "same"
            print("flag", flag, l_dist)
        n_ex += 1
        result += [(img_name, l_dist)]
        flag = ""

    # compute scores
    scores = dict()
    scores["EM"]  = em_tot / float(n_ex) if n_ex > 0 else 0
    scores["Lev"] = 1 - l_dist_tot / float(length_tot) if length_tot > 0 else 0
    savelog(result, scores, ckpt)
    return scores


def score_dirs_new(dir_ref, dir_hyp, model, dir):
    """Returns scores from a dir with images

    Args:
        dir_ref: (string)
        dir_hyp: (string)
        prepro_img: (lambda function)

    Returns:
        scores: (dict)

    """
    img_refs = [f for f in get_files(dir_ref) if f.split('.')[-1] == "png"]
    total_edit_distance = 0
    total_ref = 0
    total_num = 0
    total_correct = 0
    total_correct_eliminate = 0
    results = []
    for filename in img_refs:
        filename2 = os.path.join(dir_hyp, os.path.basename(filename))
        edit_distance, ref, match1, match2 = img_edit_distance_file(dir_ref+filename, filename2)
        results += [(filename, match1, match2, edit_distance)]
        total_edit_distance += edit_distance
        total_ref += ref
        total_num += 1
        if match1:
            total_correct += 1
        if match2:
            total_correct_eliminate += 1
        print(filename, match1, match2, edit_distance)
        if total_num % 100 == 0:
            logging.info('Total Num: %d' % total_num)
            logging.info('Accuracy (w spaces): %f' % (float(total_correct) / total_num))
            logging.info('Accuracy (w/o spaces): %f' % (float(total_correct_eliminate) / total_num))
            logging.info('Edit Dist (w spaces): %f' % (1. - float(total_edit_distance) / total_ref))
            logging.info('Total Correct (w spaces): %d' % total_correct)
            logging.info('Total Correct (w/o spaces): %d' % total_correct_eliminate)
            logging.info('Total Edit Dist (w spaces): %d' % total_edit_distance)
            logging.info('Total Ref (w spaces): %d' % total_ref)
            logging.info('')

    logging.info('------------------------------------')
    logging.info('Final')
    logging.info('Total Num: %d' % total_num)
    logging.info('Accuracy (w spaces): %f' % (float(total_correct) / total_num))
    logging.info('Accuracy (w/o spaces): %f' % (float(total_correct_eliminate) / total_num))
    logging.info('Edit Dist (w spaces): %f' % (1. - float(total_edit_distance) / total_ref))
    logging.info('Total Correct (w spaces): %d' % total_correct)
    logging.info('Total Correct (w/o spaces): %d' % total_correct_eliminate)
    logging.info('Total Edit Dist (w spaces): %d' % total_edit_distance)
    logging.info('Total Ref (w spaces): %d' % total_ref)
    # compute scores
    scores = dict()
    scores["Total Num"] = total_num
    scores["EM"] =  (float(total_correct) / total_num)
    scores["EM(w/o)"] = (float(total_correct_eliminate) / total_num)
    scores["Edit"] = (1. - float(total_edit_distance) / total_ref)
    scores["Total Correct"] = total_correct
    scores["Total Correct (w/o spaces)"] = total_correct_eliminate
    scores["Total Edit Dist (w spaces)"] = total_edit_distance
    scores["Total Ref (w spaces)"] = total_ref
    savelogs(results, scores, model, dir)
    return scores


def savelogs(results, scores, model, dir):


   with open(dir + str(model) + '_img_ss.txt', 'w') as f:
        for (img_name, match1, match2, l_dist) in results:
            f.write("{} {} {} {}\n".format(img_name, match1, match2, l_dist))
        f.write("Total Num:{}\n"
                "EM:{}\n"
                "EM(w/o):{}\n"
                "Edit:{}\n"
                "Total Correct:{}\n"
                "Total Correct (w/o spaces):{}\n"
                "Total Edit Dist (w spaces)\n"
                "Total Ref (w spaces)\n".format(scores["Total Num"], scores["EM"], scores["EM(w/o)"], scores["Edit"],
                                                       scores["Total Correct"], scores["Total Correct (w/o spaces)"],
                                                       scores["Total Edit Dist (w spaces)"], scores["Total Ref (w spaces)"]))

def savelog(result, scores, ckpt):
    if ckpt == "":
        ckpt = "base"
    with open("results/full/eval_log/" + str(ckpt) + '_img.txt', 'w') as f:
        for (img_name, l_dist) in result:
            f.write("{} {}\n".format(img_name, l_dist))
        f.write("EM:{}  Lev:{}\n".format(scores["EM"], scores["Lev"]))


def img_edit_distance(img1, img2):
    """Computes Levenshtein distance between two images.
    (From Harvard's NLP github)

    Slices the images into columns and consider one column as a character.

    Args:
        im1, im2: np arrays of shape (H, W, 1)

    Returns:
        column wise levenshtein distance
        max length of the two sequences

    """
    # load the image (H, W)
    img1, img2 = img1[:, :, 0], img2[:, :, 0]

    # transpose and convert to 0 or 1
    img1 = np.transpose(img1)
    h1 = img1.shape[1]
    w1 = img1.shape[0]
    img1 = (img1<=128).astype(np.uint8)

    img2 = np.transpose(img2)
    h2 = img2.shape[1]
    w2 = img2.shape[0]
    img2 = (img2<=128).astype(np.uint8)

    # create binaries for each column
    if h1 == h2:
        seq1 = [''.join([str(i) for i in item]) for item in img1]
        seq2 = [''.join([str(i) for i in item]) for item in img2]
    elif h1 > h2:
        seq1 = [''.join([str(i) for i in item]) for item in img1]
        seq2 = [''.join([str(i) for i in item])+''.join(['0']*(h1-h2)) for
                item in img2]
    else:
        seq1 = [''.join([str(i) for i in item])+''.join(['0']*(h2-h1)) for
                item in img1]
        seq2 = [''.join([str(i) for i in item]) for item in img2]

    # convert each column binary into int
    seq1_int = [int(item, 2) for item in seq1]
    seq2_int = [int(item, 2) for item in seq2]
    print('seq1', seq1)
    print('seq2', seq2)
    print('seq1', len(seq1_int))
    print('seq2', len(seq2_int))
    # distance
    print('seq1_int', seq1_int)
    print('seq2_int', seq2_int)

    l_dist = distance.levenshtein(seq1_int, seq2_int)
    length = float(max(len(seq1_int), len(seq2_int)))
    print('l_dist', l_dist)
    print('length', length)


    return l_dist, length
