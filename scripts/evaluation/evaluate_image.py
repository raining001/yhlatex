import sys, os, argparse, logging, glob
import numpy as np
from PIL import Image
import distance
from LevSeq import StringMatcher


def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate image related metrics.')

    parser.add_argument('--images-dir', dest='images_dir', default='error_lab/',
                        type=str, required=True,

                        help=('Images directory containing the rendered images. A subfolder with name "images_gold" for the rendered gold images, and a subfolder "images_pred" must be created beforehand by using scripts/evaluation/render_latex.py.'
                        ))

    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='EM.txt',
                        help=('Log file path, default=log.txt' 
                        ))

    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)
    images_dir = parameters.images_dir
    log_path = os.path.join(images_dir, parameters.log_path)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s'%__file__)

    gold_dir = os.path.join(images_dir, 'images_gold')
    pred_dir = os.path.join(images_dir, 'images_pred')
    assert os.path.exists(gold_dir), gold_dir 
    assert os.path.exists(pred_dir), pred_dir 
    total_edit_distance = 0
    total_ref = 0
    total_num = 0
    total_correct = 0
    total_correct_eliminate = 0
    filenames = glob.glob(os.path.join(gold_dir, '*'))
    results = []

    for filename in filenames:
        filename2 = os.path.join(pred_dir, os.path.basename(filename))
        edit_distance, ref, match1, match2 = img_edit_distance_file(filename, filename2)
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
            logging.info('Total Num: %d'%total_num)
            logging.info('Accuracy (w spaces): %f'%(float(total_correct)/total_num))
            logging.info('Accuracy (w/o spaces): %f'%(float(total_correct_eliminate)/total_num))
            logging.info('Edit Dist (w spaces): %f'%(1.-float(total_edit_distance)/total_ref))
            logging.info('Total Correct (w spaces): %d'%total_correct)
            logging.info('Total Correct (w/o spaces): %d'%total_correct_eliminate)
            logging.info('Total Edit Dist (w spaces): %d'%total_edit_distance)
            logging.info('Total Ref (w spaces): %d'%total_ref)
            logging.info('')

    logging.info('------------------------------------')
    logging.info('Final')
    logging.info ('Total Num: %d'%total_num)
    logging.info ('Accuracy (w spaces): %f'%(float(total_correct)/total_num))
    logging.info ('Accuracy (w/o spaces): %f'%(float(total_correct_eliminate)/total_num))
    logging.info ('Edit Dist (w spaces): %f'%(1.-float(total_edit_distance)/total_ref))
    logging.info ('Total Correct (w spaces): %d'%total_correct)
    logging.info ('Total Correct (w/o spaces): %d'%total_correct_eliminate)
    logging.info ('Total Edit Dist (w spaces): %d'%total_edit_distance)
    logging.info ('Total Ref (w spaces): %d'%total_ref)
    savelogs(results, images_dir)



def savelogs(results, images_dir):

   with open(images_dir + "log_img_ss.txt", 'w') as f:
        for (img_name, match1, match2, l_dist) in results:
            f.write("{} {} {} {}\n".format(img_name, match1, match2, l_dist))


# return (edit_distance, ref, match, match w/o)
def img_edit_distance(im1, im2, out_path=None):

    img_data1 = np.asarray(im1, dtype=np.uint8) # height, width
    img_data1 = np.transpose(img_data1)
    h1 = img_data1.shape[1]
    w1 = img_data1.shape[0]
    img_data1 = (img_data1<=128).astype(np.uint8)
    # print(im1)
    # print(im2)
    if im2:
        img_data2 = np.asarray(im2, dtype=np.uint8) # height, width
        img_data2 = np.transpose(img_data2)
        h2 = img_data2.shape[1]
        w2 = img_data2.shape[0]
        img_data2 = (img_data2<=128).astype(np.uint8)
    else:
        img_data2 = []
        h2 = h1
    # img_data 为二维向量，有原图像进行了转置，所以每一行其实是高，共有宽大小的行数
    # print('img_data1', len(img_data1), len(img_data1[0]), h1)  img_data1 910 91 91
    # print('img_data2', len(img_data2), len(img_data2[1]), h2)  img_data2 916 91 91
    if h1 == h2:
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]
    elif h1 > h2: # pad h2
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item])+''.join(['0']*(h1-h2)) for item in img_data2]
    else:
        seq1 = [''.join([str(i) for i in item])+''.join(['0']*(h2-h1)) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]
    # 将list中每一行转化为str，这样原来二维list转变为一维list，每一项的值表示原图像第一列中的值，
    # seq1 ['0000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000',...]
    # print('seq1', seq1)
    # print('seq2', seq2)

    # convert each column binary into int
    # seq1_int [274877906944,...]
    seq1_int = [int(item, 2) for item in seq1]
    seq2_int = [int(item, 2) for item in seq2]
    # 将每一个str转成2进制的值
    # print('seq1_int', seq1_int)
    # print('seq2_int', seq2_int)
    big = int(''.join(['0' for i in range(max(h1, h2))]), 2)
    # print('big', big)
    seq1_eliminate = []     # 用于存放 转变为 int的 无空格的 图片信息
    seq2_eliminate = []     #
    seq1_new = []           # 用于存放 还是str的 无空格的 图片信息
    seq2_new = []           # seq1_new ['0000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000',...]
    # 将非空格的值加入到seq1中
    for idx, items in enumerate(seq1_int):
        if items>big:
            seq1_eliminate.append(items)
            seq1_new.append(seq1[idx])
    # print('seq1_new', seq1_new)
    # print('seq1_new', len(seq1_new))
    # print('seq1_eliminate', len(seq1_eliminate))
    for idx, items in enumerate(seq2_int):
        if items > big:
            seq2_eliminate.append(items)
            seq2_new.append(seq2[idx])
    if len(seq2) == 0:
        return (len(seq1), len(seq1), False, False)

    # 每列 不超过 5个不同就算是相同的
    def make_strs(int_ls, int_ls2):
        d = {}
        seen = []
        def build(ls):
            for l in ls:
                if int(l, 2) in d: continue
                found = False
                l_arr = np.array(list(map(int, l)))
                # 相差不到5位，则使用该值作为目标值
                for l2, l2_arr in seen:
                    if np.abs(l_arr -l2_arr).sum() < 5:
                        d[int(l, 2)] = d[int(l2, 2)]
                        found = True
                        break
                if not found:
                    d[int(l, 2)] = chr(len(seen))           # 返回响应字符
                    # d[int(l, 2)] = str(len(seen))
                    seen.append((l, np.array(list(map(int, l)))))
                    
        build(int_ls)
        build(int_ls2)
        # print('d', d)

        return "".join([d[int(l, 2)] for l in int_ls]), "".join([d[int(l, 2)] for l in int_ls2])

    #if out_path:
    # seq1_t 和 seq2_t 会将后列的像素点与前列的像素点进行比较，如果两者相差在5个像素点内，则可以认为是同一列，这样就生成了新的关于原图列的表示
    seq1_t, seq2_t = make_strs(seq1, seq2)
    # print('seq1_t', seq1_t)
    # print('seq2_t', seq2_t)
    # edit_distance 只要每列有不同，则就+1
    edit_distance = distance.levenshtein(seq1_int, seq2_int)
    # print('edit_distance', edit_distance)

    match = True
    # 如果图片不能完全匹配， 但是操作不大，在5以内，仍然可以接受
    if edit_distance>0:
        # s1 = "abcde"
        # s2 = "abeecef"
        matcher = StringMatcher(None, seq1_t, seq2_t)
        # matcher = StringMatcher(None, s1, s2)
        ls = []
        for op in matcher.get_opcodes():
            # print(op)
            # print(op[2])
            # print(op[1])
            if op[0] == "equal" or (op[2]-op[1] < 5):
                ls += [[int(r) for r in l]
                       for l in seq1[op[1]:op[2]]
                       ]
            elif op[0] == "replace":
                a = seq1[op[1]:op[2]]
                b = seq2[op[3]:op[4]]
                ls += [[int(r1)*3 + int(r2)*2
                        if int(r1) != int(r2) else int(r1)
                        for r1, r2 in zip(a[i] if i < len(a) else [0]*1000,
                                          b[i] if i < len(b) else [0]*1000)]
                       for i in range(max(len(a), len(b)))]
                match = False
            elif op[0] == "insert":

                ls += [[int(r)*3 for r in l]
                       for l in seq2[op[3]:op[4]]]
                match = False
            elif op[0] == "delete":
                match = False
                ls += [[int(r)*2 for r in l] for l in seq1[op[1]:op[2]]]

    match1 = match


    seq1_t, seq2_t = make_strs(seq1_new, seq2_new)

    if len(seq2_new) == 0 or len(seq1_new) == 0:
        if len(seq2_new) == len(seq1_new):
            return (edit_distance, max(len(seq1_int),len(seq2_int)), match1, True)# all blank
        return (edit_distance, max(len(seq1_int),len(seq2_int)), match1, False)
    match = True
    matcher = StringMatcher(None, seq1_t, seq2_t)

    ls = []
    for op in matcher.get_opcodes():
        if op[0] == "equal" or (op[2]-op[1] < 5):
            ls += [[int(r) for r in l]
                   for l in seq1[op[1]:op[2]]
                   ] 
        elif op[0] == "replace":
            a = seq1[op[1]:op[2]]
            b = seq2[op[3]:op[4]]
            ls += [[int(r1)*3 + int(r2)*2
                    if int(r1) != int(r2) else int(r1)
                    for r1, r2 in zip(a[i] if i < len(a) else [0]*1000,
                                      b[i] if i < len(b) else [0]*1000)]
                   for i in range(max(len(a), len(b)))]
            match = False
        elif op[0] == "insert":

            ls += [[int(r)*3 for r in l]
                   for l in seq2[op[3]:op[4]]]
            match = False
        elif op[0] == "delete":
            match = False
            ls += [[int(r)*2 for r in l] for l in seq1[op[1]:op[2]]]

    match2 = match

    return (edit_distance, max(len(seq1_int),len(seq2_int)), match1, match2)




def img_edit_distance_file(file1, file2, output_path=None):

    img1 = Image.open(file1).convert('L')
    if os.path.exists(file2):
        img2 = Image.open(file2).convert('L')
    else:
        img2 = None
    return img_edit_distance(img1, img2, output_path)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
