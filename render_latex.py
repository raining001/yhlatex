import sys, os, re, argparse, logging
from scripts.utils.utils import run
from scripts.utils.image_utils import *
from multiprocessing.dummy import Pool as ThreadPool

TIMEOUT = 10

# replace \pmatrix with \begin{pmatrix}\end{pmatrix}
# replace \matrix with \begin{matrix}\end{matrix}
template = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""


def process_args(args):
    parser = argparse.ArgumentParser(description='Render latex formulas for comparison. Note that we need to render both the predicted results, and the original formulas, since we need to make sure the same environment of rendering is used.')


    parser.add_argument('-ignore', dest='ignore_style',
                        type=str,  required=True, default=0, help=('If true, when render the latex, it will ignore the style'))

    parser.add_argument('-train_set', dest='train_set',
                        type=str, required=True, default=0,
                        help=('If 1 do for train'))

    parser.add_argument('--ref', dest='ref_txt',
                        type=str,  default='error_lab/ref.txt')

    parser.add_argument('--pred', dest='pred_txt',
                        type=str,  default='error_lab/less60_pred.txt')

    parser.add_argument('--output-dir', dest='output_dir',
                        type=str,  default='results/baseline_large/results/')
    parser.add_argument('--replace', dest='replace', action='store_true',
                        help=('Replace flag, if set to false, will ignore the already existing images.'
                        ))
    parser.add_argument('--no-replace', dest='replace', action='store_false')
    parser.set_defaults(replace=True)
    parser.add_argument('--num-threads', dest='num_threads',
                        type=int, default=4,
                        help=('Number of threads, default=4.'
                        ))

    parameters = parser.parse_args(args)
    return parameters

buckets = [
    (120, 50), (160, 40), (200, 40), (200, 50), (240, 40), (240, 50), (280, 40), (280, 50), (320, 40), (320, 50),
    (360, 40), (360, 50), (360, 60),  (360, 100),
    (400, 50), (400, 160), (500, 100), (500, 200), (600, 100), (800, 100), (800, 200), (800, 400)]


def main(args):
    parameters = process_args(args)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s'%__file__)
   
    ref = parameters.ref_txt
    pred = parameters.pred_txt
    output_dir = parameters.output_dir
    ignore_style = parameters.ignore_style
    train_set = parameters.train_set


    assert os.path.exists(ref), ref
    assert os.path.exists(pred), pred
    assert os.path.exists(output_dir), output_dir

    pred_dir = os.path.join(output_dir, 'images_pred')
    gold_dir = os.path.join(output_dir, 'images_gold')
    for dirname in [pred_dir, gold_dir]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    lines = []
    ignoretxt = []

    '''
    [('7944775fc9.png', 
    '\\alpha _ { 1 } ^ { r } \\gamma _ { 1 } + \\dots + \\alpha _ { N } ^ { r } \\gamma _ { N } = 0 \\quad ( r = 1 , . . . , R ) \\; ,\n', 
    'data/full/images_rendered/images_gold/7944775fc9.png', False)]
    '''

    with open(ref) as fin:
        i = 0
        for line in fin:
            img_path = str(i) + '.png'
            ref_txt = line.strip()

            if ignore_style == '1':
                ref_txt = delstyle(ref_txt)
                ignoretxt.append(ref_txt)
            lines.append((img_path, ref_txt, os.path.join(gold_dir, img_path), parameters.replace, train_set))
            i = i + 1
    if ignore_style== '1':
        savetxt(ignoretxt, output_dir+'ns_ref.txt')
    ignoretxt = []
    with open(pred) as fin:
        i = 0
        for line in fin:
            img_path = str(i) + '.png'
            pred_txt = line.strip()
            if ignore_style== '1':
                pred_txt = delstyle(pred_txt)
                ignoretxt.append(pred_txt)
            lines.append((img_path, pred_txt, os.path.join(pred_dir, img_path), parameters.replace, train_set))
            i = i + 1
    if ignore_style== '1':
        savetxt(ignoretxt, output_dir+'ns_pred.txt')
    logging.info('Creating pool with %d threads'%parameters.num_threads)
    pool = ThreadPool(parameters.num_threads)
    logging.info('Jobs running...')
    results = pool.map(main_parallel, lines)
    pool.close() 
    pool.join()

    # if train_set == '1':
    #     for root, dirs, files in os.walk(pred_dir):
    #         for fl in files:
    #             image_dir = os.path.join(pred_dir, fl)
    #             image = Image.open(image_dir)
    #             change_image_channels(image, image_dir)
    #
    #     for root, dirs, files in os.walk(gold_dir):
    #         for fl in files:
    #             image_dir = os.path.join(gold_dir, fl)
    #             image = Image.open(image_dir)
    #             change_image_channels(image, image_dir)



def savetxt(ignoretxt, out_dir):
    fw = open(out_dir, 'w')  # 将要输出保存的文件地址
    for line in ignoretxt:  # 读取的文件
        fw.write(line)  # 将字符串写入文件中
        fw.write("\n")  # 换行


def change_image_channels(image, image_path):
    if image.mode != 'RGB':
        image = image.convert("RGB")
        os.remove(image_path)
        image.save(image_path)
    return image


def delstyle(latex):
    replace = ['\\left(', '\\right)', '\\left[', '\\right]']
    remove= ['\\biggl ', '\\biggr ', '\\Biggl ', '\\Biggr ', '\\bigg ', '\\Bigg ', '\\bigr ', '\\bigl ','\\Bigl ', '\\Bigr ', '\\Big ', '\\big ', '\\textstyle ', '\\displaystyle ']
    newstyle = ['(', ')', '[', ']']
    for i, v in enumerate(replace):
        if v in latex:
            latex = latex.replace(v, newstyle[i])

    for i, v in enumerate(remove):
        if v in latex:
            latex = latex.replace(v, "")
    return latex

def output_err(output_path, i, reason, img):
    logging.info('ERROR: %s %s\n'%(img,reason))

def main_parallel(line):
    img_path, l, output_path, replace, train_set = line
    # print(line)
    pre_name = output_path.replace('/', '_').replace('.', '_')
    l = l.strip()
    l = l.replace(r'\pmatrix', r'\mypmatrix')
    l = l.replace(r'\matrix', r'\mymatrix')
    # remove leading comments
    l = l.strip('%')
    if len(l) == 0:
        l = '\\hspace{1cm}'
    # \hspace {1 . 5 cm} -> \hspace {1.5cm}
    for space in ["hspace", "vspace"]:
        match = re.finditer(space + " {(.*?)}", l)
        if match:
            new_l = ""
            last = 0
            for m in match:
                new_l = new_l + l[last:m.start(1)] + m.group(1).replace(" ", "")
                last = m.end(1)
            new_l = new_l + l[last:]
            l = new_l    
    if replace or (not os.path.exists(output_path)):
        tex_filename = pre_name+'.tex'
        log_filename = pre_name+'.log'
        aux_filename = pre_name+'.aux'
        with open(tex_filename, "w") as w: 
            # print >> w, (template%l)
            print((template%l), file=w)
        run("pdflatex -interaction=nonstopmode %s  >/dev/null"%tex_filename, TIMEOUT)
        os.remove(tex_filename)
        os.remove(log_filename)
        os.remove(aux_filename)
        pdf_filename = tex_filename[:-4]+'.pdf'
        png_filename = tex_filename[:-4]+'.png'
        if not os.path.exists(pdf_filename):
            output_err(output_path, 0, 'cannot compile', img_path)
        else:
            # os.system("convert -density 200 -quality 100 %s %s"%(pdf_filename, png_filename))
            os.system("convert -density 110 -quality 100 %s %s" % (pdf_filename, png_filename))
            os.remove(pdf_filename)

            if os.path.exists(png_filename):

                if train_set == '1':
                    # print(png_filename, output_path)
                    crop_image(png_filename, png_filename)
                    pad_group_image(png_filename, output_path, (5,5,5,5), buckets)
                else:
                    crop_image(png_filename, output_path)
                os.remove(png_filename)

        
if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
