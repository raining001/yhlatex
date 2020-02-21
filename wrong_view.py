import argparse
import sys, argparse

def process_args(args):
    parser = argparse.ArgumentParser(description='Render latex formulas for comparison. Note that we need to render both the predicted results, and the original formulas, since we need to make sure the same environment of rendering is used.')


    parser.add_argument('--output-dir', dest='output_dir',
                        type=str,  default='results/baseline_large/results/')
    parameters = parser.parse_args(args)
    return parameters


def savetxt(ignoretxt, out_dir):
    fw = open(out_dir, 'w')  # 将要输出保存的文件地址
    for line in ignoretxt:  # 读取的文件
        fw.write(line)  # 将字符串写入文件中
        fw.write("\n")  # 换行



def main(args):
    parameters = process_args(args)
    # output_dir = parameters.output_dir + "99_/"
    output_dir = parameters.output_dir
    ref = output_dir + "less60_ref.txt"
    pred = output_dir + "less60_pred.txt"
    log_file = output_dir + "log_img.txt"
    error_result = output_dir + "error_result.txt"
    wserror_result = output_dir + "wserror_result.txt"
    preds = []
    refs = []
    error = []
    error_wos = []

    with open(ref) as fin:
        for line in fin:
            refs.append(line.strip())

    with open(pred) as f:
        for line in f:
            preds.append(line.strip())


    with open(log_file) as fin:
        for line in fin:
            log = line.strip().split(' ')
            if log[1] == 'False':
                idx = int(log[0].split('.')[0])
                error.append(idx)
            if log[2] == 'False':
                idx = int(log[0].split('.')[0])
                error_wos.append(idx)

    refs_error = [refs[i] for i in error]
    preds_error = [preds[i] for i in error]

    refs_woserror = [refs[i] for i in error_wos]
    preds_woserror = [preds[i] for i in error_wos]
    print(len(refs_woserror))
    print(len(preds_woserror))
    result_error = []
    for i in range(len(refs_error)):
        result_error.append(str(error[i]))
        result_error.append(refs_error[i])
        result_error.append(preds_error[i])
        result_error.append(" ")
    savetxt(result_error, error_result)
    result_error = []
    for i in range(len(refs_woserror)):
        result_error.append(str(error_wos[i]))
        result_error.append(refs_woserror[i])
        result_error.append(preds_woserror[i])
        result_error.append(" ")
    savetxt(result_error, wserror_result)



if __name__ == '__main__':
    main(sys.argv[1:])

