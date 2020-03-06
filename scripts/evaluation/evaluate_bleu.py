import os, sys, copy, argparse, shutil, pickle, subprocess, logging

def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate BLEU score')

    parser.add_argument('--ref', dest='ref_txt',
                        type=str,  default='results/baseline_large/results/less60_ref.txt')

    parser.add_argument('--pred', dest='pred_txt',
                        type=str,  default='results/baseline_large/results/less60_pred.txt')

    parser.add_argument('--output_dir', dest='output_dir',
                        type=str,  default='results/baseline_large/results/')


    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='bleu.txt',
                        help=('Log file path, default=log.txt'))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)

    parameters = process_args(args)
    output_dir = parameters.output_dir
    log_path = os.path.join(output_dir, parameters.log_path)
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

    ref_txt = parameters.ref_txt
    pred_txt = parameters.pred_txt
    output_dir = parameters.output_dir
    logging.info(pred_txt)
    assert os.path.exists(ref_txt), 'Label file %s not found'%ref_txt
    assert os.path.exists(pred_txt), 'Data file %s not found'%pred_txt
    assert os.path.exists(output_dir), 'Result file %s not found'%output_dir


    metric = subprocess.check_output('perl third_party/multi-bleu.perl %s < %s'%(ref_txt, pred_txt), shell=True)
    #os.remove('.tmp.pred.txt')
    #os.remove('.tmp.gold.txt')
    logging.info(metric)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
