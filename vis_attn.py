from onmt.visual_atten import vis_img_with_attention
import click

@click.command()
@click.option('--image', default="data/images/481534e498.png",
              help='Path to image to OCR')
@click.option('--attns', default="error_lab/attn.txt",
              help='Path to model json config')
@click.option('--output', default="error_lab/",
              help='Dir for results and model weights')
@click.option('--latex', default="error_lab/pred.txt",
              help='Dir for results and model weights')
def main(image, attns, output, latex):
    attn = []
    with open(attns) as f:
        for idx, line in enumerate(f):
            at = line.split("\t")
            attn.append([float(i) for i in at])
    # print(attn)
    dir_output = output
    img_path = image
    latex = latex
    #
    hyps = readlatex(latex)
    print(hyps)
    vis_img_with_attention(hyps, attn, img_path, dir_output)


def readlatex(filename):
    with open(filename) as f:
        formulas = []
        for idx, line in enumerate(f):
            formulas = line.split()
    return formulas

if __name__ == "__main__":
    main()
