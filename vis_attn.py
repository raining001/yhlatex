from onmt.visual_atten import vis_img_with_attention
import click

@click.command()
@click.option('--image', default="data/im2text/images/481534e498.png",
              help='Path to image to OCR')
@click.option('--attns', default="error_lab/attn.txt",
              help='Path to model json config')
@click.option('--output', default="error_lab/",
              help='Dir for results and model weights')
def main(image, attns, output):
    attn = []
    with open(attns) as f:
        for idx, line in enumerate(f):
            at = line.split("\t")
            attn.append([float(i) for i in at])
    # print(attn)
    dir_output = output
    img_path = image
    #
    hyps = ['m', '_', '{', '3', '/', '2', '}', '\\sim', '\\frac', '{', 'e', '^', '{', '2', 'A', '_', '{', 'S', 'U', 'S', 'Y',
     '}', '\\Lambda', '_', '{', 'S', 'U', 'S', 'Y', '}', '^', '{', '2', '}', '}', '}', '{', 'M', '_', '{', '4', '}',
     '}', '\\,', '.', '</s>']
    vis_img_with_attention(hyps, attn, img_path, dir_output)

if __name__ == "__main__":
    main()
