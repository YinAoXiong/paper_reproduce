from main import Inpainter
import argparse
import matplotlib.pyplot as plt
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ps',
        '--patch-size',
        help='the size of the patches',
        type=int,
        default=9
    )
    parser.add_argument(
        '-o',
        '--output',
        help='the file path to save the output image',
        default='output.jpg'
    )
    parser.add_argument(
        '--plot-progress',
        help='plot each generated image',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--plot',
        help='plot the output image',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-df',
        '--difference-algorithm',
        help='The algorithm for calculating the difference between pictures.Available options are sq and sq_with_eucldean.',
        default='sq'
    )
    parser.add_argument(
        'input_image',
        help='the image containing objects to be removed'
    )
    parser.add_argument(
        'mask',
        help='the mask of the region to be removed'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    image=cv.imread(args.input_image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    mask= cv.imread(args.mask, 0)
    output_image=Inpainter(
        image=image,
        mask=mask,
        patch_size=args.patch_size,
        diff_algorithm=args.difference_algorithm,
        plot_progress=args.plot_progress
    ).inpaint()
    plt.imsave(args.output,output_image)
    if args.plot:
        plt.clf()
        plt.imshow(output_image)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main()