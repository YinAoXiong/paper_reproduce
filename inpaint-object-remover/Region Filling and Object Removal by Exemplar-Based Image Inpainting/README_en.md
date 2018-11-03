# Region Filling and Object Removal by Exemplar-Based Image Inpainting

[中文介绍](README.md) | [README](README_en.md)

There are the implementation code of *Region Filling and Object Removal by Exemplar-Based Image Inpainting*.The specific implementation code can refer to the source code in inpainter/main.py, the content interpretation of the paper and the implementation process can be found in the personal blog：[my blog](https://blog.yinaoxiong.cn/2018/11/03/%E5%A4%8D%E7%8E%B0-Region-Filling-and-Object-Removal-by-Exemplar-Based-Image-Inpainting.html)

## Environmental preparation

It is recommended to use conda for dependency installation.

```shell
conda install --file requirements-conda.txt
```

pip

```shell
pip install -r requirements-pip.txt
```

## Using help

```shell
usage: inpainter [-h] [-ps PATCH_SIZE] [-o OUTPUT] [--plot-progress] [--plot]
                 [-df DIFFERENCE_ALGORITHM]
                 input_image mask

positional arguments:
  input_image           the image containing objects to be removed
  mask                  the mask of the region to be removed

optional arguments:
  -h, --help            show this help message and exit
  -ps PATCH_SIZE, --patch-size PATCH_SIZE
                        the size of the patches
  -o OUTPUT, --output OUTPUT
                        the file path to save the output image
  --plot-progress       plot each generated image
  --plot                plot the output image
  -df DIFFERENCE_ALGORITHM, --difference-algorithm DIFFERENCE_ALGORITHM
                        The algorithm for calculating the difference between
                        pictures.Available options are sq,sq_with_eucldean and
                        sq_with_gradient.
```

The actual situation may differ from the above. Please use `python inpainter -h` to see the help information.

## Quick start

```shell
python inpainter -o result/result.jpg resources/image7.jpg resources/mask7.jpg
```