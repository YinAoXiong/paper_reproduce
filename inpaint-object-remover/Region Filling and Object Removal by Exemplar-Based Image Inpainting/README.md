# Region Filling and Object Removal by Exemplar-Based Image Inpainting

[中文介绍](README.md) | [README](README_en.md)

　　该目录为对论文《Region Filling and Object Removal by Exemplar-Based Image Inpainting》复现。具体实现代码可以参考inpainter/main.py中的源代码，论文内容解释以及实现过程见个人博客：[博客地址](https://blog.yinaoxiong.cn/2018/11/03/%E5%A4%8D%E7%8E%B0-Region-Filling-and-Object-Removal-by-Exemplar-Based-Image-Inpainting.html)

## 环境准备

推荐使用conda进行依赖安装。

```shell
conda install --file requirements-conda.txt
```

pip

```shell
pip install -r requirements-pip.txt
```

## 使用帮助

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

实际情况可能和以上有出入,请通过`python inpainter -h`来查看真实帮助。

## 快速开始

```shell
python inpainter -o result/result.jpg resources/image7.jpg resources/mask7.jpg
```
