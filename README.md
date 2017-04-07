This is W1BS descriptors benchmark from paper [WxBS: Wide Baseline Stereo Generalizations](https://arxiv.org/abs/1504.06603) .

Dataset format:
    data/W1BS/ - directories with subsets. G - geometry, A - appearance, S - sensor, map2photo - map vs. photo
    Each directory contains: 1: regerence image dir, 2 - "noised" image dir, h - homography 1to2 dir
    each image dir contains several images, e.g. dir (data/W1BS/G/1) = 
    [arch.keys  obama.keys  vprice0.keys  vprice1.keys  vprice2.keys  yosemite.keys
    arch.png   obama.png   vprice0.png   vprice1.png   vprice2.png   yosemite.png]
    *.png = image, *.keys = text file with affine keypoints in format: 
        npoints
        x y 5.192*s a11 a12 a21 a22
    *.bmp - hpatches-style column image with pre-extracted patches
    

How to get example results (for now, SIFT, BRIEF and ResizeTo11x11 descriptors are available ): 

    cd data
    ./download_W1BS_dataset.sh
    cd ../code
    ./do_everything.sh

To add your descriptor to benchmark, please add corresponding script to code/descriptors directory.
The provided file should take two arguments: path to input image input_img.bmp and path to output text file with descriptors. 
Output file: one space separated line for one descriptor. Please, see example in code/descriptors/Pixels11.py or code/descriptors/SIFT.py