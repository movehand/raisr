# RAISR

A Python implementation of [RAISR](http://ieeexplore.ieee.org/document/7744595/). Supporting color images.

## How To Use

### Prerequisites

You can install most of the following packages using [pip](https://pypi.python.org/pypi/pip).

* [OpenCV-Python](https://pypi.python.org/pypi/opencv-python)
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [Python Imaging Library (PIL)](http://www.pythonware.com/products/pil/)
* [Matplotlib](https://matplotlib.org/)

### Training

Put your training images in the `train` directory. The training images are the **high resolution (HR)** ones. Run the following command to start training.

```
python train.py
```

In the training stage, the program virtually downscales the high resolution images. The program then trains the model using the downscaled version images and the original HR images. The learned filters will be saved in the root directory of the project.

### Testing

Put your testing images in the `test` directory. Basically, you can use some **low resolution (LR)** images as your testing images. By running the following command, the HR version of your testing images will be generated using RAISR.

```
python test.py
```

The result will be saved in the `results` directory.

## Visualization

### Visualing the learned filters

Uncomment the following line in `train.py`.

```
# filterplot(h, R, Qangle, Qstrength, Qcoherence, patchsize)
```

### Visualing the process of RAISR image upscaling

Uncomment the following line in `test.py`.

```
# plt.show()
```

## Testing Results

Comparing between original image, bilinear interpolation and RAISR:

|         Origin         | Bilinear Interpolation |         RAISR          |
|:----------------------:|:----------------------:|:----------------------:|
|![origin_gray_crop bmp](https://user-images.githubusercontent.com/12198424/27002908-28a69cf2-4e1f-11e7-954d-1135950cd760.png)|![cheap_crop bmp](https://user-images.githubusercontent.com/12198424/27002909-28a7834c-4e1f-11e7-875e-30bb4eaaa0d3.png)|![raisr_gray_crop bmp](https://user-images.githubusercontent.com/12198424/27002910-28ae90f6-4e1f-11e7-83e5-3e63ca07b308.png)|

Other results using images taken from [BSDS500 database](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) and [ArTe-Lab 1D Medium Barcode Dataset](http://artelab.dista.uninsubria.it/downloads/datasets/barcode/medium_barcode_1d/medium_barcode_1d.html):

|     Origin    |     RAISR     |
|:-------------:|:-------------:|
|![origin_crop bmp](https://user-images.githubusercontent.com/12198424/27002925-81cf7f88-4e1f-11e7-8a75-4975db1d37b8.png)|![raisr_crop bmp](https://user-images.githubusercontent.com/12198424/27002926-81d126b2-4e1f-11e7-8814-f2fce323caac.png)|
|![origin_crop bmp](https://user-images.githubusercontent.com/12198424/27002932-a39f6cc2-4e1f-11e7-9dea-3aa79a9d9764.png)|![raisr_crop bmp](https://user-images.githubusercontent.com/12198424/27002933-a3a248ac-4e1f-11e7-9c81-807d3a5eac90.png)|
|![origin_crop bmp](https://user-images.githubusercontent.com/12198424/27002942-c9ba697a-4e1f-11e7-8743-d41be65c09d3.png)|![raisr_crop bmp](https://user-images.githubusercontent.com/12198424/27002943-c9bcefd8-4e1f-11e7-9e55-bf5d93f17a9c.png)|

## References

* Y. Romano, J. Isidoro and P. Milanfar, "RAISR: Rapid and Accurate Image Super Resolution" in IEEE Transactions on Computational Imaging, vol. 3, no. 1, pp. 110-125, March 2017.
* P. Arbelaez, M. Maire, C. Fowlkes and J. Malik, "Contour Detection and Hierarchical Image Segmentation", IEEE TPAMI, Vol. 33, No. 5, pp. 898-916, May 2011.
* Alessandro Zamberletti, Ignazio Gallo and Simone Albertini, "Robust Angle Invariant 1D Barcode Detection", Proceedings of the 2nd Asian Conference on Pattern Recognition (ACPR), Okinawa, Japan, 2013

## License

[MIT.](https://github.com/movehand/raisr/blob/master/LICENSE) Copyright (c) 2017 James Chen
