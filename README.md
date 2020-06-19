## BSTOOL
bstool is a Python library for Building Segmentation.

It will provide the following functionalities.

- Basic parse and dump functions for building segmentation dataset
- Evaluation tools
- Visualization tools
- Dataset convert

### Requirements

- Python 3.6+
- Pytorch 1.1+
- CUDA 9.0+
- [mmcv](https://github.com/open-mmlab/mmcv)

### Installation
```
git clone https://github.com/jwwangchn/bstool.git
cd bstool
python setup.py develop
```

### Future works
- [ ] Parsing function for shp file


### Structure
- scripts
- tests
- tools
- bstool
    - structures:       structures for object detection
    - cnn:              CNN Modules by PyTorch
    - models:           Models for detection, classification and segmentation
    - datasets:         Create and read data from dataset
    - transforms:       bbox and image transformation
        - image
        - bbox
    - fileio:           file operation
        - image
        - file
        - label
    - visualization:    code for visualization
        - image
    - utils:            tools for other tasks
        - path
        - config
        - progressbar