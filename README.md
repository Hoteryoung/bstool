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
- [x] Parse shapefile
- [x] Show polygons or show polygons on corresponding image
- [x] Merge separate polygon in original shapefile
- [x] Parse ignore file (png)
- [x] Add ignore flag to properties
- [x] Show ignored polygons
- [x] Split large image and corresponding polygons
- [x] Convert Json to COCO format
- [ ] COCO format visualization codes
- [ ] Merge detection results on small image to original image
- [ ] Generate CSV file for evaluation (xian fine)
- [ ] Evaluation codes for semantic segmentation
- [ ] Evaluation codes for instance segmentation
- [ ] Evaluation codes for offset
- [ ] Visualization code for ground truth CSV file and prediction CSV file


### Structure
- scripts
- tests
- tools
- bstool
    - datasets:         Create and read data from dataset
    - transforms:       bbox and image transformation
        - image
        - bbox
    - visualization:    code for visualization
        - image
        - mask
        - bbox
    - utils:            tools for other tasks
        - path