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
- pycocotools (pip install lvis@git+https://github.com/open-mmlab/cocoapi.git#subdirectory=lvis)

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
- [x] COCO format visualization codes
- [x] Merge detection results on small image to original image
- [x] Generate CSV file for evaluation (xian fine)
- [x] Evaluation codes for semantic segmentation
- [x] Evaluation codes for instance segmentation
- [x] Visualization code for ground truth CSV file and prediction CSV file
- [x] Visualization code for TP, FN, FP (pred and gt)
- [x] Evaluation codes for offset
- [x] Evaluation codes for height


### Structure
- demo:                 Simple demo to illustrate how to use the corresponding functions
- tools:                Put the codes for projects
- bstool
    - datasets:         Parse and dump data for dataset (e.g. shapefile, coco, json file)
    - evaluation:       Detection and segmentation evaluation codes
    - generation:       Generation the specific objects (e.g. empty images, polygons on pixel annotation)
    - ops:              Operators (e.g. bbox nms, mask nms)
    - transforms:       bbox, mask, image transformation functions (e.g. mask to polygon, polygon to mask)
        - image
        - bbox
        - mask
    - visualization:    Codes for visualization
        - image
        - mask
        - bbox
        - color
        - utils
    - utils:            Small useful tools for general tasks
        - path
        - mask
    - csrc:             Codes for cuda operation (Note: if you meet the compilation errors, please comment this section in setup.py)