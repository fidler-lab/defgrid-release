# Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid

This is the official PyTorch implementation of Deformable Grid (ECCV 2020).  For technical details, please refer to:  
----------------------- ------------------------------------
**Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid**  
[Jun Gao](http://www.cs.toronto.edu/~jungao/), [Zian Wang](http://www.cs.toronto.edu/~zianwang/), [Jinchen Xuan](), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)

University of Toronto 
**[[Paper](https://arxiv.org/abs/2008.09269)] [[Video](https://www.youtube.com/watch?v=_DVxqK3_zlM)] [[Supplementary](http://www.cs.toronto.edu/~jungao/def-grid/files/suppl.pdf)] [[Project](http://www.cs.toronto.edu/~jungao/def-grid/)]**

**ECCV 2020**

<img src = "fig/defgrid.png" width="100%"/>

* In modern computer vision, images are typically represented as a fixed uniform grid with some stride and processed via a deep convolutional neural network. We argue that deforming the grid to better align with the high-frequency image content is a more effective strategy. We introduce \emph{Deformable Grid} (DefGrid), a learnable neural network module that predicts location offsets of vertices of a 2-dimensional triangular grid, such that the edges of the deformed grid align with image boundaries.
We showcase our DefGrid in a variety of use cases, i.e., by inserting it as a module at various levels of processing. 
We utilize DefGrid as an end-to-end \emph{learnable geometric downsampling} layer that replaces standard pooling methods for reducing feature resolution when feeding images into a deep CNN. We show significantly improved results at the same grid resolution compared to using CNNs on uniform grids for the task of semantic segmentation.
We also utilize DedGrid at the output layers for the task of object mask annotation, and show that reasoning about object boundaries on our predicted polygonal grid leads to more accurate results over existing pixel-wise and curve-based approaches. We finally showcase {DefGrid} as a standalone module for unsupervised image partitioning, showing superior performance over existing approaches.
----------------------- ------------------------------------


## License
```
Copyright (C) University of Toronto. Jun Gao, Zian Wang, Jinchen Xuan, Sanja Fidler
All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Permission to use, copy, modify, and distribute this software and its documentation
for any non-commercial purpose is hereby granted without fee, provided that the above
copyright notice appear in all copies and that both that copyright notice and this
permission notice appear in supporting documentation, and that the name of the author
not be used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

## Environment Setup
All the code have been run and tested on Ubuntu 16.04, Python 2.7 (and 3.8), Pytorch 1.1.0 (and 1.2.0), CUDA 10.0, TITAN X/Xp and GTX 1080Ti GPUs

- Go into the downloaded code directory
```bash
cd <path_to_downloaded_directory>
```
- Setup python environment
```bash
conda create --name defgrid
conda activate defgrid
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install opencv-python matplotlib networkx tensorboardx tqdm scikit-image ipdb
```
- Add the project to PYTHONPATH  
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```


## Eexample use cases
We provide several usecases on DefGrid, more usecases are on the way! 
We are hoping these usecases can provide insights and improvements on other image-based computer vision tasks as well.
 
### Train DefGrid on Cityscapes Images

#### Data 
- Download the Cityscapes dataset (leftImg8bit\_trainvaltest.zip) from the official [website](https://www.cityscapes-dataset.com/downloads/) [11 GB]
- Our dataloaders work with our processed annotation files which can be downloaded from [here](http://www.cs.toronto.edu/~amlan/data/polygon/cityscapes.tar.gz).
- From the root directory, run the following command with appropriate paths to get the annotation files ready for your machine
```bash
python scripts/dataloaders/change_paths.py --city_dir <path_to_downloaded_leftImg8bit_folder> --json_dir <path_to_downloaded_annotation_file> --out_dir <output_dir>
```

#### Training

Train DefGrid on the whole traininig set.
```bash
python scripts/train/train_def_grid_full.py --debug false --version train_on_cityscapes_full --encoder_backbone simplenn --resolution 512 1024 --grid_size 20 40 --w_area 0.005
```

### Train DefGrid on Cityscapes "MultiComp" cropped images


#### Training
Train DefGrid on the whole traininig set.
```bash
python scripts/train/train_def_grid_multi_comp.py --debug false --version train_on_cityscapes_multicomp
```


### Train DefGrid on Custom dataset
- Please add a new `DataLoader` class according what we have provided. 
- The mask annotation is not required. DefGrid can be trained with RGB images only. To do that, please add one more tag in the training command: `--add_mask_variance false`, and remove all the variables related to `crop_gt` in the training script.
- To tune the DefGrid, we suggest: 1. Starting from a high regularization (e.g. `--w_area 0.5 --w_laplacian 0.5`), in this case, the grid vertices will be close the initial position, and retrain regularity. 2. Gradually reduce the weight until a satisfactory result obtained.  

### Learnable downsampling for semantic segmentation on Cityscapes Images. 

We provide the code in this [branch](https://github.com/fidler-lab/defgrid-release/tree/pooling)


## Citation

If you use this code, please cite:

    @inproceedings{deformablegrid,
    title={Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid},
    author={Jun Gao and Zian Wang and Jinchen Xuan and Sanja Fidler},
    booktitle={ECCV},
    year={2020}
    }
