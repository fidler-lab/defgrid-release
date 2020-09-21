# Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid

This is the official PyTorch implementation of Deformable Grid (ECCV 2020).  For technical details, please refer to:  
----------------------- ------------------------------------
**Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid**  
[Jun Gao](http://www.cs.toronto.edu/~jungao/) <sup>1,2,3</sup>, [Zian Wang](http://www.cs.toronto.edu/~zianwang/)<sup>1,2</sup>, [Jinchen Xuan]()<sup>4</sup>, [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)<sup>1,2,3</sup>   
<sup>1</sup> University of Toronto  <sup>2</sup> Vector Institute <sup>3</sup> NVIDIA  <sup>4</sup> Peking University
**[[Paper](https://arxiv.org/pdf/2008.09269.pdf)] [[Video](https://www.youtube.com/watch?v=_DVxqK3_zlM)] [[Supplementary](http://www.cs.toronto.edu/~jungao/def-grid/files/suppl.pdf)]**

**ECCV 2020**

<img src = "fig/defgrid.png" width="100%"/>

* In modern computer vision, images are typically represented as a fixed uniform grid with some stride and processed via a deep convolutional neural network. We argue that deforming the grid to better align with the high-frequency image content is a more effective strategy. We introduce \emph{Deformable Grid} (DefGrid), a learnable neural network module that predicts location offsets of vertices of a 2-dimensional triangular grid, such that the edges of the deformed grid align with image boundaries.
We showcase our DefGrid in a variety of use cases, i.e., by inserting it as a module at various levels of processing. 
We utilize DefGrid as an end-to-end \emph{learnable geometric downsampling} layer that replaces standard pooling methods for reducing feature resolution when feeding images into a deep CNN. We show significantly improved results at the same grid resolution compared to using CNNs on uniform grids for the task of semantic segmentation.
We also utilize DedGrid at the output layers for the task of object mask annotation, and show that reasoning about object boundaries on our predicted polygonal grid leads to more accurate results over existing pixel-wise and curve-based approaches. We finally showcase {DefGrid} as a standalone module for unsupervised image partitioning, showing superior performance over existing approaches.
----------------------- ------------------------------------

# Environment Setup
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


# Eexample usecases
We provide several usecases on DefGrid, more usecases are on the way! 
We are hoping these usecases can provide insights and improvements on other image-based computer vision tasks as well.
 
## Train DefGrid on Cityscapes Full Image

### Data 
- Download the Cityscapes dataset (leftImg8bit\_trainvaltest.zip) from the official [website](https://www.cityscapes-dataset.com/downloads/) [11 GB]
- Our dataloaders work with our processed annotation files which can be downloaded from [here](http://www.cs.toronto.edu/~amlan/data/polygon/cityscapes.tar.gz).
- From the root directory, run the following command with appropriate paths to get the annotation files ready for your machine
```bash
python scripts/dataloaders/change_paths.py --city_dir <path_to_downloaded_leftImg8bit_folder> --json_dir <path_to_downloaded_annotation_file> --out_dir <output_dir>
```

### Training

Train DefGrid on the whole traininig set.
``` bash
python scripts/train/train_def_grid_full.py --debug false --version train_on_cityscapes_full --encoder_backbone simplenn --resolution 512 1024 --grid_size 20 40 --w_area 0.005
```

## Train DefGrid on Cityscapes-MultiComp


### Training
Train DefGrid on the whole traininig set.
```bash
python scripts/train/train_def_grid_multi_comp.py --debug false --version train_on_cityscapes_multicomp
```

To train on other custom dataloader, please add a new `DataLoader` class according what we have provided. 
the hyper-parameters might also need to change accordingly.  

## Learnable downsampling for semantic segmentation on Cityscapes Full Image. 

We provide the code in this [branch](https://github.com/fidler-lab/deformable-grid-internal/tree/pooling)



# License
This work is licensed under a *CC BY-NC-SA 4.0* License.

If you use this code, please cite:

    @inproceedings{deformablegrid,
    title={Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid},
    author={Jun Gao and Zian Wang and Jinchen Xuan and Sanja Fidler},
    booktitle={ECCV},
    year={2020}
    }
