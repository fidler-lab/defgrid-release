# Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid

This code branch provides an example of using DefGrid for learnable feature pooling 
on Cityscapes semantic segmentation benchmark. For technical details, please refer to:  
----------------------- ------------------------------------
**Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid**  
[Jun Gao](http://www.cs.toronto.edu/~jungao/) <sup>1,2,3</sup>, [Zian Wang](http://www.cs.toronto.edu/~zianwang/)<sup>1,2</sup>, [Jinchen Xuan]()<sup>4</sup>, [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)<sup>1,2,3</sup>   
<sup>1</sup> University of Toronto  <sup>2</sup> Vector Institute <sup>3</sup> NVIDIA  <sup>4</sup> Peking University
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


## Data preparation

- Download the Cityscapes dataset (leftImg8bit\_trainvaltest.zip) from the official [website](https://www.cityscapes-dataset.com/downloads/) [11 GB]

```
mkdir dataset
ln -s /<path_to_cityscapes_datase> dataset/cityscapes
```

- Download the data list from [semseg](https://github.com/hszhao/semseg) repo ([link](https://drive.google.com/drive/folders/1Om9Sg2JlJsd-GI-aMKUMLtVr-Gavnd38)).


## Code structure

```
.
├── ...        				# cloned master branch DefGrid folders
|   
├── lib/sync_bn        			# synchronized batchnorm 
├── util           			# helper function for semantic segmentation
├── DGNet.py				# model arch 
└── gridpool.py        			# training and testing
```


## Training

Run gridpool.py to train the model. One sample command is:

```
python gridpool.py --train_h 1024 --train_w 2048 --batch <batch_size> --exp_name "gridpooling33x33" --grid_size 33 33 
```

## Citation
If you use this code, please cite:

    @inproceedings{deformablegrid,
    title={Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid},
    author={Jun Gao and Zian Wang and Jinchen Xuan and Sanja Fidler},
    booktitle={ECCV},
    year={2020}
    }

-----------------------------------------------------------
The code skeleton is adapted from [semseg](https://github.com/hszhao/semseg) repo.
We thank the authors for releasing their code.  

