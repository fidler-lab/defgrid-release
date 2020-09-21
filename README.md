# Beyond Fixed Grid: Learning Geometric Image Representation with a Deformable Grid

This code branch provides an example of using DefGrid for learnable feature pooling 
on Cityscapes semantic segmentation benchmark.


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

# License

This work is licensed under a *CC BY-NC-SA 4.0* License.

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

