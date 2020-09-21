import os
import torch
import numpy as np
import scipy.misc as m
import imageio
import random
random.seed(1234)

from torch.utils import data


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

cities = {
    'train':['aachen',	'bochum',	'bremen',	'cologne',  'darmstadt',  'dusseldorf',	'erfurt',	'hamburg',  'hanover',
             'jena',	'krefeld',  'monchengladbach',  'strasbourg',  'stuttgart',  'tubingen',  'ulm'],
    'train_val':[ 'weimar',	'zurich']
}

class cityscapesFullLoader(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        split="train",
        img_size=(1024, 2048),
        img_norm=True,
        version="cityscapes",
        transform=None,
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        if os.path.exists('/h/zianwang/disk/dataset/cityscapes'):
            self.root = '/h/zianwang/disk/dataset/cityscapes'
        elif os.path.exists('/scratch/ssd001/home/linghuan/datasets/dataset/cityscapes'):
            self.root = '/scratch/ssd001/home/linghuan/datasets/dataset/cityscapes'
        else:
            raise ValueError('cityscapes-full data does not exist!')
        self.split = split
        self.img_norm = img_norm
        self.n_classes = 250
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        if split == 'train':
            self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
            self.filter_city(split)
        elif split == 'train_val':
            self.images_base = os.path.join(self.root, "leftImg8bit", 'train')
            self.annotations_base = os.path.join(self.root, "gtFine", 'train')
            self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
            self.filter_city(split)
        elif split == 'val':
            self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.files[split].sort()
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 19 #normalize 19.7.26
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.transform = transform

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("\nFound %d %s images" % (len(self.files[split]), split))

    def filter_city(self, split):
        new_files = []
        for f in self.files[split]:
            for c in cities[split]:
                if c in f:
                    new_files.append(f)
                    break
        self.files[split] = new_files

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = imageio.imread(img_path)
        img = np.array(img, dtype=np.float32) / 255.  # normalize

        lbl = imageio.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        sample = dict()
        sample['image'] = img
        sample['gt'] = lbl
        sample = self.transform(sample)
        # print(sample['crop_image'].max(), sample['crop_image'].min())
        return sample


    def random_crop(self, img, lbl):
        img_h, img_w = img.shape[0], img.shape[1]
        top = random.randint(0, img_h - self.crop_size)
        left = random.randint(0, img_w - self.crop_size)
        crop_img = img[top:top + self.crop_size, left:left + self.crop_size]
        crop_lbl = lbl[top:top + self.crop_size, left:left + self.crop_size]
        return crop_img, crop_lbl



    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
