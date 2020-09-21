import json
import multiprocessing.dummy as multiprocessing
import PIL
from torch.utils.data import Dataset
import cv2
from Utils.mypath import Path
from dataloaders.helpers import *
import torch.nn.functional as F

category_scale = {'person': 2.998630762209037, 'train': 340.6666666666667, 'bicycle': 14.155124653739612, 'motorcycle': 72.65402843601896, \
    'bus': 132.53602305475505, 'car': 1.8922811059907834, 'truck': 101.29955947136564, 'rider': 30.039190071848466}


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def process_info(fname, skip_multicomp=False):
    with open(fname, 'r') as f:
        ann = json.load(f)

    ret = []
    idx = 0
    for obj in ann:
        if obj['label'] not in [
            "car",
            "truck",
            "train",
            "bus",
            "motorcycle",
            "bicycle",
            "rider",
            "person"
        ]:
            continue

        components = obj['components']
        candidates = [c for c in components if len(c['poly']) >= 3]
        candidates = [c for c in candidates if c['area'] >= 100]

        instance = dict()
        instance['polygon'] = [np.array(comp['poly']) for comp in candidates]
        instance['im_size'] = (obj['img_height'], obj['img_width'])
        instance['im_path'] = obj['img_path']
        instance['label'] = obj['label']
        instance['area'] = [c['area'] for c in candidates]
        instance['idx'] = str(idx)
        idx += 1

        if skip_multicomp and len(candidates) > 1:
            continue
        if candidates:
            ret.append(instance)

    return ret


class CityScapesProcessed(Dataset):
    """CityScapes dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, split='train',
                 db_root_dir=Path.db_root_dir('cityscapes-processed'),
                 transform=None,
                 retname=False):
                 
        self.instance_num = 2.0
        self.class_num = 2

        self.split = split
        self.db_root_dir = db_root_dir
        self.retname = retname
        self.transform = transform
        self.ann_list = self.get_ann_list()

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):

        ann = self.ann_list[idx]
        # print('Load image %s'%(ann['im_path']))
        # img = np.array(cv2.imread(ann['im_path']))
        img = np.array(PIL.Image.open(ann['im_path']).convert('RGB')).astype(np.float32)
        gt = np.zeros(ann['im_size'])
        poly = ann['polygon']
        gt = cv2.fillPoly(gt, poly, 1)
        sample = {'image': img, 'gt': gt}

        # import ipdb
        # ipdb.set_trace()

        if self.retname:
            sample['meta'] = {'image': ann['im_path'].split('/')[-1][:-4],
                              'object': ann['idx'],
                              'category': ann['label'],
                              'im_size': ann['im_size']}

        if self.transform is not None:
            sample = self.transform(sample)
        # import ipdb
        # ipdb.set_trace()
        return sample

    def get_ann_list(self):
        numpy_path = '/scratch/ssd001/home/jungao/dataset/cityscapes'
        if not os.path.exists(numpy_path):
            numpy_path = '/scratch/gobi1/jungao/dataset/cityscapes'
        ann_list_path = os.path.join(numpy_path, self.split + '_processed_ann_list_sort.npy')
        if os.path.exists(ann_list_path):
            return np.load(ann_list_path).tolist()
        else:
            # preprocessing
            print("Preprocessing of CityScapes Dataset. This would be done only once. ")
        data_dir = os.path.join(self.db_root_dir, self.split)
        ann_path_list = recursive_glob(data_dir, suffix='.json')
        ann_path_list.sort()
        print('##### ann_list sorted! #####')
        pool = multiprocessing.Pool(4)
        ann_list = pool.map(process_info, ann_path_list)
        ann_list = [obj for ann in ann_list for obj in ann]
        np.save(ann_list_path, ann_list)
        return ann_list

def multi_collate_fn(batch_list):
    keys = batch_list[0].keys()
    collated = {}
    # import ipdb
    # ipdb.set_trace()
    all_valid = [np.array(b['valid']) for b in batch_list]
    all_valid = np.concatenate(all_valid)
    collated['valid'] = all_valid

    for key in keys:
        if key == 'valid': continue
        val = [item[key] for item in batch_list]
        t = type(batch_list[0][key])

        if t is np.ndarray:
            try:
                val = torch.from_numpy(np.stack(val, axis=0))
            except:
                # for items that are not the same shape
                # for eg: orig_poly
                val = [item[key] for item in batch_list]
        if t is torch.Tensor:
            try:
                val = torch.cat([v for v in val], dim = 0)
            except:
                val = [item[key] for item in batch_list]

        collated[key] = val

    return collated

class CityScapesProcessedStretchMulticomp(Dataset):
    """CityScapes dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True, split='train',
                 db_root_dir=Path.db_root_dir('cityscapes-processed'),
                 transform=None,
                 retname=True,
                 min_poly=3,
                 select_maximum=False,
                 inference=False):

        self.select_maximum = select_maximum
        self.train = train
        self.split = split
        self.db_root_dir = db_root_dir
        self.retname = retname
        self.transform = transform
        self.min_poly = min_poly
        self.ann_list = self.get_ann_list()
        self.inference = inference

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        ann = self.ann_list[idx]
        img = np.array(PIL.Image.open(ann['im_path']).convert('RGB')).astype(np.float32)
        gt_list = []
        max_area = 0.0
        max_area_idx = 0
        for idx, poly in enumerate(ann['polygon']):
            gt = np.zeros(ann['im_size'])
            gt = cv2.fillPoly(gt, [poly], 1)
            gt_list.append(gt)
            area = ann['area'][idx]
            if area > max_area:
                max_area_idx = idx
                max_area = max_area
        if self.select_maximum:
            gt = np.asarray(gt_list)
            gt = np.sum(gt, axis=0)
            gt[gt > 0.5] = 1
            gt_list = [gt]
            gt_polygon = [ann['polygon'][max_area_idx]]
        elif not self.inference:
            n_all = len(ann['polygon'])
            select_idx = np.random.choice(n_all)
            gt_polygon = [ann['polygon'][select_idx]]
            gt_list = [gt_list[select_idx]]
        else:
            gt_polygon = ann['polygon']

        sample = {'image': img, 'gt': gt_list, 'gt_polygon': gt_polygon}

        if self.retname:
            sample['meta'] = {'image': ann['im_path'].split('/')[-1][:-4],
                              'object': ann['idx'],
                              'category': ann['label'],
                              'im_size': ann['im_size'],
                              'ori_ann': ann
                              }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_ann_list(self):
        numpy_path = '/scratch/ssd001/home/jungao/dataset/cityscapes'
        if not os.path.exists(numpy_path):
            numpy_path = '/scratch/gobi1/jungao/dataset/cityscapes'
        ann_list_path = os.path.join(numpy_path, self.split + '_processed_ann_list_sort_with_area_new.npy')
        if os.path.exists(ann_list_path):
            return np.load(ann_list_path, allow_pickle=True).tolist()
        else:
            # preprocessing
            print("Preprocessing of CityScapes Dataset. This would be done only once. ")
        data_dir = os.path.join(self.db_root_dir, self.split)
        ann_path_list = recursive_glob(data_dir, suffix='.json')
        ann_path_list.sort()
        print('##### ann_list sorted! #####')
        pool = multiprocessing.Pool(4)
        ann_list = pool.map(process_info, ann_path_list)
        ann_list = [obj for ann in ann_list for obj in ann]
        np.save(ann_list_path, ann_list)
        return ann_list



def check_positive(poly):
    area = (poly[:-1, 0] + poly[1:, 0]) * (poly[:-1, 1] - poly[1:, 1])
    area = np.sum(area)
    area += ((poly[-1, 0] + poly[0, 0]) * (poly[-1, 1] - poly[0, 1]))
    return area
