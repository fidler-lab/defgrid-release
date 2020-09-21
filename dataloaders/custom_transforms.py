import torch, cv2
from scipy import misc, ndimage
import numpy.random as random
import numpy as np
import dataloaders.helpers as helpers
import copy
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class CropCenter(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, keys=['image', 'gt'],
                 crop_size=[332, 500], drop_origin=True):

        self.keys = keys
        self.crop_size = crop_size
        self.drop_origin = drop_origin

    def __call__(self, sample):
        

        for i, k in enumerate(self.keys):
            if not k in sample.keys():
                continue
            elem = sample[k]
            # import ipdb
            # ipdb.set_trace()
            if i == 0:
                h, w = elem.shape[0], elem.shape[1]
                new_h = self.crop_size[0]
                new_w = self.crop_size[1]
        
                top = (h - new_h) // 2
                left = (w - new_w) // 2

            new_elem = elem[top: top + new_h,
                        left: left + new_w]

            sample['crop_' + k] = new_elem
            
            if self.drop_origin:
                del sample[k]

        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'

class MakeEdgeMask(object):

    def __init__(self, original_size=(1024, 2048), new_size=(224, 224), key_name='edge_mask', delete=True):
        self.original_size = original_size
        self.new_size = new_size
        self.key_name = key_name

        self.x_scale = float(new_size[1]) / float(original_size[1])
        self.y_scale = float(new_size[0]) / float(original_size[0])

    def __call__(self, sample):
        
        poly = sample['poly']
        
        new_poly = []
        for component in poly:
            import ipdb; ipdb.set_trace()
            component[:, 0] = component[:, 0] * self.x_scale
            component[:, 1] = component[:, 1] * self.y_scale
            component = component.astype(np.int32)
            new_poly.append(component)
         
        edge_mask = np.zeros(self.new_size)
        cv2.polylines(edge_mask, new_poly, True, [1])
        sample['edge_mask'] = edge_mask
        del sample['poly']
        return sample

class CropFromMask(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, crop_elems=('image', 'gt'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False,
                 dummy=False,
                 for_grid=False,
                 grid_size=20):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad
        self.dummy = dummy  # if True: keep a copy of data without crop.
        self.for_grid = for_grid
        self.grid_size = grid_size

    def __call__(self, sample):
        if self.dummy:
            for elem in self.crop_elems:

                sample['crop_' + elem] = sample[elem].copy()
            return sample

        _target = sample[self.mask_elem]
        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(np.ascontiguousarray(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax,
                                                                                 zero_pad=self.zero_pad, for_grid=self.for_grid, grid_size=self.grid_size)))
            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(np.ascontiguousarray(helpers.crop_from_mask(_img, _tmp_target, relax=self.relax,
                                                                                 zero_pad=self.zero_pad, for_grid=self.for_grid, grid_size=self.grid_size)))
            if len(_crop) == 1:
                sample['crop_' + elem] = _crop[0]
            else:
                sample['crop_' + elem] = _crop
        sample['crop_mask'] = sample['crop_gt']
        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'

class NormalMap(object):
    def __init__(self, keys=['crop_image']):
        self.keys = keys

    def __call__(self, sample):
        for k in self.keys:

            elem = sample[k]
            h, w, c = elem.shape
            elem = elem.mean(axis=-1)
            normal_map_x1 = np.zeros([h, w])
            normal_map_x2 = np.zeros([h, w])
            normal_map_x = np.zeros([h, w])
            normal_map_x1[:, :w - 1] = elem[:, 1:] - elem[:, :w - 1]
            normal_map_x2[:, 1:] = elem[:, 1:] - elem[:, :w-1]
            mask = abs(normal_map_x1) > abs(normal_map_x2)
            normal_map_x = mask * normal_map_x1 + (1 - mask) * normal_map_x2
            
            normal_map_y1 = np.zeros([h, w])
            normal_map_y2 = np.zeros([h, w])
            normal_map_y = np.zeros([h, w])
            normal_map_y1[:h - 1, :] = elem[1:, :] - elem[:h - 1, :]
            normal_map_y2[1:, :] = elem[1:, :] - elem[:h - 1, :]
            mask = abs(normal_map_y1) > abs(normal_map_y2)
            normal_map_y = mask * normal_map_y1 + (1 - mask) * normal_map_y2

            normal_map_x = normal_map_x.reshape(h, w, 1)
            normal_map_y = normal_map_y.reshape(h, w, 1)
            normal_map = np.concatenate((normal_map_x, normal_map_y), axis=-1)

            sample['normal_' + k] = normal_map
        return sample

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem or elem == 'scale' or elem == 'poly':
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'

class ImageGTResizeAndDelete(object):
    def __init__(self, resolution=(512,512), keys=['crop_gt', 'crop_image'], delete=True, original_keys=['image', 'gt']):

        self.keys = keys
        self.resolution = resolution
        self.delete = delete
        self.original_keys = original_keys

    def __call__(self, sample):
        # import ipdb
        # ipdb.set_trace()
        for k in sample.keys():
            if k == 'meta' or k == 'scale' or k == 'poly':
                continue
           
            elif k in self.keys:

                elem = sample[k]
                elem = cv2.resize(elem, self.resolution, cv2.INTER_LINEAR)
                sample[k] = np.ascontiguousarray(elem)
                
            elif (k in self.original_keys and self.delete) or k not in self.original_keys:
                del sample[k]
            

        return sample


class GetSemanticEdgeMultiComp(object):
    def __init__(self, size=224):
        self.size = size

    def __call__(self, sample):
        crop_gt = np.zeros((self.size, self.size, len(sample['crop_polygon'])))
        for idx, poly in enumerate(sample['crop_polygon']):
            mask = np.zeros((self.size, self.size))
            poly = copy.deepcopy(poly) * self.size
            poly = np.round(poly).astype(np.int32)
            if not isinstance(poly, np.ndarray):
                poly = np.array(poly, dtype=np.int32)
            poly = poly.astype(np.int32)
            cv2.polylines(mask, [poly], isClosed=True,  color=1)
            crop_gt[..., idx] = mask
        sample['semantic_edge'] = crop_gt
        return sample



class MakeGT(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        crop_gt = np.zeros((self.size, self.size, len(sample['crop_polygon'])))
        for idx, poly in enumerate(sample['crop_polygon']):

            mask = np.zeros((self.size, self.size))
            poly = copy.deepcopy(poly) * self.size
            poly = np.round(poly).astype(np.int32)
            if not isinstance(poly, np.ndarray):
                poly = np.array(poly, dtype=np.int32)
            poly = poly.astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)
            crop_gt[...,idx] = mask
        sample['crop_gt'] = crop_gt
        return sample


class MakePolygon(object):
    def __init__(self, cp_num):
        self.cp_num = cp_num

    def __call__(self, sample):

        # create circle polygon data
        pointsnp = np.zeros(shape=(self.cp_num, 2), dtype=np.float32)
        for i in range(self.cp_num):
            thera = 1.0 * i / self.cp_num * 2 * np.pi
            x = np.cos(thera)
            y = -np.sin(thera)
            pointsnp[i, 0] = x
            pointsnp[i, 1] = y

        fwd_poly = (0.7 * pointsnp + 1) / 2

        arr_fwd_poly = np.ones((self.cp_num, 2), np.float32) * 0.
        arr_fwd_poly[:, :] = fwd_poly

        sample['init_polygon'] = [copy.deepcopy(arr_fwd_poly) for i in range(len(sample['crop_polygon']))]

        return sample



class CropFromMaskStretchMulticomp(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, crop_elems=('image', 'gt', 'gt_polygon'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False,
                 dummy=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad
        self.dummy = dummy  # if True: keep a copy of data without crop.

    def __call__(self, sample):
        _target_list = sample[self.mask_elem]
        sample['crop_image'] = []
        sample['crop_gt'] = []
        sample['crop_polygon'] = []
        sample['valid'] = []
        for i_comp, _target in enumerate(_target_list):
            _target = np.expand_dims(_target, axis=-1)
            elem = 'image'
            _img = sample[elem]
            if _img.ndim == 2:
                _img = np.expand_dims(_img, axis=-1)
            _crop_img = []
            for k in range(0, _target.shape[-1]):
                if np.max(_target[..., k]) == 0:
                    _crop_img.append(np.zeros(_img.shape, dtype=_img.dtype))
                else:
                    _tmp_target = _target[..., k]
                    _crop_img.append(helpers.crop_from_mask(_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            sample['crop_image'].append(_crop_img[0])

            elem = 'gt'
            _img = sample[elem][i_comp]
            _crop_img = []
            if _img.ndim == 2:
                _img = np.expand_dims(_img, axis=-1)
            for k in range(0, _target.shape[-1]):
                _tmp_img = _img[..., k]
                _tmp_target = _target[..., k]
                if np.max(_target[..., k]) == 0:
                    _crop_img.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                else:
                    _crop_img.append(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            sample['crop_gt'].append(_crop_img[0])
            # import ipdb
            # ipdb.set_trace()
            elem = 'gt_polygon'
            gt_polygon = sample[elem][i_comp]
            valid = 1
            try:
                _crop_polygon = helpers.crop_from_mask_polygon(gt_polygon, _img, _target[..., 0], relax=self.relax, zero_pad=self.zero_pad)
            except:
                _crop_polygon = np.asarray([[0.5,0.5],[0.5,0.6],[0.6,0.6],[0.6,0.5]])
                valid = 0
            sample['valid'].append(valid)
            sample['crop_polygon'].append(_crop_polygon)
        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'


class FixedResizeStretchMulticomp(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample
        elems = list(sample.keys())
        for elem in elems:
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)



class GTBinarization(object):
    def __init__(self, keys=['crop_gt'], thre=0.5, output_value=1.0):
        self.keys = keys
        self.thre = thre
        self.output_value = output_value

    def __call__(self, sample):
                
        for k in self.keys:
            s = sample[k]
            s = s > self.thre
            s = self.output_value * s.astype(np.float)
            sample[k] = s
        
        return sample

class ToImage(object):
    """
    Return the given elements between 0 and 255
    """
    def __init__(self, norm_elem='image', custom_max=1.0):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'

class ConcatInputs(object):

    def __init__(self, elems=('image', 'point')):
        self.elems = elems

    def __call__(self, sample):

        res = sample[self.elems[0]]

        for elem in self.elems[1:]:
            assert(sample[self.elems[0]].shape[:2] == sample[elem].shape[:2])

            # Check if third dimension is missing
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            res = np.concatenate((res, tmp), axis=2)

        sample[self.elems[0]] = res

        return sample

    def __str__(self):
        return 'ConcatInputs:'+str(self.elems)

class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())
        for elem in elems:
            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue
            if 'extreme_points_coord' in elem and elem in self.resolutions:
                bbox = sample['bbox']
                crop_size = np.array([bbox[3]-bbox[1]+1, bbox[4]-bbox[2]+1])
                res = np.array(self.resolutions[elem]).astype(np.float32)
                sample[elem] = np.round(sample[elem]*res/crop_size).astype(np.int)
                continue
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
            else:
                del sample[elem]

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)



class BilateralFiltering(object):
    def __init__(self, keys, w1=9, w2=75, w3=75):
        self.keys = keys
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def __call__(self, sample):
        for k in self.keys:
            elem = sample[k]
            bifilter_elem = cv2.bilateralFilter(elem, self.w1, self.w2, self.w3)
            sample['bifilter_' + k] = bifilter_elem

        return sample

class BilateralFilteringMultiComp(object):
    def __init__(self, keys, w1=9, w2=75, w3=75):
        self.keys = keys
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def __call__(self, sample):
        for k in self.keys:
            elem = sample[k]
            all = np.zeros_like(elem)
            for i in range(elem.shape[-1]):
                bifilter_elem = cv2.bilateralFilter(elem[..., i], self.w1, self.w2, self.w3)
                all[..., i] = bifilter_elem
            sample['bifilter_' + k] = all

        return sample

class ZeroOneNormalize(object):
    """
    Normalize pixel value to 0~1 range
    """
    def __init__(self, normalize_dict):
        self.normalize_dict = normalize_dict

    def __call__(self, sample):

        for k in sample.keys():
            
            if k in self.normalize_dict:
                sample[k] = sample[k] / self.normalize_dict[k]

        return sample

class ZeroOneNormalizeMultiComp(object):
    """
    Normalize pixel value to 0~1 range
    """

    def __init__(self, normalize_dict):
        self.normalize_dict = normalize_dict

    def __call__(self, sample):

        for k in sample.keys():
            if k in self.normalize_dict:
                sample[k] = sample[k] / self.normalize_dict[k]

        return sample
        
class CropRandom(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, keys=['image', 'gt'],
                 crop_size=[332, 500], drop_origin=True):

        self.keys = keys
        self.crop_size = crop_size
        self.drop_origin = drop_origin

    def __call__(self, sample):
        

        for i, k in enumerate(self.keys):
            if not k in sample.keys():
                continue
            elem = sample[k]
            # import ipdb
            # ipdb.set_trace()
            if i == 0:
                h, w = elem.shape[0], elem.shape[1]
                new_h = self.crop_size[0]
                new_w = self.crop_size[1]
        
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)

            new_elem = elem[top: top + new_h,
                        left: left + new_w]

            sample['crop_' + k] = new_elem
            
            if self.drop_origin:
                del sample[k]

        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem or elem == 'poly':
                continue

            elif 'bbox' in elem or elem == 'scale':
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]
            
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.FloatTensor(tmp.copy()) #prevent the negative stride error

        return sample

    def __str__(self):
        return 'ToTensor'


class ToTensorStretchMulticomp(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]
            if elem == 'init_polygon_list':
                # n_p = len(tmp[0])
                sample[elem] = [torch.from_numpy(t) for t in tmp]
                # for i in range(n_p):
                #     sample[elem].append(torch.cat([torch.from_numpy(t[i]) for t in tmp]))
                continue

            if 'polygon' in elem:
                # gt_polygon: ground truth polygon
                # crop_polygon: polygon in the cropped image
                # sample_polygon: polygon that sampled to a same length.
                # init_polygon: initial polygon
                sample[elem] = [torch.from_numpy(t).unsqueeze(0) for t in tmp]
                if 'sample' in elem or 'init' in elem:
                    sample[elem] = torch.cat(sample[elem], dim = 0)
                continue

            if elem == 'gt':
                # only the ground truth has this shape
                tmp = [torch.from_numpy(t).unsqueeze(0).unsqueeze(0) for t in tmp]
                # sample[elem] = torch.cat(tmp, dim = 0) # treat it as gt
                sample[elem] = tmp # in Ade the size is different for that in cityscapes, so error exists here
                continue

            if elem == 'crop_image' or elem == 'extreme_points' or elem == 'concat' or elem == 'canny_edge' or elem == 'bifilter_crop_image':
                tmp = tmp.transpose((3, 2, 0, 1))
                sample[elem] = torch.FloatTensor(tmp)
                continue

            if elem == 'image' :
                tmp = tmp.transpose((2, 0, 1))
                sample[elem] = torch.FloatTensor(tmp).unsqueeze(0)
                continue
            if elem == 'crop_gt' or elem == 'dt' or elem == 'sdt' or elem == 'semantic_edge':
                tmp = tmp.transpose((2, 0, 1))
                sample[elem] = torch.FloatTensor(tmp)
                continue

            if elem == 'vertex_mask' or elem == 'edge_mask':
                sample[elem] = torch.from_numpy(tmp).float()
                continue
            if elem == 'valid':
                continue
            raise ValueError

        return sample

    def __str__(self):
        return 'ToTensor'


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, key, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std
        self.key = key

    def __call__(self, sample):
        image = sample[self.key]
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        sample[self.key] = image
        return sample