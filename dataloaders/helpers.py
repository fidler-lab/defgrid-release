import os

import torch
import cv2
import random
import numpy as np
import copy

def tens2image(im):
    if im.size()[0] == 1:
        tmp = np.squeeze(im.numpy(), axis=0)
    else:
        tmp = im.numpy()
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))

def gtmask2onehot(gtmask):
    batch_size, channel_num, h, w = gtmask.shape

    max_index = gtmask.max().long() + 1
    onehot = torch.zeros(batch_size, max_index, h, w, device=gtmask.device)
    onehot.scatter_(dim=1, index=gtmask.long(), value=1.0)

    return onehot

def get_bbox(mask, points=None, pad=0, zero_pad=False):
    # we need to adjust pad if we want to do for the bigger grid
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None
    # if for_grid:
    #
    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return x_min, y_min, x_max, y_max
    
def crop2fullmask(crop_mask, bbox, im=None, im_size=None, zero_pad=False, relax=0, mask_relax=True,
                  interpolation=cv2.INTER_LINEAR, scikit=False, bg_value=20):
    if scikit:
        from skimage.transform import resize as sk_resize
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size'
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borers of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Bounding box of initial mask
    bbox_init = (bbox[0] + relax,
                 bbox[1] + relax,
                 bbox[2] - relax,
                 bbox[3] - relax)

    if zero_pad:
        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])
    else:
        assert((bbox == bbox_valid).all())
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    if scikit:
        crop_mask = sk_resize(crop_mask, (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), order=0, mode='constant').astype(crop_mask.dtype)
    else:

        crop_mask = cv2.resize(crop_mask, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)
    result_ = np.ones(im_si) * bg_value
    result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
        crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]

    result = np.ones(im_si) * bg_value
    if mask_relax:
        result[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1] = \
            result_[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1]
    else:
        result = result_

    return result


def crop_from_bbox(img, bbox, zero_pad=False, return_crop_bbox=False):
    # Borders of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    if zero_pad:
        # Initialize crop size (first 2 dimensions)
        crop = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=img.dtype)

        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])

    else:
        assert(bbox == bbox_valid)
        crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    img = np.squeeze(img)
    if img.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]
    if return_crop_bbox:
        return crop, bbox_valid
    return crop

def crop_from_mask(img, mask, relax=0, zero_pad=False, for_grid=False, grid_size=20):
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(img.shape[:2])), interpolation=cv2.INTER_LINEAR)

    assert(mask.shape[:2] == img.shape[:2])

    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad, for_grid=for_grid, grid_size=grid_size)

    if bbox is None:
        return None

    crop = crop_from_bbox(img, bbox, zero_pad)

    return crop

def create_init_polygon(cp_num):
    # create circle polygon data
    pointsnp = np.zeros(shape=(cp_num, 2), dtype=np.float32)
    for i in range(cp_num):
        thera = 1.0 * i / cp_num * 2 * np.pi
        x = np.cos(thera)
        y = -np.sin(thera)
        pointsnp[i, 0] = x
        pointsnp[i, 1] = y

    fwd_poly = (0.7 * pointsnp + 1) / 2

    arr_fwd_poly = np.ones((cp_num, 2), np.float32) * 0.
    arr_fwd_poly[:, :] = fwd_poly
    return arr_fwd_poly

EPS = 1e-7

def crop_from_bbox_polygon(polygon, img, bbox, zero_pad=False):
    # Borders of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Offsets for x and y
    offsets = (-bbox[0], -bbox[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    new_polygon = np.zeros_like(polygon).astype(np.float)
    polygon = polygon.astype(np.float)
    new_polygon[:, 1] = (polygon[:, 1] - bbox_valid[1] + inds[1]) / float(bbox_valid[3] - bbox_valid[1] + 1)
    new_polygon[:, 0] = (polygon[:, 0] - bbox_valid[0] + inds[0]) / float(bbox_valid[2] - bbox_valid[0] + 1)
    new_polygon = np.clip(new_polygon, 0 + EPS, 1 - EPS)

    try:
        assert np.sum((new_polygon > 1).astype(np.float)) == 0
        assert np.sum((new_polygon < 0).astype(np.float)) == 0
    except:
        print('==> Polygon')
        print(polygon)
        print('==> BBox')
        print(bbox)
        print('==> Index')
        print(inds)
        print('==> BBox valid')
        print(bbox_valid)
        # print('==> me
    return new_polygon


def crop_from_mask_polygon(poly, img, mask, relax=0, zero_pad=False):

    assert (mask.shape[:2] == img.shape[:2])

    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad)

    crop = crop_from_bbox_polygon(poly, img, bbox, zero_pad)

    return crop


def make_gt(img, labels, sigma=10, one_mask_per_point=False):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h//2, w//2), sigma=sigma)
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]
        if one_mask_per_point:
            gt = np.zeros(shape=(h, w, labels.shape[0]))
            for ii in range(labels.shape[0]):
                gt[:, :, ii] = make_gaussian((h, w), center=labels[ii, :], sigma=sigma)
        else:
            gt = np.zeros(shape=(h, w), dtype=np.float64)
            for ii in range(labels.shape[0]):
                gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))

    gt = gt.astype(dtype=img.dtype)

    return gt

def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def extreme_points(mask, pert):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)+pert)), # left
                     find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)-pert)), # right
                     find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)+pert)), # top
                     find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)-pert)) # bottom
                     ])

def fixed_resize(sample, resolution, flagval=None):

    resolution = tuple(resolution)
    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            # flagval = cv2.INTER_CUBIC
            # TODO: https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
            flagval = cv2.INTER_LINEAR
    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(round(float(resolution)/np.min(sample.shape[:2])*np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):

        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample


def gather_feature(id, feature):
    feature_id = id.unsqueeze_(2).long().expand(id.size(0),
                                                id.size(1),
                                                feature.size(2)).detach()

    cnn_out = torch.gather(feature, 1, feature_id).float()

    return cnn_out

