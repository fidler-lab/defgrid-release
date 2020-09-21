import os
import sys

import time
import logging
import argparse

import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from util import dataset, transform
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
from util.f_boundary import eval_mask_boundary

from Utils.plot_sample import plot_deformed_lattice_on_image
from DGNet import DGNet


plt.switch_backend('agg')
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(s):
    lower = s.lower()
    assert(lower == 'true' or lower == 'false')
    return lower == 'true'


def get_parser():
    # load defgrid params
    args = torch.load('defgrid_args.pth')
    args_dict = vars(args)

    # new params start here
    parser = argparse.ArgumentParser()

    # grid
    parser.add_argument('--grid_size', type=int, nargs='+', default=[33, 33])
    parser.add_argument('--add_mask_variance', type=str2bool, default=True)
    parser.add_argument('--mask_coef', type=float, default=3.0)
    parser.add_argument('--interp', type=str, default="area", choices=['area', 'nearest'])

    # data
    parser.add_argument('--data_root', type=str, default='dataset/cityscapes')
    parser.add_argument('--train_list', type=str, default='dataset/cityscapes/list/fine_train.txt')
    parser.add_argument('--val_list', type=str, default='dataset/cityscapes/list/fine_val.txt')
    parser.add_argument('--classes', type=int, default=19)
    parser.add_argument('--train_h', type=int, default=1024)
    parser.add_argument('--train_w', type=int, default=2048)
    parser.add_argument('--scale_min', type=float, default=0.5)
    parser.add_argument('--scale_max', type=float, default=2.0)
    parser.add_argument('--rotate_min', type=float, default=-10)
    parser.add_argument('--rotate_max', type=float, default=10)
    parser.add_argument('--workers', type=int, default=8)

    # train
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--lr_mult', type=float, default=0.01)
    parser.add_argument('--sync_bn', type=str2bool, default=True)
    parser.add_argument('--ignore_label', type=int, default=255)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--batch_val', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--aux_weight', type=float, default=0.4)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--manual_seed', type=int, default=None)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--evaluate', type=str2bool, default=True)

    # test
    parser.add_argument('--eval_only', type=str2bool, default=False)
    parser.add_argument('--flip', type=str2bool, default=False)
    parser.add_argument('--eval_ckpt', type=str, default='None')
    parser.add_argument('--eval_thresholds', type=int, nargs='+', default=[4,])

    newargs = parser.parse_args()
    args_dict.update(vars(newargs))     # update args

    if args.exp_name is not None:
        args.save_path = 'exp/{}/model'.format(args.exp_name)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        else:
            print('FOLDER EXISTS!!! ')
    return args


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    args = get_parser()
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manualSeed)
        torch.cuda.manual_seed(args.manualSeed)
        torch.cuda.manual_seed_all(args.manualSeed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    main_worker(args)


def main_worker(argss):
    global args, logger, writer
    args = argss
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    ''' model '''
    dgpsp = DGNet(args)

    ''' optimizer '''
    params_list = []
    for module in dgpsp.modules_dg:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * args.lr_mult))
    for module in dgpsp.modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in dgpsp.modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    args.index_split = 3
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    dgpsp = torch.nn.DataParallel(dgpsp.cuda())

    ''' Load weights '''
    # model ckpt
    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            dgpsp.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    filename = None
    for epoch_log in range(200, -1, -1):
        if os.path.isfile(args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'):
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            break
    if not args.resume:
        args.resume = filename
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            dgpsp.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    ''' data '''
    value_scale = 1
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.Resize((args.train_h, args.train_w), interpolation=cv2.INTER_AREA if args.interp is 'area' else cv2.INTER_NEAREST),
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.NormalizeWithOriginal(mean=mean, std=std)])
    train_data = dataset.CityscapesData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    if args.evaluate:
        val_transform = transform.Compose([
            transform.Resize((args.train_h, args.train_w), interpolation=cv2.INTER_AREA if args.interp is 'area' else cv2.INTER_NEAREST),
            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.NormalizeWithOriginal(mean=mean, std=std)])
        val_data = dataset.CityscapesData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_val, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.eval_only:
        assert os.path.isfile(args.eval_ckpt)
        logger.info("=> loading checkpoint '{}'".format(args.eval_ckpt))
        checkpoint = torch.load(args.eval_ckpt, map_location=lambda storage, loc: storage.cuda())
        dgpsp.load_state_dict(checkpoint['state_dict'], strict=False)
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        validate(val_loader, dgpsp, criterion)
        exit(0)

    start_epoch = args.start_epoch
    for epoch in range(start_epoch, args.epochs):  # training loop!
        epoch_log = epoch + 1

        # train
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, dgpsp, optimizer, epoch)
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        # save ckpt
        if (epoch_log % args.save_freq == 0):
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': dgpsp.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                if os.path.exists(deletename):
                    os.remove(deletename)
        # validate
        if args.evaluate and (epoch_log % 5 == 0):
            criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
            loss_val, mIoU_val, mAcc_val, allAcc_val, mean_F, mean_Fc = validate(val_loader, dgpsp, criterion)
            writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('full_image_mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('full_image_mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('full_image_allAcc_val', allAcc_val, epoch_log)
            for thres in args.eval_thresholds:
                writer.add_scalar('full_image_F_{}_val'.format(thres), mean_F[thres], epoch_log)


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    for i, (sample, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.unsqueeze(1)
        original_image = sample['original'].cuda()
        input = sample['image'].cuda()
        target = target.cuda()
        target_ignore_19 = target.clone()
        target_ignore_19[target_ignore_19 > 19] = 19

        # process the input with deform grid
        input_dict = {'net_input': input, 'crop_gt': target_ignore_19, 'bifilter_crop_image': original_image, 'inference': False}
        x, condition, pred_points, aux, deform_loss = model.forward(**input_dict)
        output = x.max(1)[1]

        # compute target
        target[target == 255] = 19
        ind2onehot = lambda gt: torch.zeros(gt.shape[0], 20, gt.shape[2], gt.shape[3]).to(gt.device).scatter_(dim=1, index=gt.long(), value=1.0)
        one_hot_gt = ind2onehot(target)
        grid_one_hot_gt, grid_pixel_size = model.module.grid_pooling(one_hot_gt, condition, return_size=True)
        grid_one_hot_gt = torch.cat((grid_one_hot_gt, grid_pixel_size), dim=1)
        grid_one_hot_gt[:, -1, :, :] = ((grid_one_hot_gt[:, -1, :, :] == 1) | (grid_pixel_size[:, -1, :, :] == 0)).float() * 2
        grid_target = torch.argmax(grid_one_hot_gt, dim=1)
        grid_target[grid_target >= 19] = 255
        target[target >= 19] = 255

        # forward
        main_loss = torch.mean(criterion(x, grid_target))
        aux_loss = torch.mean(criterion(aux, grid_target))
        deform_loss = deform_loss.mean()
        loss = main_loss + args.aux_weight * aux_loss + deform_loss
        print(main_loss.item(), deform_loss.item())

        optimizer.zero_grad()
        loss.backward()
        for dgmodule in model.module.modules_dg:
            torch.nn.utils.clip_grad_norm(dgmodule.parameters(), 5)
        optimizer.step()

        n = input.size(0)
        intersection, union, target = intersectionAndUnionGPU(output, grid_target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr * args.lr_mult     # defgrid module lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
        writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    full_image_intersection_meter = AverageMeter()
    full_image_union_meter = AverageMeter()
    full_image_target_meter = AverageMeter()

    # F score buffer
    Fpc = dict()
    Fc = dict()
    for thresh in args.eval_thresholds:
        Fpc[thresh] = np.zeros(19)
        Fc[thresh] = np.zeros(19)

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (sample, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            original_image = sample['original'].cuda(non_blocking=True)
            input = sample['image'].cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True).unsqueeze(1)
            full_image_target = target.clone()

            # forward
            input_dict = {'net_input': input, 'crop_gt': target, 'bifilter_crop_image': original_image, 'inference': True, 'semantic_edge': None}
            x, condition, pred_points = model.forward(**input_dict)

            # compute grid target
            target[target == 255] = 19
            ind2onehot = lambda gt: torch.zeros(gt.shape[0], 20, gt.shape[2], gt.shape[3]).to(gt.device).scatter_(dim=1, index=gt.long(), value=1.0)
            one_hot_gt = ind2onehot(target)
            grid_one_hot_gt, grid_pixel_size = model.module.grid_pooling(one_hot_gt, condition, return_size=True)
            grid_one_hot_gt = torch.cat((grid_one_hot_gt, grid_pixel_size), dim=1)
            grid_one_hot_gt[:, -1, :, :] = ((grid_one_hot_gt[:, -1, :, :] == 1) | (grid_pixel_size[:, -1, :, :] == 0)).float() * 2
            grid_target = torch.argmax(grid_one_hot_gt, dim=1)
            grid_target[grid_target >= 19] = 255
            target[target >= 19] = 255

            # metrics
            loss = criterion(x, grid_target)
            loss = torch.mean(loss)

            output = x.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, grid_target, args.classes, args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))

            # full image eval
            full_image_prob = model.module.grid2image(x, condition, H=args.train_h, W=args.train_w)
            if args.flip:
                flip_input_dict = {'net_input': input.flip(-1), 'crop_gt': full_image_target.flip(-1), 'bifilter_crop_image': original_image.flip(-1),
                                   'inference': True, 'semantic_edge': None}
                flip_prob, flip_condition, pred_points = model.forward(**flip_input_dict)
                flip_full_image_prob = model.module.grid2image(flip_prob, flip_condition, H=args.train_h, W=args.train_w)
                full_image_prob = (full_image_prob + flip_full_image_prob.flip(-1)) / 2
            print(full_image_prob.shape)

            full_image_pred = full_image_prob.max(1)[1]
            print(full_image_pred.shape, full_image_target.shape)
            intersection, union, target = intersectionAndUnionGPU(full_image_pred, full_image_target.squeeze(1), args.classes, args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            full_image_intersection_meter.update(intersection), full_image_union_meter.update(union), full_image_target_meter.update(target)

            # F score eval
            for thresh in args.eval_thresholds:
                _Fpc, _Fc = eval_mask_boundary(full_image_pred.cpu().numpy(), full_image_target.squeeze(1).cpu().numpy(), 19, bound_th=thresh)
                Fc[thresh] += _Fc
                Fpc[thresh] += _Fpc

            # vis
            if ((i + 1) % args.print_freq == 0):
                grid_image = model.module.grid_pooling(original_image, condition)
                grid_image_paste_back = model.module.grid2image(grid_image, condition, H=args.train_h, W=args.train_w)
                grid_gt_paste_back = model.module.grid2image(grid_one_hot_gt, condition, H=args.train_h, W=args.train_w)
                grid_gt_paste_back = torch.argmax(grid_gt_paste_back, dim=1).unsqueeze(1)
                grid_gt_paste_back = grid_gt_paste_back.float() / grid_gt_paste_back.max()
                grid_pred_paste_back = full_image_pred.unsqueeze(1)
                # writer.add_image('pooling/grid_image', grid_image[0].cpu(), i)
                writer.add_image('pooling/original_image', original_image[0].cpu(), i)
                writer.add_image('pooling/original_gt', full_image_target[0].cpu(), i)
                writer.add_image('pooling/grid_image_paste_back', grid_image_paste_back[0].cpu(), i)
                writer.add_image('pooling/grid_gt_paste_back', grid_gt_paste_back[0].cpu(), i)
                writer.add_image('pooling/grid_pred_paste_back', grid_pred_paste_back[0].cpu(), i)

                plot_pred_points = pred_points[0].detach().cpu().numpy()
                plot_pred_points[:, 0] = plot_pred_points[:, 0] * args.train_w
                plot_pred_points[:, 1] = plot_pred_points[:, 1] * args.train_h
                module = model.module if type(model) == nn.DataParallel else model
                base_point_adjacent = module.matrix.init_point_adjacent
                fig = plot_deformed_lattice_on_image(plot_pred_points,
                                                     original_image[0].permute(1, 2, 0).detach().cpu().numpy(),
                                                     base_point_adjacent[0].detach().cpu(),
                                                     mask=None, return_fig=True)
                writer.add_figure('pooling/grid', fig, i)

            # print
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % args.print_freq == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    full_image_iou_class = full_image_intersection_meter.sum / (full_image_union_meter.sum + 1e-10)
    full_image_accuracy_class = full_image_intersection_meter.sum / (full_image_target_meter.sum + 1e-10)
    full_image_mIoU = np.mean(full_image_iou_class)
    full_image_mAcc = np.mean(full_image_accuracy_class)
    full_image_allAcc = sum(full_image_intersection_meter.sum) / (sum(full_image_target_meter.sum) + 1e-10)

    # F score calculation
    mean_F = dict()
    mean_Fc = dict()
    for thresh in args.eval_thresholds:
        logger.info('Threshold: ' + str(thresh))
        mean_F[thresh] = np.sum(Fpc[thresh]/Fc[thresh])/19
        mean_Fc[thresh] = Fpc[thresh]/Fc[thresh]
        logger.info('F_Score: ' + str(mean_F[thresh]))
        logger.info('F_Score (Classwise): ' + str(mean_Fc[thresh]))

    logger.info('Val result: full_image_mIoU/full_image_mAcc/full_image_allAcc {:.4f}/{:.4f}/{:.4f}.'.format(full_image_mIoU, full_image_mAcc, full_image_allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, full_image_iou_class[i], full_image_accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, full_image_mIoU, full_image_mAcc, full_image_allAcc, mean_F, mean_Fc


if __name__ == '__main__':
    main()
