import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
from collections import defaultdict
import sys
from tqdm import tqdm
code_path = os.getcwd()
sys.path.append(code_path)
from datetime import datetime
from Utils.parser import get_args
from Utils.plot_sample import plot_deformed_lattice_on_image
from Utils.matrix_utils import MatrixUtils
from dataloaders import cityscapes_processed
from dataloaders import custom_transforms as tr
from Models.deformable_grid import DeformableGrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False

setup_seed(1)
torch.backends.cudnn.benchmark = True

root_path = '/h/jungao/lab-code/deformable-grid-internal'
class Trainer(object):
    def __init__(self, args):


        # epoch and step
        self.max_epochs = args.max_epochs
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.sample_freq = args.sample_freq
        self.print_freq = args.print_freq

        # size
        self.resolution = args.resolution
        self.state_dim = args.state_dim
        self.grid_size = args.grid_size
        self.feature_channel_num = args.feature_channel_num

        # training type
        self.grid_type = args.grid_type
        self.add_mask_variance = args.add_mask_variance

        self.dataset = args.dataset
        self.encoder_backbone = args.encoder_backbone
        self.deform_layer_num = args.deform_layer_num
        self.mlp_expansion = args.mlp_expansion

        # learning parameters
        self.lr_decay = args.lr_decay
        self.lr_decay_step = args.lr_decay_step
        self.grad_clip = args.grad_clip
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.mask_coef = args.mask_coef
        self.w_variance = args.w_variance
        self.w_area = args.w_area
        self.w_laplacian = args.w_laplacian
        self.w_reconstruct_loss = args.w_reconstruct_loss

        self.gamma = args.gamma
        self.sigma = args.sigma


        # transform
        self.relax_crop = args.relax_crop
        self.zero_pad_crop = args.zero_pad_crop
        self.extreme_point = args.extreme_point

        # others
        self.load = args.load
        self.load_version = args.load_version
        self.load_epoch = args.load_epoch
        self.debug = args.debug
        self.debug_num = args.debug_num
        self.num_workers = args.num_workers
        self.version = args.version
        self.debug_index = args.debug_index
        self.shuffle = args.shuffle
        self.val_debug = args.val_debug

        # addition
        self.input_channel = 3
        self.epoch = 0

        if self.debug:
            self.val_batch_size = self.batch_size
        # paths
        self.log_path = os.path.join(root_path, 'logs', self.version)
        self.loss_path = os.path.join(root_path, 'loss', self.version)
        self.model_path = os.path.join(root_path, self.version)

        if self.debug:
            self.sample_path = os.path.join(root_path, 'samples', self.version, 'debug')
        else:
            self.sample_path = os.path.join(root_path, 'samples', self.version, 'train')

        self.check_point_path = os.path.join('./checkpoint/jungao', self.version)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.loss_path):
            os.makedirs(self.loss_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.check_point_path):
            os.makedirs(self.check_point_path)

        # saving the hyper-parameters
        f = open(os.path.join(self.log_path, 'hp.txt'), 'w')
        f.write(str(args))
        f.close()
        torch.save(args, os.path.join(self.log_path, 'args.pth'))

        self.global_step = 0
        if self.debug:
            self.train_writer = SummaryWriter(os.path.join(self.loss_path, 'debug'))
        else:
            self.train_writer = SummaryWriter(os.path.join(self.loss_path, 'train'))
            self.val_writer = SummaryWriter(os.path.join(self.loss_path, 'val'))


        self.device_count = torch.cuda.device_count()

        # grids facility
        self.matrix = MatrixUtils(1, self.grid_size,  self.grid_type, device)
        self.val_matrix = MatrixUtils(1, self.grid_size, self.grid_type, device)

        self.build_data()

        # models
        self.model = DeformableGrid(args, device)

        # GPU settings
        if self.device_count > 1:
            self.device_ids = [i for i in range(self.device_count)]
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        else:
            self.device_ids = [0]

        params_dict = []
        no_wd = []
        wd = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                # No optimization for frozen params
                continue
            print(name)
            if 'bn' in name or 'bias' in name:
                no_wd.append(p)
            else:
                wd.append(p)

        params_dict.append({'params': no_wd, 'weight_decay': 0.0})
        params_dict.append({'params': wd})

        # Allow individual options
        self.optimizer = optim.Adam(
            params_dict,
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=False)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_decay_step,
                                                        gamma=self.gamma)
        self.init_step = 0

    def build_data(self):
        self.normalize_dict = {'crop_image': 255.0, 'bifilter_crop_image': 255.0}
        self.composed_transforms_tr = transforms.Compose([
            tr.CropFromMaskStretchMulticomp(crop_elems=('image', 'gt', 'gt_polygon'), relax=self.relax_crop,
                                             zero_pad=self.zero_pad_crop),
            tr.FixedResizeStretchMulticomp(resolutions={'crop_image': self.resolution,
                                                        'crop_gt': self.resolution}),
            tr.BilateralFilteringMultiComp(['crop_image']),
            tr.ZeroOneNormalizeMultiComp(normalize_dict=self.normalize_dict),
            tr.MakeGT(size=self.resolution[0]),
            tr.ToTensorStretchMulticomp()])

        self.composed_transforms_ts = transforms.Compose([
            tr.CropFromMaskStretchMulticomp(crop_elems=('image', 'gt', 'gt_polygon'), relax=self.relax_crop,
                                            zero_pad=self.zero_pad_crop),
            tr.FixedResizeStretchMulticomp(resolutions={'crop_image': self.resolution,
                                                        'crop_gt': self.resolution}),
            tr.BilateralFilteringMultiComp(['crop_image']),
            tr.ZeroOneNormalizeMultiComp(normalize_dict=self.normalize_dict),
            tr.MakeGT(size=self.resolution[0]),
            tr.ToTensorStretchMulticomp()])

        self.trainset = cityscapes_processed.CityScapesProcessedStretchMulticomp(train=True, split='train',
                                                                                 transform=self.composed_transforms_tr,
                                                                                 inference=False)

        self.valset = cityscapes_processed.CityScapesProcessedStretchMulticomp(train=False,
                                                                               split='train_val',
                                                                               transform=self.composed_transforms_ts,
                                                                               retname=True,
                                                                               inference=True)

        self.trainset_num = len(self.trainset)
        self.valset_num = len(self.valset)

        self.train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=self.shuffle,
                                       num_workers=self.num_workers, drop_last=True,
                                       collate_fn=cityscapes_processed.multi_collate_fn)

        self.val_loader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers, drop_last=True,
                                     collate_fn=cityscapes_processed.multi_collate_fn)

        self.train_loader_num = len(self.train_loader)
        self.val_loader_num = len(self.val_loader)

        self.sample_step = self.train_loader_num // self.sample_freq
        self.print_step = self.train_loader_num // self.print_freq

        self.min_loss = 100000

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def weight_clip(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

    def loop(self):
        for epoch in range(self.epoch, self.epoch + self.max_epochs):
            if self.debug:
                self.debug_train(epoch)
            else:
                print('LR Before is: ', self.optimizer.param_groups[0]['lr'])
                if epoch == 7:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr * 0.1
                else:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr
                print('Self LR: ', self.lr)
                print('LR is now: ', self.optimizer.param_groups[0]['lr'])
                self.train()
                self.normal_train(epoch)
            self.save_checkpoint(self.max_epochs + self.epoch)

    def save_checkpoint(self, epoch, step=0, check_point=False):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        if check_point:
            save_name = os.path.join(self.check_point_path, 'epoch_latest.pth')
        else:
            save_name = os.path.join(self.model_path, 'epoch_%d_iter_%d.pth' \
                                     % (epoch, step))

        torch.save(save_state, save_name)


    def normal_train(self, epoch):
        print('Starting training')
        torch.cuda.empty_cache()
        eval_step = 2000

        for step, data in tqdm(enumerate(self.train_loader)):
            if step % eval_step == 0:
                torch.cuda.empty_cache()
                self.eval()
                with torch.no_grad():
                    self.validate()
                torch.cuda.empty_cache()
                self.train()

            self.train_batch(data, step, epoch)
            self.global_step += 1

    def debug_train(self, epoch):

        it = iter(self.train_loader)
        debug_batch = next(it)
        for step in range(100):
            self.train_batch(debug_batch, step, epoch)
            self.global_step += 1
        exit()


    def train_batch(self, data, step, epoch=0):
        crop_gt = data['crop_gt'].to(device)
        crop_gt = crop_gt.unsqueeze(1)
        crop_image = data['crop_image'].to(device)
        net_input = crop_image.clone()

        n_batch = net_input.shape[0]

        input_dict = {'net_input': net_input, 'crop_gt': crop_gt,
                      }

        base_point = self.matrix.init_point
        base_normalized_point_adjacent = self.matrix.init_normalized_point_adjacent
        base_point_mask = self.matrix.init_point_mask
        base_triangle2point = self.matrix.init_triangle2point
        base_area_mask = self.matrix.init_area_mask
        base_triangle_mask = self.matrix.init_triangle_mask


        input_dict['base_point'] = base_point.expand(n_batch, -1, -1)
        input_dict['base_normalized_point_adjacent'] = base_normalized_point_adjacent.expand(n_batch, -1, -1)
        input_dict['base_point_mask'] = base_point_mask.expand(n_batch, -1, -1)
        input_dict['base_triangle2point'] = base_triangle2point.expand(n_batch, -1, -1)
        input_dict['base_area_mask'] = base_area_mask.expand(n_batch, -1)
        input_dict['base_triangle_mask'] = base_triangle_mask.expand(n_batch, -1)
        input_dict['grid_size'] = np.max(self.grid_size)

        condition, laplacian_loss, variance, area_variance, reconstruct_loss, pred_points = self.model(
            **input_dict)

        deform_loss = variance * self.w_variance + \
                      area_variance * self.w_area + \
                      laplacian_loss * self.w_laplacian + \
                      reconstruct_loss * self.w_reconstruct_loss

        total_loss = deform_loss

        total_loss = total_loss.mean()

        total_loss.backward()
        if self.grad_clip > 0:
            self.weight_clip()

        self.optimizer.step()


        if self.global_step % 50 == 0:
            i_image = 0
            base_point_adjacent = self.matrix.init_point_adjacent

            plot_pred_points = pred_points[i_image].detach().cpu().numpy()
            plot_pred_points[:, 0] = plot_pred_points[:, 0] * self.resolution[1]
            plot_pred_points[:, 1] = plot_pred_points[:, 1] * self.resolution[0]
            fig = plot_deformed_lattice_on_image(plot_pred_points,
                                                 crop_image[i_image].permute(1, 2, 0).detach().cpu().numpy(), \
                                                 base_point_adjacent[0].detach().cpu().numpy(),
                                                 mask=None, return_fig=True,
                                                 save_path=os.path.join(self.sample_path, '%d.pdf'%(self.global_step)))
            self.train_writer.add_figure('output/deformed_img', fig, self.global_step)

            self.train_writer.add_scalar('loss', total_loss.mean().item(), self.global_step)
            self.train_writer.add_scalar('variance', variance.mean().item(), self.global_step)
            self.train_writer.add_scalar('area', area_variance.mean().item(), self.global_step)

            self.train_writer.add_scalar('laplacian', laplacian_loss.mean().item(), self.global_step)
            self.train_writer.add_scalar('reconstruct_loss', reconstruct_loss.mean().item(), self.global_step)

            print_string = "[%s] Epoch: %d, Step: %d, All Loss: %.5f, Variance: %.5f, Area: %.5f,  Laplacian: %.5f, Reconstrct: %.5f" % (str(datetime.now()), epoch, self.global_step, \
                    total_loss.mean().item(), variance.mean().item(), area_variance.mean().item(),
                    laplacian_loss.mean().item(), \
                    reconstruct_loss.mean().item())
            print(print_string)

        self.global_step += 1


    def validate(self):
        total_loss_dict = defaultdict(float)
        for data in tqdm(self.val_loader):
            crop_gt = data['crop_gt'].to(device)
            crop_gt = crop_gt.unsqueeze(1)
            crop_image = data['crop_image'].to(device)
            net_input = crop_image.clone()

            n_batch = net_input.shape[0]

            input_dict = {'net_input': net_input, 'crop_gt': crop_gt,
                          }

            base_point = self.matrix.init_point
            base_normalized_point_adjacent = self.matrix.init_normalized_point_adjacent
            base_point_mask = self.matrix.init_point_mask
            base_triangle2point = self.matrix.init_triangle2point
            base_area_mask = self.matrix.init_area_mask
            base_triangle_mask = self.matrix.init_triangle_mask


            input_dict['base_point'] = base_point.expand(n_batch, -1, -1)
            input_dict['base_normalized_point_adjacent'] = base_normalized_point_adjacent.expand(n_batch, -1, -1)
            input_dict['base_point_mask'] = base_point_mask.expand(n_batch, -1, -1)
            input_dict['base_triangle2point'] = base_triangle2point.expand(n_batch, -1, -1)
            input_dict['base_area_mask'] = base_area_mask.expand(n_batch, -1)
            input_dict['base_triangle_mask'] = base_triangle_mask.expand(n_batch, -1)
            input_dict['grid_size'] = np.max(self.grid_size)

            condition, laplacian_loss, variance, area_variance, reconstruct_loss, pred_points = self.model(
                **input_dict)

            deform_loss = variance * self.w_variance + \
                          area_variance * self.w_area + \
                          laplacian_loss * self.w_laplacian + \
                          reconstruct_loss * self.w_reconstruct_loss

            total_loss = deform_loss

            total_loss = total_loss.mean()
            total_loss_dict['total'] += total_loss.item()

            total_loss_dict['variance'] += variance.mean().item()
            total_loss_dict['area_variance'] += area_variance.mean().item()
            total_loss_dict['laplacian_loss'] += laplacian_loss.mean().item()
            total_loss_dict['reconstruct_loss'] += reconstruct_loss.mean().item()
            total_loss_dict['cnt'] += 1

        for v in total_loss_dict.keys():
            self.val_writer.add_scalar('val_'+v, total_loss_dict[v] / total_loss_dict['cnt'], self.global_step)


if __name__ == '__main__':
    args = get_args()
    print(args)
    trainer = Trainer(args)
    trainer.loop()
