import os

# cslab cluster
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':

            if os.path.exists('/h/zianwang/disk/dataset/PASCAL/VOCdevkit/VOC2012'):
                voc_root = '/h/zianwang/disk/dataset/PASCAL/VOCdevkit/VOC2012'
            elif os.path.exists('/scratch/ssd001/home/linghuan/datasets/dataset/PASCAL/VOCdevkit/VOC2012'):
                voc_root = '/scratch/ssd001/home/linghuan/datasets/dataset/PASCAL/VOCdevkit/VOC2012'
            else:
                raise ValueError('PASCAL VOC dataset path does not exist!')
            return voc_root

        elif database == 'sbd':
            return '/h/zianwang/disk/dataset/SBD/'        # folder with benchmark_RELEASE/

        elif database == 'davis2016':
            return '/h/zianwang/disk/dataset/DAVIS/'      # folder with Annotations/, ImageSets/, JPEGImages/, ...

        elif database == 'cityscapes':                    # folder with gtCoarse/, gtFine/, leftImg8bit/
            return '/h/zianwang/disk/dataset/cityscapes/'

        elif database == 'bsd500':
            return '/h/jungao/dataset/BSR/BSDS500/data'

        elif database == 'kitti-processed':
            return '/scratch/ssd001/home/jungao/cross_domain_polydata/kitti'
        elif database == 'ade-processed':
            return '/scratch/ssd001/home/jungao/cross_domain_polydata/ADE'
        elif database == 'rooftop-processed':
            return '/scratch/ssd001/home/jungao/cross_domain_polydata/rooftop'
        elif database == 'cardiac-processed':
            return '/scratch/ssd001/home/jungao/cross_domain_polydata/cardiac'
        elif database == 'sstem-processed':
            return '/scratch/ssd001/home/jungao/cross_domain_polydata/sstem'
        elif database == 'cityscapes-processed':
            #return '/h/zianwang/disk/dataset/polyrnn-pp-pytorch/data/cityscapes_processed'
            if os.path.exists('/h/zianwang/disk/dataset/polyrnn-pp-pytorch/data/cityscapes_processed'):
                return '/h/zianwang/disk/dataset/polyrnn-pp-pytorch/data/cityscapes_processed'
            elif os.path.exists('/scratch/ssd001/home/linghuan/datasets/dataset/polyrnn-pp-pytorch/data/cityscapes_processed'):
                return '/scratch/ssd001/home/linghuan/datasets/dataset/polyrnn-pp-pytorch/data/cityscapes_processed'

        elif database == 'cityscapes-processed-ade':
            if os.path.exists('/scratch/gobi2/jungao/dataset/city_and_ade_processed'):
                return '/scratch/gobi2/jungao/dataset/city_and_ade_processed'
            return '/h/linghuan/datasets/dataset/city_and_ade_process'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return 'models/'