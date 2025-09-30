"""

   The exhausting but necessary step of testing all the GPU-based transformations
   to make sure they match the corresponding CPU-based transformationy


   Please let AI be able to do this (reliably) soon....




    ~NormalizeColor
    ~NormalizeCoord
    ~PositiveShift
    ~CenterShift
    ~RandomShift
    ~PointClip

    RandomDropout
    RandomRotate
    RandomRotateTargetAngle
    RandomScale
    RandomFlip
    RandomJitter
    ClipGaussianJitter


    ~ChromaticAutoContrast
    ~ChromaticTranslation
    ChromaticJitter
    RandomColorGrayScale

    RandomColorJitter
    HueSaturationTranslation
    RandomColorDrop

    ElasticDistortion

    GridSample

    SphereCrop
    ShufflePoint
    CropBoundary

    ContrastiveViewsGenerator
    MultiViewGenerator
    InstanceParser

"""



import numpy as np
import torch as th
import random


# ---------------- Helper Functions ----------------


def set_random_seed(seed):
    random.seed(seed)
    th.random.manual_seed(seed)
    np.random.seed(seed)

def check_pair(d0,d1,keys=None):
    if keys is None:
        keys = list(d0.keys())

    for key in keys:
        delta = np.mean(1.0*(d0[key]-d1[key])**2)
        # print(delta)
        passes = delta < 1e-8
        if not(passes):
            print(key,delta)
        assert(passes)

def dict_to_numpy(gpu_dict):
    cpu_dict = {}
    for k in gpu_dict:
        cpu_dict[k] = gpu_dict[k].cpu().numpy()
    return cpu_dict

def dict_to_tensor(gpu_dict):
    cpu_dict = {}
    for k in gpu_dict:
        cpu_dict[k] = gpu_dict[k].cpu().numpy()
    return cpu_dict

def get_data_loader():

    # -- get data loader --
    from pointcept.datasets import build_dataset, collate_stack
    dataset_type = "ScanNetDataset"
    data_root = "data/scannet"
    train_cfg = dict(type=dataset_type,split="train",data_root=data_root,
                     transform=[dict(type="ToTensor")], test_mode=False)
    train_data = build_dataset(train_cfg)
    train_loader = th.utils.data.DataLoader(train_data,batch_size=1,
                                               shuffle=False,num_workers=0,
                                               sampler=None,collate_fn=collate_stack,
                                               pin_memory=False)
    return train_loader

def get_sample(train_loader):

    # -- sample --
    sample = next(iter(train_loader))

    # -- numpy --
    sample_cpu = {}
    for key in sample:
        sample_cpu[key] = sample[key].cpu().numpy()
    

    # -- torch+cuda --
    sample_gpu = {}
    for key in sample:
        sample_gpu[key] = sample[key].cuda()


    return sample_cpu,sample_gpu



# ---------------- Rotation / Geometric ----------------


def test_NormalizeCoord(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import NormalizeCoord as NormalizeCoord_cpu
    from pointcept.engines.hooks.transform_gpu import NormalizeCoord as NormalizeCoord_gpu
    
    # -- params --
    seed = int(100*random.random())
    
    # -- init --
    xform_cpu = NormalizeCoord_cpu()
    xform_gpu = NormalizeCoord_gpu()
    
    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)
    
    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_PositiveShift(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import PositiveShift as PositiveShift_cpu
    from pointcept.engines.hooks.transform_gpu import PositiveShift as PositiveShift_gpu
    
    # -- params --
    seed = int(100*random.random())
    
    # -- init --
    xform_cpu = PositiveShift_cpu()
    xform_gpu = PositiveShift_gpu()
    
    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)
    
    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_CenterShift(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import CenterShift as CenterShift_cpu
    from pointcept.engines.hooks.transform_gpu import CenterShift as CenterShift_gpu
    
    # -- params --
    seed = int(100*random.random())
    
    # -- init --
    xform_cpu = CenterShift_cpu()
    xform_gpu = CenterShift_gpu()
    
    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)
    
    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_RandomShift(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import RandomShift as RandomShift_cpu
    from pointcept.engines.hooks.transform_gpu import RandomShift as RandomShift_gpu
    
    # -- params --
    seed = int(100*random.random())
    shift = [0.5*random.random() for _ in range(3)]
    shift = [[-s,s] for s in shift]
    
    # -- init --
    xform_cpu = RandomShift_cpu(shift=shift)
    xform_gpu = RandomShift_gpu(shift=shift)
    
    # -- get rand --
    # shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
    # shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
    # shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
    # rand_cpu = np.random.uniform(-1, 1, size=(3,))
    # rand_gpu = th.from_numpy(rand_cpu).cuda()
    
    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)
    
    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_PointClip(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import PointClip as PointClip_cpu
    from pointcept.engines.hooks.transform_gpu import PointClip as PointClip_gpu
    
    # -- params --
    seed = int(100*random.random())
    point_cloud_range = [random.random() * 10 for _ in range(3)]
    point_cloud_range = [-p for p in point_cloud_range] + point_cloud_range # cat
    
    # -- init --
    xform_cpu = PointClip_cpu(point_cloud_range=point_cloud_range)
    xform_gpu = PointClip_gpu(point_cloud_range=point_cloud_range)
    
    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)
    
    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord', 'color', 'segment'])


def test_RandomDropout(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import RandomDropout as RandomDropout_cpu
    from pointcept.engines.hooks.transform_gpu import RandomDropout as RandomDropout_gpu
    
    # -- params --
    seed = int(100*random.random())
    dropout_ratio = 0.5 * random.random()
    
    # -- init --
    xform_cpu = RandomDropout_cpu(dropout_ratio=dropout_ratio)
    xform_gpu = RandomDropout_gpu(dropout_ratio=dropout_ratio)
    
    # -- get rand --
    num_points = len(data_cpu['coord'])
    rand_cpu = np.random.rand(num_points)
    rand_gpu = th.from_numpy(rand_cpu).cuda()
    
    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)
    
    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord', 'color', 'segment'])

def test_RandomRotate(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomRotate as RandomRotate_cpu
    from pointcept.engines.hooks.transform_gpu import RandomRotate as RandomRotate_gpu

    seed = int(100*random.random())
    xform_cpu = RandomRotate_cpu()
    xform_gpu = RandomRotate_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_RandomRotateTargetAngle(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomRotateTargetAngle as RandomRotateTargetAngle_cpu
    from pointcept.engines.hooks.transform_gpu import RandomRotateTargetAngle as RandomRotateTargetAngle_gpu

    seed = int(100*random.random())
    xform_cpu = RandomRotateTargetAngle_cpu()
    xform_gpu = RandomRotateTargetAngle_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_RandomScale(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomScale as RandomScale_cpu
    from pointcept.engines.hooks.transform_gpu import RandomScale as RandomScale_gpu

    seed = int(100*random.random())
    xform_cpu = RandomScale_cpu()
    xform_gpu = RandomScale_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_RandomFlip(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomFlip as RandomFlip_cpu
    from pointcept.engines.hooks.transform_gpu import RandomFlip as RandomFlip_gpu

    seed = int(100*random.random())
    xform_cpu = RandomFlip_cpu()
    xform_gpu = RandomFlip_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_RandomJitter(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomJitter as RandomJitter_cpu
    from pointcept.engines.hooks.transform_gpu import RandomJitter as RandomJitter_gpu

    seed = int(100*random.random())
    xform_cpu = RandomJitter_cpu()
    xform_gpu = RandomJitter_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_ClipGaussianJitter(data_cpu, data_gpu):
    from pointcept.datasets.transform import ClipGaussianJitter as ClipGaussianJitter_cpu
    from pointcept.engines.hooks.transform_gpu import ClipGaussianJitter as ClipGaussianJitter_gpu

    seed = int(100*random.random())
    xform_cpu = ClipGaussianJitter_cpu()
    xform_gpu = ClipGaussianJitter_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


# ---------------- Color / Chromatic ----------------


def test_NormalizeColor(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import NormalizeColor as NormalizeColor_cpu
    from pointcept.engines.hooks.transform_gpu import NormalizeColor as NormalizeColor_gpu
    
    # -- params --
    seed = int(100*random.random())
    
    # -- init --
    xform_cpu = NormalizeColor_cpu()
    xform_gpu = NormalizeColor_gpu()
    
    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)
    
    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['color'])


def test_ChromaticAutoContrast(data_cpu,data_gpu):

    # -- imports --
    from pointcept.datasets.transform import ChromaticAutoContrast as ChromaticAutoContrast_cpu
    from pointcept.engines.hooks.transform_gpu import ChromaticAutoContrast as ChromaticAutoContrast_gpu

    # -- params --
    seed = int(100*random.random())
    p = 1.0 # always
    blend_factor = np.random.rand(3).reshape((1,3))
    blend_factor_gpu = th.from_numpy(blend_factor).cuda()

    # -- init --
    xform_cpu = ChromaticAutoContrast_cpu(p,blend_factor)
    xform_gpu = ChromaticAutoContrast_gpu(p,blend_factor_gpu)

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu,data_gpu,['color'])


def test_ChromaticTranslation(data_cpu,data_gpu):

    # -- imports --
    from pointcept.datasets.transform import ChromaticTranslation as ChromaticTranslation_cpu
    from pointcept.engines.hooks.transform_gpu import ChromaticTranslation as ChromaticTranslation_gpu

    # -- params --
    seed = int(100*random.random())
    p = 1.0 # always
    ratio = 0.5*random.random()

    # -- init --
    xform_cpu = ChromaticTranslation_cpu(p,ratio)
    xform_gpu = ChromaticTranslation_gpu(p,ratio)

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu,data_gpu,['color'])


def test_ChromaticJitter(data_cpu, data_gpu):
    from pointcept.datasets.transform import ChromaticJitter as ChromaticJitter_cpu
    from pointcept.engines.hooks.transform_gpu import ChromaticJitter as ChromaticJitter_gpu

    seed = int(100*random.random())
    xform_cpu = ChromaticJitter_cpu()
    xform_gpu = ChromaticJitter_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['color'])


def test_RandomColorGrayScale(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomColorGrayScale as RandomColorGrayScale_cpu
    from pointcept.engines.hooks.transform_gpu import RandomColorGrayScale as RandomColorGrayScale_gpu

    seed = int(100*random.random())
    xform_cpu = RandomColorGrayScale_cpu()
    xform_gpu = RandomColorGrayScale_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['color'])


def test_RandomColorJitter(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomColorJitter as RandomColorJitter_cpu
    from pointcept.engines.hooks.transform_gpu import RandomColorJitter as RandomColorJitter_gpu

    seed = int(100*random.random())
    xform_cpu = RandomColorJitter_cpu()
    xform_gpu = RandomColorJitter_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['color'])


def test_HueSaturationTranslation(data_cpu, data_gpu):
    from pointcept.datasets.transform import HueSaturationTranslation as HueSaturationTranslation_cpu
    from pointcept.engines.hooks.transform_gpu import HueSaturationTranslation as HueSaturationTranslation_gpu

    seed = int(100*random.random())
    xform_cpu = HueSaturationTranslation_cpu()
    xform_gpu = HueSaturationTranslation_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['color'])


def test_RandomColorDrop(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomColorDrop as RandomColorDrop_cpu
    from pointcept.engines.hooks.transform_gpu import RandomColorDrop as RandomColorDrop_gpu

    seed = int(100*random.random())
    xform_cpu = RandomColorDrop_cpu()
    xform_gpu = RandomColorDrop_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['color'])






# ---------------- Spatial / Sampling ----------------

def test_ElasticDistortion(data_cpu, data_gpu):
    from pointcept.datasets.transform import ElasticDistortion as ElasticDistortion_cpu
    from pointcept.engines.hooks.transform_gpu import ElasticDistortion as ElasticDistortion_gpu

    seed = int(100*random.random())
    xform_cpu = ElasticDistortion_cpu()
    xform_gpu = ElasticDistortion_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_GridSample(data_cpu, data_gpu):
    from pointcept.datasets.transform import GridSample as GridSample_cpu
    from pointcept.engines.hooks.transform_gpu import GridSample as GridSample_gpu

    seed = int(100*random.random())
    xform_cpu = GridSample_cpu()
    xform_gpu = GridSample_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_SphereCrop(data_cpu, data_gpu):
    from pointcept.datasets.transform import SphereCrop as SphereCrop_cpu
    from pointcept.engines.hooks.transform_gpu import SphereCrop as SphereCrop_gpu

    seed = int(100*random.random())
    xform_cpu = SphereCrop_cpu()
    xform_gpu = SphereCrop_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_ShufflePoint(data_cpu, data_gpu):
    from pointcept.datasets.transform import ShufflePoint as ShufflePoint_cpu
    from pointcept.engines.hooks.transform_gpu import ShufflePoint as ShufflePoint_gpu

    seed = int(100*random.random())
    xform_cpu = ShufflePoint_cpu()
    xform_gpu = ShufflePoint_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_CropBoundary(data_cpu, data_gpu):
    from pointcept.datasets.transform import CropBoundary as CropBoundary_cpu
    from pointcept.engines.hooks.transform_gpu import CropBoundary as CropBoundary_gpu

    seed = int(100*random.random())
    xform_cpu = CropBoundary_cpu()
    xform_gpu = CropBoundary_gpu()

    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _test_rand=rand_gpu)

    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])



def main():


    train_loader = get_data_loader()

    nchecks = 100
    for _ in range(nchecks):
        data_cpu,data_gpu = get_sample(train_loader)

        # test_NormalizeColor(data_cpu,data_gpu)
        # test_NormalizeCoord(data_cpu,data_gpu)
        # test_PositiveShift(data_cpu,data_gpu)

        # test_CenterShift(data_cpu,data_gpu)
        # test_RandomShift(data_cpu,data_gpu)
        test_PointClip(data_cpu,data_gpu)


        # test_ChromaticAutoContrast(data_cpu,data_gpu)
        # test_ChromaticTranslation(data_cpu,data_gpu)


if __name__ == "__main__":
    main()
