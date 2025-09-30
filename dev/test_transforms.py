"""

   The exhausting but necessary step of testing all the GPU-based transformations
   to make sure they match the corresponding CPU-based transformationy


   Please let AI be able to do this (reliably) soon....



    NormalizeColor
    NormalizeCoord
    PositiveShift
    CenterShift
    RandomShift
    PointClip
    RandomDropout
    RandomRotate
    RandomRotateTargetAngle
    RandomScale
    RandomFlip
    RandomJitter
    ClipGaussianJitter


    ChromaticAutoContrast
    ChromaticTranslation
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

    # -- get rand --
    rand_cpu = np.random.rand(1, 3)
    rand_gpu = th.from_numpy(rand_cpu).cuda()

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu,_test_rand=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu,_test_rand=rand_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu,data_gpu,['color'])


def main():


    train_loader = get_data_loader()

    nchecks = 100
    for _ in range(nchecks):
        data_cpu,data_gpu = get_sample(train_loader)
        # test_ChromaticAutoContrast(data_cpu,data_gpu)
        test_ChromaticTranslation(data_cpu,data_gpu)


if __name__ == "__main__":
    main()
