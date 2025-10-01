"""

   The exhausting but necessary step of testing all the GPU-based transformations
   to make sure they match the corresponding CPU-based transformationy


   Please let AI be able to do this (reliably) soon....


    #
    # -- spatial --
    # 

    ~NormalizeCoord
    ~PositiveShift
    ~CenterShift
    ~RandomShift
    ~PointClip

    ~RandomDropout
    ~RandomRotate
    ~RandomRotateTargetAngle
    ~RandomScale
    ~RandomFlip
    ~RandomJitter
    ~ClipGaussianJitter

    SphereCrop
    ShufflePoint
    CropBoundary


    #
    # -- color --
    # 

    ~NormalizeColor
    ~ChromaticAutoContrast
    ~ChromaticTranslation
    ChromaticJitter
    RandomColorJitter
    RandomColorGrayScale
    HueSaturationTranslation
    RandomColorDrop

    #
    # -- sampling --
    #

    ElasticDistortion
    GridSample


    TODO:
    - ContrastiveViewsGenerator
    - MultiViewGenerator
    - InstanceParser

"""


import copy
dcopy = copy.deepcopy

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
        passes = delta < 1e-5
        if not(passes): print("mean: ",key,delta)
        assert(passes)

        delta = np.max(1.0*(d0[key]-d1[key])**2)
        passes = delta < 1e-5
        if not(passes): print("max: ",key,delta)
        assert(passes)

def dict_to_numpy(gpu_dict):
    cpu_dict = {}
    for k in gpu_dict:
        if k == "index_valid_keys": continue
        cpu_dict[k] = gpu_dict[k].cpu().numpy()
    return cpu_dict

def dict_to_tensor(gpu_dict):
    cpu_dict = {}
    for k in gpu_dict:
        cpu_dict[k] = gpu_dict[k].cpu().numpy()
    return cpu_dict

def get_data_loader(batch_size=1):

    # -- get data loader --
    from pointcept.datasets import build_dataset, collate_stack
    dataset_type = "ScanNetDataset"
    data_root = "data/scannet"
    train_cfg = dict(type=dataset_type,split="train",data_root=data_root,
                     transform=[dict(type="ToTensor")], test_mode=False)
    train_data = build_dataset(train_cfg)
    train_loader = th.utils.data.DataLoader(train_data,batch_size=batch_size,
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


def index_data(data_dict,index):
    data_dict_ = dict()
    for key in data_dict.keys():
        if key in data_dict["index_valid_keys"]:
            data_dict_[key] = data_dict[key][index]
        elif key == "index_valid_keys":
            data_dict_[key] = copy.copy(data_dict[key])
        else:
            data_dict_[key] = data_dict[key]
    return data_dict_

def write_data(tgt_data,src_data,index):
    for key in src_data.keys():
        if key in src_data["index_valid_keys"]:
            tgt_data[key][index] = src_data[key]

def append_data(tgt_data,src_data):
    for key in src_data.keys():
        if key in src_data["index_valid_keys"]:
            if not(key in tgt_data):
                tgt_data[key] = []
            tgt_data[key].append(src_data[key])

def concat_data(data):
    for key in data.keys():
        # if key in data["index_valid_keys"]:
        data[key] = th.cat(data[key])
        # print(key,data[key].shape)
            
def run_serially(xform_gpu,data_gpu):

    data_gpu["index_valid_keys"] = [
        "coord",
        "color",
        "normal",
        "superpoint",
        "strength",
        "segment",
        "instance",
        "bids"
    ]

    output = {}#dcopy(data_gpu)
    offset = data_gpu['offset']
    istart = th.cumsum(offset,0)

    B = len(offset)
    for b in range(B):

        # -- get indices for subset --
        if b > 0: start = istart[b-1]
        else: start = 0
        stop = start+offset[b]
        index = th.arange(start,stop).cuda()


        # -- get subset --
        data_gpu_b = index_data(data_gpu, index)

        # -- check --
        bids = data_gpu_b['bids']
        bids[...] = 0
        data_gpu_b['offset'] = data_gpu['offset'][[b]]
        # print("Num Batch: ",len(th.unique(bids)))

        # -- run --
        data_gpu_b = xform_gpu(data_gpu_b)
        data_gpu_b['bids'][...] = b

        # -- write --
        # write_data(output,data_gpu_b,index)
        append_data(output,data_gpu_b)

    concat_data(output)

    return output


# ---------------- Rotation / Geometric ----------------


def test_NormalizeCoord(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import NormalizeCoord as NormalizeCoord_cpu
    from pointcept.engines.hooks.transform_gpu import NormalizeCoord as NormalizeCoord_gpu
    
    # -- params --
    seed = int(10000*random.random())
    
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

def test_NormalizeCoord_batch(data_gpu):

    # -- imports --
    from pointcept.engines.hooks.transform_gpu import NormalizeCoord as NormalizeCoord_gpu
    
    # -- params --
    seed = int(10000*random.random())
    
    # -- init --
    xform_gpu = NormalizeCoord_gpu()
    data_gpu_batch = dcopy(data_gpu)
    data_gpu_serial = dcopy(data_gpu)
    
    # -- run --
    set_random_seed(seed)
    data_gpu_batch = xform_gpu(data_gpu_batch)
    data_gpu_serial = run_serially(xform_gpu,data_gpu_serial)

    # -- check --
    data_gpu_batch = dict_to_numpy(data_gpu_batch)
    data_gpu_serial = dict_to_numpy(data_gpu_serial)
    check_pair(data_gpu_batch, data_gpu_serial, ['coord'])


def test_PositiveShift(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import PositiveShift as PositiveShift_cpu
    from pointcept.engines.hooks.transform_gpu import PositiveShift as PositiveShift_gpu
    
    # -- params --
    seed = int(10000*random.random())
    
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
    seed = int(10000*random.random())
    apply_z = random.random() > 0.5

    # -- init --
    xform_cpu = CenterShift_cpu(apply_z=apply_z)
    xform_gpu = CenterShift_gpu(apply_z=apply_z)
    
    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)
    
    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_CenterShift_batch(data_gpu):

    # -- imports --
    from pointcept.engines.hooks.transform_gpu import CenterShift as CenterShift_gpu
    
    # -- params --
    seed = int(10000*random.random())
    apply_z = random.random() > 0.5
    
    # -- init --
    xform_gpu = CenterShift_gpu(apply_z=apply_z)
    data_gpu_batch = dcopy(data_gpu)
    data_gpu_serial = dcopy(data_gpu)
    
    # -- run --
    set_random_seed(seed)
    data_gpu_batch = xform_gpu(data_gpu_batch)
    data_gpu_serial = run_serially(xform_gpu,data_gpu_serial)

    # -- check --
    data_gpu_batch = dict_to_numpy(data_gpu_batch)
    data_gpu_serial = dict_to_numpy(data_gpu_serial)
    check_pair(data_gpu_batch, data_gpu_serial, ['coord'])


def test_RandomShift(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import RandomShift as RandomShift_cpu
    from pointcept.engines.hooks.transform_gpu import RandomShift as RandomShift_gpu
    
    # -- params --
    seed = int(10000*random.random())
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
    seed = int(10000*random.random())
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
    seed = int(10000*random.random())
    dropout_ratio = 0.8 * random.random()
    
    # -- init --
    xform_cpu = RandomDropout_cpu(dropout_ratio=dropout_ratio,dropout_application_ratio=1.)
    xform_gpu = RandomDropout_gpu(dropout_ratio=dropout_ratio,dropout_application_ratio=1.)
    
    # -- get rand --
    n = len(data_cpu['coord'])
    rand_cpu = np.random.choice(n, int(n * (1 - dropout_ratio)), replace=False)
    rand_gpu = th.from_numpy(rand_cpu).cuda()
    
    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _idx=rand_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _idx=rand_gpu)
    
    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord', 'color', 'segment'])


def test_RandomDropout_batch(data_gpu):

    # -- imports --
    from pointcept.engines.hooks.transform_gpu import RandomDropout as RandomDropout_gpu
    
    # -- params --
    seed = int(10000*random.random())
    dropout_ratio = 0.8 * random.random()
    
    # -- init --
    xform_gpu = RandomDropout_gpu(dropout_ratio=dropout_ratio,dropout_application_ratio=1.)
    data_gpu_batch = dcopy(data_gpu)
    data_gpu_serial = dcopy(data_gpu)
    
    # # -- get rand --
    # n = len(data_cpu['coord'])
    # rand_gpu = th.from_numpy(rand_cpu).cuda()
    
    # -- run --
    set_random_seed(seed)
    data_gpu_batch = xform_gpu(data_gpu_batch)
    th.cuda.synchronize()
    set_random_seed(seed)
    data_gpu_serial = run_serially(xform_gpu,data_gpu_serial)
    
    # -- check --
    data_gpu_batch = dict_to_numpy(data_gpu_batch)
    data_gpu_serial = dict_to_numpy(data_gpu_serial)

    # check_pair(data_gpu_batch, data_gpu_serial, ['coord', 'color', 'segment'])
    check_pair(data_gpu_batch, data_gpu_serial, ['bids'])



def test_RandomRotate(data_cpu, data_gpu):

    # -- imports --
    from pointcept.datasets.transform import RandomRotate as RandomRotate_cpu
    from pointcept.engines.hooks.transform_gpu import RandomRotate as RandomRotate_gpu

    # -- params --
    seed = int(10000*random.random())
    angle = random.random()
    angle = [-angle,angle]
    axis = np.random.choice(["x","y","z"])

    # -- init --
    center = None
    p = 1.0 # actually always run
    xform_cpu = RandomRotate_cpu(angle=angle,center=center,axis=axis,p=p)
    xform_gpu = RandomRotate_gpu(angle=angle,center=center,axis=axis,p=p)

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord','normal'])


def test_RandomRotateTargetAngle(data_cpu, data_gpu):

    # -- imports --
    from pointcept.datasets.transform import RandomRotateTargetAngle as RandomRotateTargetAngle_cpu
    from pointcept.engines.hooks.transform_gpu import RandomRotateTargetAngle as RandomRotateTargetAngle_gpu

    # -- params --
    seed = int(10000*random.random())
    angle = list(np.random.uniform(0, 2*np.pi, size=3))
    axis = np.random.choice(["x","y","z"])

    # -- init --
    xform_cpu = RandomRotateTargetAngle_cpu(angle,axis=axis,p=1.0)
    xform_gpu = RandomRotateTargetAngle_gpu(angle,axis=axis,p=1.0)

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_RandomScale(data_cpu, data_gpu):

    # -- imports --
    from pointcept.datasets.transform import RandomScale as RandomScale_cpu
    from pointcept.engines.hooks.transform_gpu import RandomScale as RandomScale_gpu

    # -- params --
    seed = int(10000*random.random())
    scale = 0.2 * random.random()
    scale = [1 - scale, 1 + scale]
    anisotropic = random.random() < 0.5

    # -- init --
    xform_cpu = RandomScale_cpu(scale,anisotropic)
    xform_gpu = RandomScale_gpu(scale,anisotropic)

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_RandomFlip(data_cpu, data_gpu):

    # -- imports --
    from pointcept.datasets.transform import RandomFlip as RandomFlip_cpu
    from pointcept.engines.hooks.transform_gpu import RandomFlip as RandomFlip_gpu

    # -- params --
    seed = int(10000*random.random())
    xform_cpu = RandomFlip_cpu()
    xform_gpu = RandomFlip_gpu()

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_RandomJitter(data_cpu, data_gpu):

    # -- imports --
    from pointcept.datasets.transform import RandomJitter as RandomJitter_cpu
    from pointcept.engines.hooks.transform_gpu import RandomJitter as RandomJitter_gpu

    # -- params --
    seed = int(10000*random.random())
    randn_cpu = np.random.randn(data_cpu["coord"].shape[0], 3)
    randn_gpu = th.from_numpy(randn_cpu).to("cuda")
    sigma = 0.05 * random.random()
    clip = 0.1 * random.random()

    # -- init --
    xform_cpu = RandomJitter_cpu(sigma,clip)
    xform_gpu = RandomJitter_gpu(sigma,clip)

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu, _randn=randn_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu, _randn=randn_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_ClipGaussianJitter(data_cpu, data_gpu):

    # -- imports --
    from pointcept.datasets.transform import ClipGaussianJitter as ClipGaussianJitter_cpu
    from pointcept.engines.hooks.transform_gpu import ClipGaussianJitter as ClipGaussianJitter_gpu

    # -- parmams --
    seed = int(10000*random.random())
    scalar = 0.3 * random.random()
    N = data_gpu["coord"].shape[0]
    randn_gpu = th.randn(N, 3, device="cuda")
    randn_cpu = randn_gpu.cpu().numpy()

    # -- init --
    xform_cpu = ClipGaussianJitter_cpu(scalar)
    xform_gpu = ClipGaussianJitter_gpu(scalar)

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu,_randn=randn_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu,_randn=randn_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])

def test_SphereCrop(data_cpu, data_gpu):

    # -- imports --
    from pointcept.datasets.transform import SphereCrop as SphereCrop_cpu
    from pointcept.engines.hooks.transform_gpu import SphereCrop as SphereCrop_gpu

    # -- params --
    seed = int(10000*random.random())
    point_max = np.random.choice([40000,60000,80000])
    mode = np.random.choice(["random", "center"])
    mode = "center"

    # -- init --
    xform_cpu = SphereCrop_cpu(point_max,mode=mode)
    xform_gpu = SphereCrop_gpu(point_max,mode=mode)

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- inspect --
    # print(data_cpu['coord'][:10])
    # print(data_gpu['coord'][:10])

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    if mode == "center":
        delta = np.mean((data_cpu['coord']-data_gpu['coord'])**2,-1)
        perror = np.mean(1.0*(delta < 1e-10))
        # print(perror) # percent error
        assert(perror > 0.99)
    check_pair(data_cpu, data_gpu, ['bids'])


def test_SphereCrop_batch(data_gpu):

    # -- imports --
    from pointcept.engines.hooks.transform_gpu import SphereCrop as SphereCrop_gpu

    # -- params --
    seed = int(10000*random.random())
    point_max = np.random.choice([40000,60000,80000])
    mode = np.random.choice(["random", "center"])
    mode = "center"
    data_gpu['coord'] += th.clamp(0.01*th.randn_like(data_gpu['coord']),-0.05,0.05)

    # -- init --
    xform_gpu = SphereCrop_gpu(point_max=point_max,mode=mode)
    data_gpu_batch = dcopy(data_gpu)
    data_gpu_serial = dcopy(data_gpu)
    
    # -- run --
    set_random_seed(seed)
    data_gpu_batch = xform_gpu(data_gpu_batch)
    th.cuda.synchronize()
    set_random_seed(seed)
    data_gpu_serial = run_serially(xform_gpu,data_gpu_serial)
    
    # -- check --
    data_gpu_batch = dict_to_numpy(data_gpu_batch)
    data_gpu_serial = dict_to_numpy(data_gpu_serial)

    # print(data_gpu_batch['coord'][-10:])
    # print(data_gpu_serial['coord'][-10:])

    if mode == "center":
        delta = np.mean((data_gpu_batch['coord']-data_gpu_serial['coord'])**2,-1)
        perror = np.mean(1.0*(delta < 1e-10))
        assert(perror > 0.98) # a larger batch size has small erros in each which accumulate; so test few
    check_pair(data_gpu_batch, data_gpu_serial, ['bids'])

def test_ShufflePoint(data_cpu, data_gpu):

    # -- imports --
    from pointcept.datasets.transform import ShufflePoint as ShufflePoint_cpu
    from pointcept.engines.hooks.transform_gpu import ShufflePoint as ShufflePoint_gpu

    # -- params --
    seed = int(10000*random.random())

    # -- init --
    xform_cpu = ShufflePoint_cpu()
    xform_gpu = ShufflePoint_gpu()

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])


def test_ShufflePoint_batch(data_gpu):

    # -- imports --
    from pointcept.engines.hooks.transform_gpu import ShufflePoint as ShufflePoint_gpu

    # -- params --
    seed = int(10000*random.random())

    # -- init --
    xform_gpu = ShufflePoint_gpu()

    # -- run --
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- check --
    bids = data_gpu['bids']
    delta = bids[1:] - bids[:-1]
    nz_inds = th.where(delta>0)[0]+1
    offset = th.cumsum(data_gpu['offset'][:-1],0)
    delta = th.all(nz_inds == offset)
    assert(delta)

def test_CropBoundary(data_cpu, data_gpu):

    # -- imports --
    from pointcept.datasets.transform import CropBoundary as CropBoundary_cpu
    from pointcept.engines.hooks.transform_gpu import CropBoundary as CropBoundary_gpu

    # -- params --
    seed = int(10000*random.random())
    

    # -- init --
    xform_cpu = CropBoundary_cpu()
    xform_gpu = CropBoundary_gpu()

    # -- run --
    set_random_seed(seed)
    data_cpu = xform_cpu(data_cpu)
    set_random_seed(seed)
    data_gpu = xform_gpu(data_gpu)

    # -- check --
    data_gpu = dict_to_numpy(data_gpu)
    check_pair(data_cpu, data_gpu, ['coord'])

# ---------------- Color / Chromatic ----------------


def test_NormalizeColor(data_cpu, data_gpu):
    # -- imports --
    from pointcept.datasets.transform import NormalizeColor as NormalizeColor_cpu
    from pointcept.engines.hooks.transform_gpu import NormalizeColor as NormalizeColor_gpu
    
    # -- params --
    seed = int(10000*random.random())
    
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
    seed = int(10000*random.random())
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
    seed = int(10000*random.random())
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

    seed = int(10000*random.random())
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


def test_RandomColorJitter(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomColorJitter as RandomColorJitter_cpu
    from pointcept.engines.hooks.transform_gpu import RandomColorJitter as RandomColorJitter_gpu

    seed = int(10000*random.random())
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


def test_RandomColorDrop(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomColorDrop as RandomColorDrop_cpu
    from pointcept.engines.hooks.transform_gpu import RandomColorDrop as RandomColorDrop_gpu

    seed = int(10000*random.random())
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


def test_HueSaturationTranslation(data_cpu, data_gpu):
    from pointcept.datasets.transform import HueSaturationTranslation as HueSaturationTranslation_cpu
    from pointcept.engines.hooks.transform_gpu import HueSaturationTranslation as HueSaturationTranslation_gpu

    seed = int(10000*random.random())
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

def test_RandomColorGrayScale(data_cpu, data_gpu):
    from pointcept.datasets.transform import RandomColorGrayScale as RandomColorGrayScale_cpu
    from pointcept.engines.hooks.transform_gpu import RandomColorGrayScale as RandomColorGrayScale_gpu

    seed = int(10000*random.random())
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




# ---------------- Spatial / Sampling ----------------

def test_ElasticDistortion(data_cpu, data_gpu):
    from pointcept.datasets.transform import ElasticDistortion as ElasticDistortion_cpu
    from pointcept.engines.hooks.transform_gpu import ElasticDistortion as ElasticDistortion_gpu

    seed = int(10000*random.random())
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

    seed = int(10000*random.random())
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




def main():


    batch_size = 3
    train_loader = get_data_loader(batch_size)

    nchecks = 100
    for idx in range(nchecks):

        # -- ... --
        set_random_seed(idx)
        data_cpu,data_gpu = get_sample(train_loader)

        # -- coords --
        # test_NormalizeCoord(data_cpu,data_gpu)
        # if batch_size > 1:
        #     test_NormalizeCoord_batch(data_gpu)

        # test_PositiveShift(data_cpu,data_gpu)
        # test_CenterShift(data_cpu,data_gpu)
        # if batch_size > 1:
        #     test_CenterShift_batch(data_gpu)

        # test_RandomShift(data_cpu,data_gpu)
        # test_PointClip(data_cpu,data_gpu)

        # test_SphereCrop(data_cpu, data_gpu)
        if batch_size > 1:
            test_SphereCrop_batch(data_gpu)


        # test_ShufflePoint(data_cpu, data_gpu)
        # if batch_size > 1:
        #     test_ShufflePoint_batch(data_gpu)

        # test_CropBoundary(data_cpu, data_gpu)

        # -- ... --
        # test_RandomDropout(data_cpu,data_gpu)
        # if batch_size > 1:
        #     test_RandomDropout_batch(data_gpu)

        # test_RandomRotate(data_cpu,data_gpu)
        # test_RandomRotateTargetAngle(data_cpu,data_gpu)
        # test_RandomScale(data_cpu,data_gpu)
        # test_RandomFlip(data_cpu,data_gpu)
        # test_RandomJitter(data_cpu,data_gpu)
        # test_ClipGaussianJitter(data_cpu,data_gpu)
    
        # -- color --
        # test_NormalizeColor(data_cpu,data_gpu)
        # test_ChromaticAutoContrast(data_cpu,data_gpu)
        # test_ChromaticTranslation(data_cpu,data_gpu)


if __name__ == "__main__":
    main()
