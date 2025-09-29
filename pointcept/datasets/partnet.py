
"""
ShapeNet Part Dataset (Unmaintained)

get processed shapenet part dataset
at "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""



import os
import h5py
import json
from pathlib import Path

import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class PartNetDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/partnet/",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        super().__init__()
        self.data_root = Path(data_root) / "ins_seg_h5"
        self.split = split
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.cache = {}

        # load categories file
        self.categories = []
        self.category2part = {
            "Airplane": [0, 1, 2, 3],
            "Bag": [4, 5],
            "Cap": [6, 7],
            "Car": [8, 9, 10, 11],
            "Chair": [12, 13, 14, 15],
            "Earphone": [16, 17, 18],
            "Guitar": [19, 20, 21],
            "Knife": [22, 23],
            "Lamp": [24, 25, 26, 27],
            "Laptop": [28, 29],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Mug": [36, 37],
            "Pistol": [38, 39, 40],
            "Rocket": [41, 42, 43],
            "Skateboard": [44, 45, 46],
            "Table": [47, 48, 49],
        }

        self.token2category = {
            'Airplane': '02691156',
            'Bag': '02773838',
            'Cap': '02954340',
            'Car': '02958343',
            'Chair': '03001627',
            'Earphone': '03261776',
            'Guitar': '03467517',
            'Knife': '03624134',
            'Lamp': '03636649',
            'Laptop': '03642806',
            'Motorbike': '03790512',
            'Mug': '03797390',
            'Pistol': '03948459',
            'Rocket': '04099429',
            'Skateboard': '04225987',
            'Table': '04379243',
        }
        self.categories = list(self.token2category.keys())
    
        # self.token2category = {}
        # withopen(os.path.join(self.data_root,"synsetoffset2category.txt"), "r") as f:
        #     for line in f:
        #         ls = line.strip().split()
        #         self.token2category[ls[1]] = len(self.categories)
        #         self.categories.append(ls[0])

        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # load data list
        if isinstance(self.split, str):
            self.data_list = self.load_data_list(self.split)
        elif isinstance(self.split, list):
            self.data_list = []
            for s in self.split:
                self.data_list += self.load_data_list(s)
        else:
            raise NotImplementedError

        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    # def read_sample(self,data_idx):
    #     self.data
    #     pass

    def read_sample(self, idx):

        # -- init --
        data_elem = self.data_list[idx % len(self.data_list)]
        # print(data_elem)
        # print(data_elem.split("+"))
        # exit()
        data_path,data_index = data_elem.split("+")
        data_index = int(data_index)
        # name = self.get_data_name(idx)
        # split = self.get_split_name(idx)

        # -- data --
        data_dict = {}
        # data_dict["name"] = name
        # data_dict["split"] = split

        # -- read h5file --
        fields = {"pts":"coord","rgb":"color","nor":"norm","label":"segment"}
        with h5py.File(data_path, 'r') as f:
            # print(list(f.keys()))
            for in_key,out_key in fields.items():
                if out_key in ["segment"]:
                    data_dict[out_key] = f[in_key][data_index,...].astype(np.int32)
                    # data_dict[out_key] = np.array(f[in_key][data_index],dtype=np.int32)
                else:
                    data_dict[out_key] = f[in_key][data_index,...].astype(np.float32)
                    # data_dict[out_key] = np.array(f[in_key][data_index],dtype=np.float32)

        # -- fill --
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        # print(data_dict['coord'].shape)
        # exit()

        return data_dict


    def load_data_list(self,split):

        # -- get data root --
        # droot = Path(self.data_root) / "ins_seg_h5"
        droot = self.data_root

        # -- get names --
        names = []
        for name in droot.iterdir():
            if name.is_dir():
                names.append(name.stem)
        names = sorted(names)
        self.names = names

        # -- read nums --
        data_list = []
        MAX_NAME_NUM = 10 # way too big
        for name in names:
            for name_num in range(MAX_NAME_NUM):
                h5file = droot / name / ("%s-%02d.h5" % (split,name_num))
                if not(h5file.exists()): break
                size = self.h5file_size(h5file)
                data_list.extend([str(h5file)+"+%d"%ix for ix in range(size)])
        return data_list

    def h5file_size(self,fname):
        with h5py.File(fname, 'r') as f:
            size = len(f['rgb'])
        return size

    def prepare_train_data(self, idx):
        # load data
        data_idx = idx % len(self.data_list)
        if data_idx in self.cache:
            # coord, norm, segment, cls_token = self.cache[data_idx]
            coord, color, norm, segment = self.cache[data_idx]
        else:
            # coord, color, norm, segment = self.read_sample(data_idx)
            data_dict = self.read_sample(data_idx)
            # cat = self.get_sample_cat(data_idx)
            # cls_token = self.token2category[sample_cat]

            # data = np.loadtxt(self.data_list[data_idx]).astype(np.float32)
            # cls_token = self.token2category[
            #     os.path.basename(os.path.dirname(self.data_list[data_idx]))
            # ]
            # coord, norm, segment = (
            #     data[:, :3],
            #     data[:, 3:6],
            #     data[:, 6].astype(np.int32),
            # )
            # self.cache[data_idx] = (coord, norm, segment, cls_token)
            # self.cache[data_idx] = (coord, color, norm, segment, cls_token)

            # -- pack into cache --
            coord   = data_dict['coord']
            color   = data_dict['color']
            norm    = data_dict['norm']
            segment = data_dict['segment']
            self.cache[data_idx] = (coord, color, norm, segment,)

        # data_dict = dict(coord=coord,norm=norm,segment=segment,cls_token=cls_token)
        data_dict = dict(coord=coord, color=color, norm=norm, segment=segment)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        # data_idx = self.data_idx[idx % len(self.data_idx)]
        # data = np.loadtxt(self.data_list[data_idx]).astype(np.float32)
        # cls_token = self.token2category[
        #     os.path.basename(os.path.dirname(self.data_list[data_idx]))
        # ]
        # coord, norm, segment = data[:, :3], data[:, 3:6], data[:, 6].astype(np.int32)
        # data_dict = dict(coord=coord, norm=norm, cls_token=cls_token)

        data_dict = self.read_sample(idx)
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(self.post_transform(aug(deepcopy(data_dict))))
        data_dict = dict(
            fragment_list=data_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict


    def get_data_name(self, idx):
        fname,index = self.data_list[idx % len(self.data_list)].split("+")
        fname = os.path.basename(fname)
        # "bag-100"
        print(fname,index)
        exit()
        return fname

    def get_split_name(self, idx):
        fname = self.data_list[idx % len(self.data_list)].split("+")[0]
        fname = Path(fname).stem
        print(fname)
        exit()
        return os.path.basename(self.data_list[idx % len(self.data_list)])
        return self.split


    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


# """
# S3DIS Dataset

# Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
# Please cite our work if the code is helpful to you.
# """

# import os
# import h5py
# from pathlib import Path

# from .defaults import DefaultDataset
# from .builder import DATASETS


# @DATASETS.register_module()
# class PartNetDataset(DefaultDataset):

#     def get_data_list(self):

#         # -- get data root --
#         droot = Path(self.data_root) / "ins_seg_h5"

#         # -- get names --
#         names = []
#         for name in droot.iterdir():
#             if name.isdir():
#                 names.append(names.stem)
#         names = sorted(names)
#         self.names = names

#         # -- read nums --
#         data_list = []
#         MAX_NAME_NUM = 10 # way too big
#         for name in names:
#             for name_num in range(MAX_NAME_NUM):
#                 h5file = droot / name / "%s-%d.h5" % (split,name_num)
#                 if not(h5file.exists()): break
#                 size = self.h5file_size(h5file)
#                 data_list.extend([str(h5file)+"_%d"%ix for ix in range(size)])
#         return data_list

#     def get_data(self, idx):

#         # -- init --
#         data_elem = self.data_list[idx % len(self.data_list)]
#         data_path,data_index = data_elem.split("_")
#         name = self.get_data_name(idx)
#         split = self.get_split_name(idx)

#         # -- data --
#         data_dict = {}
#         data_dict["name"] = name
#         data_dict["split"] = split

#         # -- read h5file --
#         fields = {"coord":"coord","color":"color","normal":"normal","labels":"segment"}
#         with h5py.File('sem_seg_h5/data_file.h5', 'r') as f:
#             for in_key,out_key in fields.items():
#                 if out_key in ["segment"]:
#                     data_dict[out_key] = f[in_key][:].astype(np.int32)
#                 else:
#                     data_dict[out_key] = f[in_key][:].astype(np.float32)

#         # -- fill --
#         if "instance" in data_dict.keys():
#             data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
#         else:
#             data_dict["instance"] = (
#                 np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
#             )

#         return data_dict


#     def get_data_name(self, idx):
#         fname,index = self.data_list[idx % len(self.data_list)].split("_")
#         fname = os.path.basename(fname)
#         # "bag-100"
#         print(fname,index)
#         exit()
#         return fname

#     def get_split_name(self, idx):
#         fname = self.data_list[idx % len(self.data_list)].split("_")[0]
#         fname = Path(fname).stem
#         print(fname)
#         exit()
#         return os.path.basename(self.data_list[idx % len(self.data_list)])
#         return self.split

#     @static
#     def h5file_size(self,fname):
#         with h5py.File('sem_seg_h5/data_file.h5', 'r') as f:
#             size = len(f['rgb'])
#         return size
