"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
import torch as th
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {
            key: (
                collate_fn([d[key] for d in batch])
                if "offset" not in key
                # offset -> bincount -> concat bincount-> concat offset
                else torch.cumsum(
                    collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                    dim=0,
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        if "offset" in batch.keys():
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch


def collate_stack(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """

    
    # -- check --
    # B = len(batch)
    # for b in range(B):
    #     print(type(batch[b]['coord']))
    # exit()
    
    # -- get batch ids --
    B = len(batch)
    offset = th.tensor([batch[i]['coord'].shape[0] for i in range(B)])
    bids = th.arange(len(offset)).repeat_interleave(offset)
    # bids = np.repeat(np.arange(len(offset)), offset)

    # -- unpack, stack, cat --
    keys = ["color","coord","normal","segment","instance"]
    input_dict = {"bids":bids}
    for key in keys:

        # -- stack --
        for i in range(B):
            if not(key in input_dict):
                input_dict[key] = []
            input_dict[key].append(batch[i][key])

        # -- cat --
        # print(key,input_dict[key][0])
        input_dict[key] = th.cat(input_dict[key],dim=0)
        # if th.is_tensor(input_dict[key][0]):
        #     input_dict[key] = th.cat(input_dict[key],dim=0)
        # else:
        #     input_dict[key] = th.tensor(input_dict[key])

    # -- check --
    # for key in input_dict.keys():
    #     print(key,input_dict[key].shape)
    input_dict['offset'] = offset

    return input_dict



def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
