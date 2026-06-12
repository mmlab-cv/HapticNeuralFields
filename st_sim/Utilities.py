import torch as t



def pad_blocks(x: t.Tensor, bl: int):
    numblocks = int(t.ceil(x.size(1) / t.tensor(bl)))
    temp = t.zeros((x.size(0), numblocks * bl)).to(x.device)
    temp[:, : x.size(1)] = x
    x = temp
    x = t.reshape(x, (x.size(0), numblocks, bl))
    return x
