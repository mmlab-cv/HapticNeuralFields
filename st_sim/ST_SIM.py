import torch as t
import torch.nn as nn
import numpy as np
from st_sim.DCT import get_dct_mat, calc_dct
from st_sim.Utilities import pad_blocks

pd = t.distributions.normal.Normal(0, 12)
eps = 1e-8

a = 62
c = 1 / 550
b = 1 - 250 * c
e = 77



class STSIMMetric:
    def __init__(self, fs, bl):
        self.bl = bl
        self.fs = fs

        if t.cuda.is_available():
            self.mat = get_dct_mat(bl).to("cuda")
        else:
            self.mat = get_dct_mat(bl)


    def apply(self, ref: t.Tensor, dist: t.Tensor, dct=True, eta=2 / 3):
        if t.any(t.isnan(dist)):
            print("ST-SIM: nan detected in input")

        if t.any(t.isnan(ref)):
            print("ST-SIM: nan detected in reference")

        percthres = perceptualThreshold(self.bl, self.fs).to(ref.device)

        avg = False
        if ref.size(1) > self.bl:
            ref = pad_blocks(ref, self.bl)
            dist = pad_blocks(dist, self.bl)
            avg = True

        mat_temp = self.mat
        if not ref.device == "cuda":
            mat_temp = mat_temp.to(ref.device)

        if dct:
            spect = calc_dct(ref, mat_temp)
        else:
            spect = ref
        spect = 20 * t.log10(t.abs(spect) + eps)

        if dct:
            dist_spect = calc_dct(dist, mat_temp)
        else:
            dist_spect = dist
        dist_spect = 20 * t.log10(t.abs(dist_spect) + eps)

        diffspect = pd.cdf(spect - percthres)
        diffspect_dist = pd.cdf(dist_spect - percthres)

        ssim = t.sum(diffspect * diffspect_dist, 1) / (
            t.sum(t.square(diffspect), 1) + eps
        )

        normalized = t.div(
            (t.sub(ref, t.mean(ref, 1)[:, None])), (t.std(ref, 1) + eps)[:, None]
        )
        normalized_dist = t.div(
            (t.sub(dist, t.mean(dist, 1)[:, None])), (t.std(dist, 1) + eps)[:, None]
        )

        tsim = t.sum(normalized * normalized_dist, 1) / t.sum(t.square(normalized), 1)

        tsim = t.clamp_min(tsim, 0)

        stsim = (tsim**eta) * (ssim ** (1 - eta))

        if avg:
            return t.mean(stsim)
        else:
            return stsim


def perceptualThreshold(bl, fs):
    freq = freq_vect(fs, bl)
    percthres = t.abs(a / (np.log10(b)) ** 2 * (t.log10(c * freq + b)) ** 2) - e
    percthres[percthres > 0] = 0
    return percthres


def freq_vect(fs, bl):
    freq = np.linspace(0, fs, 2 * bl)
    freq = freq[:bl]
    return t.from_numpy(freq)


class STSIMLoss(nn.Module):
    """
    Loss = 1 - ST-SIM(ref, pred)
    - ref: ground-truth signal (no grad)
    - pred: model output (grad flows here)
    """
    def __init__(self, fs, bl, eta=2/3, use_dct=True, reduction="mean"):
        super().__init__()
        assert reduction in ("none", "mean", "sum")
        self.metric = STSIMMetric(fs=fs, bl=bl)   # your class as-is
        self.eta = eta
        self.use_dct = use_dct
        self.reduction = reduction

    def forward(self, pred, ref):
        with t.cuda.amp.autocast(enabled=False):   # loss in fp32
            pred32 = pred.float()
            ref32  = ref.detach().float()
            stsim  = self.metric.apply(ref=ref32, dist=pred32, dct=True, eta=2/3).float()
            stsim  = t.clamp(stsim, 0.0, 1.0)
            loss32 = 1.0 - stsim
            loss32 = loss32.mean()
        return loss32.to(pred.dtype)  # match model param dtype

