import torch
import torch.nn.functional as F
from typing import Optional
from core.modules import LayerNorm
#import pycwt
import numpy as np
from sklearn import preprocessing

class VariancePredictor(torch.nn.Module):
    def __init__(
        self,
        idim: int,
        n_layers: int = 2,
        n_chans: int = 256,
        out: int = 1,
        kernel_size: int = 3,
        dropout_rate: float = 0.5,
        offset: float = 1.0,
    ):
        super(VariancePredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, out)

    def _forward(
        self,
        xs: torch.Tensor,
        is_inference: bool = False,
        is_log_output: bool = False,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if is_inference and is_log_output:
            #     # NOTE: calculate in linear domain
            xs = torch.clamp(
                torch.round(xs.exp() - self.offset), min=0
            ).long()  # avoid negative value
        xs = xs * alpha

        return xs

    def forward(
        self, xs: torch.Tensor, x_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        xs = self._forward(xs)
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs

    def inference(
        self, xs: torch.Tensor, is_log_output: bool = False, alpha: float = 1.0
    ) -> torch.Tensor:
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(
            xs, is_inference=True, is_log_output=is_log_output, alpha=alpha
        )


class EnergyPredictor(torch.nn.Module):
    def __init__(
        self,
        idim,
        n_layers=2,
        n_chans=256,
        kernel_size=3,
        dropout_rate=0.1,
        offset=1.0,
        min=0,
        max=0,
        n_bins=256,
    ):
        """Initilize Energy predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(EnergyPredictor, self).__init__()
        # self.bins = torch.linspace(min, max, n_bins - 1).cuda()
        self.register_buffer("energy_bins", torch.linspace(min, max, n_bins - 1))
        self.predictor = VariancePredictor(idim)

    def forward(self, xs: torch.Tensor, x_masks: torch.Tensor):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self.predictor(xs, x_masks)

    def inference(self, xs: torch.Tensor, alpha: float = 1.0):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        out = self.predictor.inference(xs, False, alpha=alpha)
        #print(out.shape, type(out))
        #out = torch.from_numpy(np.load("/results/chkpts/LJ/Fastspeech2_V2/data/energy/LJ001-0001.npy")).cuda()
        #print(out, "Energy Pricted")
        out = torch.exp(out)
        return self.to_one_hot(out), out  # Need to do One hot code

    def to_one_hot(self, x):
        # e = de_norm_mean_std(e, hp.e_mean, hp.e_std)
        # For pytorch > = 1.6.0

        quantize = torch.bucketize(x, self.energy_bins).to(device=x.device)  # .cuda()
        return F.one_hot(quantize.long(), 256).float()


class PitchPredictor(torch.nn.Module):
    def __init__(
        self,
        idim,
        n_layers=2,
        n_chans=384,
        kernel_size=3,
        dropout_rate=0.1,
        offset=1.0,
        min=0,
        max=0,
        n_bins=256,
        out=5,
    ):
        """Initilize pitch predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(PitchPredictor, self).__init__()
        # self.bins = torch.exp(torch.linspace(torch.log(torch.tensor(min)), torch.log(torch.tensor(max)), n_bins - 1)).cuda()
        self.register_buffer(
            "pitch_bins",
            torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(min)),
                    torch.log(torch.tensor(max)),
                    n_bins - 1,
                )
            ),
        )
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.spectrogram_out = torch.nn.Linear(n_chans, out)
        self.mean = torch.nn.Linear(n_chans, 1)
        self.std = torch.nn.Linear(n_chans, 1)

    def forward(self, xs: torch.Tensor, olens: torch.Tensor, x_masks: torch.Tensor):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = xs.transpose(1, -1)
        f0_spec = self.spectrogram_out(xs)  # (B, Tmax, 10)

        if x_masks is not None:
            # print("olen:", olens)
            #f0_spec = f0_spec.transpose(1, -1)
            # print("F0 spec dimension:", f0_spec.shape)
            # print("x_masks dimension:", x_masks.shape)
            f0_spec = f0_spec.masked_fill(x_masks, 0.0)
            #f0_spec = f0_spec.transpose(1, -1)
            # print("F0 spec dimension:", f0_spec.shape)
            #xs = xs.transpose(1, -1)
            xs = xs.masked_fill(x_masks, 0.0)
            #xs = xs.transpose(1, -1)
            # print("xs dimension:", xs.shape)
        x_avg = xs.sum(dim=1).squeeze(1)
        # print(x_avg)
        # print("xs dim :", x_avg.shape)
        # print("olens ;", olens.shape)
        if olens is not None:
            x_avg = x_avg / olens.unsqueeze(1)
        # print(x_avg)
        f0_mean = self.mean(x_avg).squeeze(-1)
        f0_std = self.std(x_avg).squeeze(-1)

        # if x_masks is not None:
        #     f0_spec = f0_spec.masked_fill(x_masks, 0.0)

        return f0_spec, f0_mean, f0_std

    def inference(self, xs: torch.Tensor, olens = None, alpha: float = 1.0):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        f0_spec, f0_mean, f0_std = self.forward(xs, olens, x_masks=None)  # (B, Tmax, 10)
        #print(f0_spec)
        f0_reconstructed = self.inverse(f0_spec, f0_mean, f0_std)
        #print(f0_reconstructed)
        #f0_reconstructed = torch.from_numpy(np.load("/results/chkpts/LJ/Fastspeech2_V2/data/pitch/LJ001-0001.npy").reshape(1,-1)).cuda()
        #print(f0_reconstructed, "Pitch coef output")

        return self.to_one_hot(f0_reconstructed), f0_reconstructed

    def to_one_hot(self, x: torch.Tensor):
        # e = de_norm_mean_std(e, hp.e_mean, hp.e_std)
        # For pytorch > = 1.6.0

        quantize = torch.bucketize(x, self.pitch_bins).to(device=x.device)  # .cuda()
        return F.one_hot(quantize.long(), 256).float()

    def inverse(self, Wavelet_lf0, f0_mean, f0_std):
        scales =  np.array([0.01, 0.02, 0.04, 0.08, 0.16])  #np.arange(1,11)
        #print(Wavelet_lf0.shape)
        Wavelet_lf0 = Wavelet_lf0.squeeze(0).cpu().numpy()
        lf0_rec = np.zeros([Wavelet_lf0.shape[0], len(scales)])
        for i in range(0,len(scales)):
            lf0_rec[:,i] = Wavelet_lf0[:,i]*((i+200+2.5)**(-2.5))

        lf0_rec_sum = np.sum(lf0_rec,axis = 1)
        lf0_rec_sum_norm = preprocessing.scale(lf0_rec_sum)

        f0_reconstructed = (torch.Tensor(lf0_rec_sum_norm).cuda()*f0_std) + f0_mean

        f0_reconstructed = torch.exp(f0_reconstructed)
        #print(f0_reconstructed.shape)
        #print(f0_reconstructed.shape)
        return f0_reconstructed.reshape(1,-1)


class PitchPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(PitchPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        # NOTE: We convert the output in log domain low error value
        # print("Output :", outputs[0])
        # print("Before Output :", targets[0])
        # targets = torch.log(targets.float() + self.offset)
        # print("Before Output :", targets[0])
        # outputs = torch.log(outputs.float() + self.offset)
        loss = self.criterion(outputs, targets)
        # print(loss)
        return loss


class EnergyPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(EnergyPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        # NOTE: outputs is in log domain while targets in linear
        # targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss
