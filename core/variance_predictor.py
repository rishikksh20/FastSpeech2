import torch
import torch.nn.functional as F
from typing import Optional
from core.modules import LayerNorm


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
        return self.to_one_hot(out)  # Need to do One hot code

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
        return self.to_one_hot(out)

    def to_one_hot(self, x: torch.Tensor):
        # e = de_norm_mean_std(e, hp.e_mean, hp.e_std)
        # For pytorch > = 1.6.0

        quantize = torch.bucketize(x, self.pitch_bins).to(device=x.device)  # .cuda()
        return F.one_hot(quantize.long(), 256).float()


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
