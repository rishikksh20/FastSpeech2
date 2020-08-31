import torch
import torch.nn.functional as F
from core.variance_predictor import VariancePredictor


class EnergyPredictor(torch.nn.Module):

    def __init__(self ,idim, n_layers=2, n_chans=256, kernel_size=3, dropout_rate=0.1, offset=1.0,
                 min = 0, max = 0, n_bins = 256):
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
        self.bins = torch.linspace(min, max, n_bins - 1)
        self.predictor = VariancePredictor(idim)

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self.predictor(xs, x_masks)

    def inference(self, xs, x_masks=None, alpha = 1.0):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        out = self.predictor.inference(xs, x_masks, False, alpha=alpha)
        return self.to_one_hot(out) # Need to do One hot code

    def to_one_hot(self, x):
        # e = de_norm_mean_std(e, hp.e_mean, hp.e_std)
        # For pytorch > = 1.6.0

        quantize = torch.bucketize(x, self.bins)
        return F.one_hot(quantize.long(), 256).float()



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
