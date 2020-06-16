import torch
from core.variance_predictor import VariancePredictor


class PitchPredictor(torch.nn.Module):

    def __init__(self ,idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
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

    def inference(self, xs, x_masks=None):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self.predictor.inference(xs, x_masks) # Need to do One hot code



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
        # targets = torch.log(targets.float() + self.offset)
        # outputs = torch.log(outputs.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss