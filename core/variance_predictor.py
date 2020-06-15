import torch
from core.layer_norm import LayerNorm

class VariancePredictor(torch.nn.Module):

    def __init__(self, idim, n_layers=2, n_chans=256, out=1, kernel_size=3, dropout_rate=0.5, offset=1.0):
        super(VariancePredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, out)


    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if is_inference:
            # NOTE: calculate in linear domain
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, x_masks, True)