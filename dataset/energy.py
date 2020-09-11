"""Energy extractor."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import torch
import torch.nn.functional as F
from typeguard import check_argument_types
from utils.stft import TacotronSTFT


class Energy():
    """Energy extractor."""

    def __init__(
        self,
        fs: int= 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        window: Optional[str] = "hann",
        num_mel: int = 80,
        fmin: int = 0,
        fmax: int = 8000,
        use_token_averaged_energy: bool = True,
    ):
        assert check_argument_types()
        super().__init__()

        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.use_token_averaged_energy = use_token_averaged_energy

        self.stft = TacotronSTFT(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=num_mel,
            sampling_rate=fs,
            mel_fmin=fmin,
            mel_fmax=fmax,
        )

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            use_token_averaged_energy=self.use_token_averaged_energy,
        )

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor = None,
        feats_lengths: torch.Tensor = None,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If not provide, we assume that the inputs have the same length
        if input_lengths is None:
            input_lengths = (
                input.new_ones(input.shape[0], dtype=torch.long) * input.shape[1]
            )

        # Domain-conversion: e.g. Stft: time -> time-freq
        #input_stft, energy_lengths = self.stft(input, input_lengths)
        mel, mag = self.stft.mel_spectrogram(input)

        energy = torch.norm(mag, dim=0)


        # (Optional): Average by duration to calculate token-wise energy
        if self.use_token_averaged_energy:
            energy = [
                self._average_by_duration(e[:el].view(-1), d)
                for e, el, d in zip(energy, feats_lengths, durations)
            ]
            energy_lengths = durations_lengths

        # Return with the shape (B, T, 1)
        return energy.unsqueeze(-1), energy_lengths

    @staticmethod
    def _average_by_duration(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        assert d.sum() == len(x)
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            x[start:end].mean() if len(x[start:end]) != 0 else x.new_tensor(0.0)
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        return torch.stack(x_avg)

    @staticmethod
    def _adjust_num_frames(x: torch.Tensor, num_frames: torch.Tensor) -> torch.Tensor:
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x