"""F0 extractor using DIO + Stonemask algorithm."""

import logging

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import numpy as np
import pyworld
import torch
import torch.nn.functional as F
import pywt

from scipy.interpolate import interp1d
from typeguard import check_argument_types


class Dio():
    """F0 estimation with dio + stonemask algortihm.
    This is f0 extractor based on dio + stonmask algorithm introduced in `WORLD:
    a vocoder-based high-quality speech synthesis system for real-time applications`_.
    .. _`WORLD: a vocoder-based high-quality speech synthesis system for real-time
        applications`: https://doi.org/10.1587/transinf.2015EDP7457
    Note:
        This module is based on NumPy implementation. Therefore, the computational graph
        is not connected.
    Todo:
        Replace this module with PyTorch-based implementation.
    """


    def __init__(
            self,
            fs: int = 22050,
            n_fft: int = 1024,
            hop_length: int = 256,
            f0min: Optional[int] = 71,
            f0max: Optional[int] = 400,
            use_token_averaged_f0: bool = False,
            use_continuous_f0: bool = True,
            use_log_f0: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_period = 1000 * hop_length / fs
        self.f0min = f0min
        self.f0max = f0max
        self.use_token_averaged_f0 = use_token_averaged_f0
        self.use_continuous_f0 = use_continuous_f0
        self.use_log_f0 = use_log_f0

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            f0min=self.f0min,
            f0max=self.f0max,
            use_token_averaged_f0=self.use_token_averaged_f0,
            use_continuous_f0=self.use_continuous_f0,
            use_log_f0=self.use_log_f0,
        )

    def forward(
            self,
            input: torch.Tensor,
            feats_lengths: torch.Tensor = None,
            durations: torch.Tensor = None,
            utterance: list = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If not provide, we assume that the inputs have the same length
        # F0 extraction

        # input shape = [T,]
        pitch, pitch_log = self._calculate_f0(input)
        # (Optional): Adjust length to match with the mel-spectrogram
        if feats_lengths is not None:
            pitch = [
                self._adjust_num_frames(p, fl).view(-1)
                for p, fl in zip(pitch, feats_lengths)
            ]
            pitch_log = [
                self._adjust_num_frames(p, fl).view(-1)
                for p, fl in zip(pitch_log, feats_lengths)
            ]

        pitch_log_norm, mean, std = self._normalize(pitch_log, durations)
        coefs = self._cwt(pitch_log_norm.numpy())
        # (Optional): Average by duration to calculate token-wise f0
        if self.use_token_averaged_f0:
            pitch = self._average_by_duration(pitch, durations)
            pitch_lengths = len(durations)
        else:
            pitch_lengths = 22 #input.new_tensor([len(p) for p in pitch], dtype=torch.long)
        # Return with the shape (B, T, 1)
        return pitch, mean, std, coefs


    def _calculate_f0(self, input: torch.Tensor) -> torch.Tensor:
        x = input.cpu().numpy().astype(np.double)
        #print(self.frame_period)
        f0, timeaxis = pyworld.dio(
            x,
            self.fs,
            f0_floor=self.f0min,
            f0_ceil=self.f0max,
            frame_period=self.frame_period,
        )

        f0 = pyworld.stonemask(x, f0, timeaxis, self.fs)
        if self.use_continuous_f0:
            f0 = self._convert_to_continuous_f0(f0)

        if self.use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0_log[nonzero_idxs] = np.log(f0[nonzero_idxs])

        return input.new_tensor(f0.reshape(-1), dtype=torch.float), input.new_tensor(f0_log.reshape(-1), dtype=torch.float)


    @staticmethod
    def _adjust_num_frames(x: torch.Tensor, num_frames: torch.Tensor) -> torch.Tensor:
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x


    @staticmethod
    def _convert_to_continuous_f0(f0: np.array) -> np.array:
        if (f0 == 0).all():
            logging.warn("All frames seems to be unvoiced.")
            return f0

        # padding start and end of f0 sequence
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        #start_idx = np.where(f0 == start_f0)[0][0]
        #end_idx = np.where(f0 == end_f0)[0][-1]
        if f0[0] == 0:
            f0[0] = 1.845
        if f0[-1] == 0:
            f0[-1] = 1.845      # get non-zero frame index
        nonzero_idxs = np.where(f0 != 0)[0]
        # perform linear interpolation
        interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs], kind= 'linear')
        f0 = interp_fn(np.arange(0, f0.shape[0]))
        return f0

    @staticmethod
    def _average_by_duration(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        #print(d.sum(), len(x))
        if d.sum() != len(x):
            d[-1] += 1
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            x[start:end].mean() if len(x[start:end]) != 0 else x.new_tensor(0.0)
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        return torch.stack(x_avg)

    def _normalize(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor :
        #if d.sum() != len(x):
        #    d[-1] += 1
        #d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        norm_pitch = (x - x.mean())/x.std()
        p_average = x.mean()
        p_std = x.std()

        """
        for i in range(0, len(d_cumsum)-1):
            pitch_i = x[d_cumsum[i]: d_cumsum[i+1]]
            #print(pitch_i, "Pitch input")
            p_average.append(pitch_i.mean())
            p_std.append(pitch_i.std())
            #print(pitch_i.std(), "pitch std")
            #print(pitch_i.mean(), "pitch mean")
            norm_pitch.extend((pitch_i - pitch_i.mean())/pitch_i.std())
            #print(norm_pitch[i], "Normalised pitch")
        #print(norm_pitch, p_average, p_std)
        """
        return norm_pitch, p_average, p_std

    def _cwt(self, x: torch.Tensor) -> torch.Tensor:
        scales = np.arange(1,11)
        coefs, freq = pywt.cwt(x, scales, 'mexh') #coefs shape = [10, T]

        return coefs
