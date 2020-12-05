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
import pycwt as wavelet

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
            f0max: Optional[int] = 500,
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
        pitch = self._calculate_f0(input)
        # (Optional): Adjust length to match with the mel-spectrogram
        pitch, pitch_log = self._convert_to_continuous_f0(pitch)

        if feats_lengths is not None:
            pitch = [
                self._adjust_num_frames(p, fl).view(-1)
                for p, fl in zip(pitch, feats_lengths)
            ]
            pitch_log = [
                self._adjust_num_frames(p, fl).view(-1)
                for p, fl in zip(pitch_log, feats_lengths)
            ]

        pitch_log_norm, mean, std = self._normalize(pitch_log)
        coefs, scales = self._cwt(pitch_log_norm)
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
        _f0, t = pyworld.dio(x, self.fs, f0_floor = self.f0min, f0_ceil=self.f0max, frame_period=self.frame_period)            # raw pitch extractor
        f0 = pyworld.stonemask(x, _f0, t, self.fs)  # pitch refinement
        #sp = pw.cheaptrick(x, f0, t, self.fs, fft_size=self.n_fft)
        #ap = pw.d4c(x, f0, t, fs, fft_size=self.n_fft) # extract aperiodicity

        return input.new_tensor(f0.reshape(-1), dtype=torch.float)


    @staticmethod
    def _adjust_num_frames(x: torch.Tensor, num_frames: torch.Tensor) -> torch.Tensor:
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x


    @staticmethod
    def _convert_to_continuous_f0(f0: np.array) -> np.array:

        uv = np.float64(f0 != 0)
        # get start and end of f0
        if (f0 == 0).all():
            print("all of the f0 values are 0.")
            return uv, f0
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]

        # padding start and end of f0 sequence
        start_idx = np.where(f0 == start_f0)[0][0]
        end_idx = np.where(f0 == end_f0)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx:] = end_f0

        # get non-zero frame index
        nz_frames = np.where(f0 != 0)[0]

        # perform linear interpolation
        f = interp1d(nz_frames, f0[nz_frames])
        cont_f0 = f(np.arange(0, f0.shape[0]))
        cont_f0_lpf = np.log(cont_f0)

        return cont_f0, cont_f0_lpf

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

    def _normalize(self, x: torch.Tensor) -> torch.Tensor :

        norm_pitch = (x - x.mean())/x.std()
        return norm_pitch, x.mean(), x.std()

    def _cwt(self, x: torch.Tensor) -> np.array:
        mother = wavelet.MexicanHat()
        dt = 0.005
        dj = 2
        s0 = dt*2
        J = 5 - 1
        Wavelet_lf0, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(x, dt, dj, s0, J, mother)
        Wavelet_lf0 = np.real(Wavelet_lf0).T

        return Wavelet_lf0, scales
