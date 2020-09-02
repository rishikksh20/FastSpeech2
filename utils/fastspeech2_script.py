#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FastSpeech related loss."""

import logging

import torch
from core.duration_modeling.duration_predictor import DurationPredictor
from core.duration_modeling.duration_predictor import DurationPredictorLoss
from core.variance_predictor import EnergyPredictor, EnergyPredictorLoss
from core.variance_predictor import PitchPredictor, PitchPredictorLoss
from core.duration_modeling.length_regulator import LengthRegulator
from utils.util import make_non_pad_mask_script
from utils.util import make_pad_mask_script
from core.embedding import PositionalEncoding
from core.embedding import ScaledPositionalEncoding
from core.encoder import Encoder
from core.modules import initialize
from core.modules import Postnet
from typeguard import check_argument_types
from typing import Dict, Tuple, Sequence


class FeedForwardTransformer(torch.nn.Module):
    def __init__(self, idim: int, odim: int, hp: Dict):
        """Initialize feed-forward Transformer module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
        """
        # initialize base classes
        assert check_argument_types()
        torch.nn.Module.__init__(self)

        # fill missing arguments

        # store hyperparameters
        self.idim = idim
        self.odim = odim

        self.use_scaled_pos_enc = hp.model.use_scaled_pos_enc
        self.use_masking = hp.model.use_masking

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=hp.model.adim, padding_idx=padding_idx
        )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=hp.model.adim,
            attention_heads=hp.model.aheads,
            linear_units=hp.model.eunits,
            num_blocks=hp.model.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            pos_enc_class=pos_enc_class,
            normalize_before=hp.model.encoder_normalize_before,
            concat_after=hp.model.encoder_concat_after,
            positionwise_layer_type=hp.model.positionwise_layer_type,
            positionwise_conv_kernel_size=hp.model.positionwise_conv_kernel_size,
        )

        self.duration_predictor = DurationPredictor(
            idim=hp.model.adim,
            n_layers=hp.model.duration_predictor_layers,
            n_chans=hp.model.duration_predictor_chans,
            kernel_size=hp.model.duration_predictor_kernel_size,
            dropout_rate=hp.model.duration_predictor_dropout_rate,
        )

        self.energy_predictor = EnergyPredictor(
            idim=hp.model.adim,
            n_layers=hp.model.duration_predictor_layers,
            n_chans=hp.model.duration_predictor_chans,
            kernel_size=hp.model.duration_predictor_kernel_size,
            dropout_rate=hp.model.duration_predictor_dropout_rate,
            min=hp.data.e_min,
            max=hp.data.e_max,
        )
        self.energy_embed = torch.nn.Linear(hp.model.adim, hp.model.adim)

        self.pitch_predictor = PitchPredictor(
            idim=hp.model.adim,
            n_layers=hp.model.duration_predictor_layers,
            n_chans=hp.model.duration_predictor_chans,
            kernel_size=hp.model.duration_predictor_kernel_size,
            dropout_rate=hp.model.duration_predictor_dropout_rate,
            min=hp.data.p_min,
            max=hp.data.p_max,
        )
        self.pitch_embed = torch.nn.Linear(hp.model.adim, hp.model.adim)

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder because fastspeech's decoder is the same as encoder
        self.decoder = Encoder(
            idim=256,
            attention_dim=256,
            attention_heads=hp.model.aheads,
            linear_units=hp.model.dunits,
            num_blocks=hp.model.dlayers,
            input_layer=None,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            pos_enc_class=pos_enc_class,
            normalize_before=hp.model.decoder_normalize_before,
            concat_after=hp.model.decoder_concat_after,
            positionwise_layer_type=hp.model.positionwise_layer_type,
            positionwise_conv_kernel_size=hp.model.positionwise_conv_kernel_size,
        )

        # define postnet
        self.postnet = (
            None
            if hp.model.postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=hp.model.postnet_layers,
                n_chans=hp.model.postnet_chans,
                n_filts=hp.model.postnet_filts,
                use_batch_norm=hp.model.use_batch_norm,
                dropout_rate=hp.model.postnet_dropout_rate,
            )
        )

        # define final projection
        self.feat_out = torch.nn.Linear(hp.model.adim, odim * hp.model.reduction_factor)

        # initialize parameters
        self._reset_parameters(
            init_type=hp.model.transformer_init,
            init_enc_alpha=hp.model.initial_encoder_alpha,
            init_dec_alpha=hp.model.initial_decoder_alpha,
        )

        # define criterions
        self.duration_criterion = DurationPredictorLoss()
        self.energy_criterion = EnergyPredictorLoss()
        self.pitch_criterion = PitchPredictorLoss()
        self.criterion = torch.nn.L1Loss(reduction="mean")
        self.use_weighted_masking = hp.model.use_weighted_masking

    def _forward(self, xs: torch.Tensor, ilens: torch.Tensor):
        # forward encoder
        x_masks = self._source_mask(
            ilens
        )  # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])

        hs, _ = self.encoder(
            xs, x_masks
        )  # (B, Tmax, adim) -> torch.Size([32, 121, 256])
        # print("ys :", ys.shape)

        # # forward duration predictor and length regulator
        d_masks = make_pad_mask_script(ilens).to(xs.device)

        d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, Tmax)
        hs = self.length_regulator(hs, d_outs, ilens)  # (B, Lmax, adim)

        one_hot_energy = self.energy_predictor.inference(hs)  # (B, Lmax, adim)

        one_hot_pitch = self.pitch_predictor.inference(hs)  # (B, Lmax, adim)

        hs = hs + self.pitch_embed(one_hot_pitch)  # (B, Lmax, adim)
        hs = hs + self.energy_embed(one_hot_energy)  # (B, Lmax, adim)

        # # forward decoder
        #  h_masks = self._source_mask(olens) we can find olens from length regulator and then calculate mask
        # h_masks = torch.empty(0)

        zs, _ = self.decoder(hs, None)  # (B, Lmax, adim)

        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(
            1, 2
        )
        return after_outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate the sequence of features given the sequences of characters.
        Args:
            x (Tensor): Input sequence of characters (T,).
            inference_args (Namespace): Dummy for compatibility.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).
        Returns:
            Tensor: Output sequence of features (1, L, odim).
            None: Dummy for compatibility.
            None: Dummy for compatibility.
        """
        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)

        # inference
        outs = self._forward(xs, ilens)  # (L, odim)

        return outs[0]

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.
        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask_script(ilens)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float = 1.0, init_dec_alpha: float = 1.0
    ):
        # initialize parameters
        initialize(self, init_type)
        #
        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)
