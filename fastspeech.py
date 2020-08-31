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
from utils.util import make_non_pad_mask
from utils.util import make_pad_mask
from core.embedding import PositionalEncoding
from core.embedding import ScaledPositionalEncoding
from core.encoder import Encoder
from core.modules import initialize
from core.modules import Postnet
from typeguard import check_argument_types
from typing import Dict, Tuple, Sequence

class FeedForwardTransformer(torch.nn.Module):
    """Feed Forward Transformer for TTS a.k.a. FastSpeech.
    This is a module of FastSpeech, feed-forward Transformer with duration predictor described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_, which does not require any auto-regressive
    processing during inference, resulting in fast decoding compared with auto-regressive Transformer.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """


    def __init__(self, idim:int, odim:int, hp:Dict):
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
        pos_enc_class = ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim,
            embedding_dim=hp.model.adim,
            padding_idx=padding_idx
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
            positionwise_conv_kernel_size=hp.model.positionwise_conv_kernel_size
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
            min = hp.data.e_min,
            max = hp.data.e_max,
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
            positionwise_conv_kernel_size=hp.model.positionwise_conv_kernel_size
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
        self._reset_parameters(init_type=hp.model.transformer_init,
                               init_enc_alpha=hp.model.initial_encoder_alpha,
                               init_dec_alpha=hp.model.initial_decoder_alpha)


        # define criterions
        self.duration_criterion = DurationPredictorLoss()
        self.energy_criterion = EnergyPredictorLoss()
        self.pitch_criterion = PitchPredictorLoss()
        self.criterion = torch.nn.L1Loss(reduction='mean')
        self.use_weighted_masking = hp.model.use_weighted_masking

    def _forward(self, xs: torch.Tensor, ilens: torch.Tensor, olens: torch.Tensor = None,
                 ds: torch.Tensor = None, es: torch.Tensor = None, ps: torch.Tensor = None,
                 is_inference: bool = False) -> Sequence[torch.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens) # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])

        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])
        # print("ys :", ys.shape)


        # forward duration predictor and length regulator
        d_masks = make_pad_mask(ilens).to(xs.device)

        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, d_outs, ilens)  # (B, Lmax, adim)
            one_hot_energy = self.energy_predictor.inference(hs) # (B, Lmax, adim)
            one_hot_pitch = self.pitch_predictor.inference(hs) # (B, Lmax, adim)
        else:
            with torch.no_grad():
                # ds = self.duration_calculator(xs, ilens, ys, olens)  # (B, Tmax)
                one_hot_energy = self.energy_predictor.to_one_hot(es) # (B, Lmax, adim)   torch.Size([32, 868, 256])
                # print("one_hot_energy:", one_hot_energy.shape)
                one_hot_pitch = self.pitch_predictor.to_one_hot(ps)   # (B, Lmax, adim)   torch.Size([32, 868, 256])
                # print("one_hot_pitch:", one_hot_pitch.shape)
            mel_masks = make_pad_mask(olens).to(xs.device)
            #print("Before Hs:", hs.shape)  # torch.Size([32, 121, 256])
            d_outs = self.duration_predictor(hs, d_masks)  # (B, Tmax)
            #print("d_outs:", d_outs.shape)      #  torch.Size([32, 121])
            hs = self.length_regulator(hs, ds, ilens)  # (B, Lmax, adim)
            #print("After Hs:",hs.shape)  #torch.Size([32, 868, 256])
            e_outs = self.energy_predictor(hs, mel_masks)
            #print("e_outs:", e_outs.shape)  #torch.Size([32, 868])
            p_outs = self.pitch_predictor(hs, mel_masks)
            #print("p_outs:", p_outs.shape)   #torch.Size([32, 868])
        hs = hs + self.pitch_embed(one_hot_pitch) # (B, Lmax, adim)
        hs = hs + self.energy_embed(one_hot_energy) # (B, Lmax, adim)
        # forward decoder
        if olens is not None:
            h_masks = self._source_mask(olens)
        else:
            h_masks = None

        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        if is_inference:
            return before_outs, after_outs, d_outs
        else:
            return before_outs, after_outs, d_outs, e_outs, p_outs

    def forward(self, xs: torch.Tensor, ilens: torch.Tensor, ys: torch.Tensor, olens: torch.Tensor, ds: torch.Tensor,
                es: torch.Tensor, ps: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).
        Returns:
            Tensor: Loss value.
        """
        # remove unnecessary padded part (for multi-gpus)
        xs = xs[:, :max(ilens)] # torch.Size([32, 121]) -> [B, Tmax]
        ys = ys[:, :max(olens)] # torch.Size([32, 868, 80]) -> [B, Lmax, odim]

        # forward propagation
        before_outs, after_outs, d_outs, e_outs, p_outs = self._forward(xs, ilens, olens, ds, es, ps,
                                                                        is_inference=False)

        # modifiy mod part of groundtruth
        # if hp.model.reduction_factor > 1:
        #     olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
        #     max_olen = max(olens)
        #     ys = ys[:, :max_olen]

        # apply mask to remove padded part
        if self.use_masking:
            in_masks = make_non_pad_mask(ilens).to(xs.device)
            d_outs = d_outs.masked_select(in_masks)
            ds = ds.masked_select(in_masks)
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            mel_masks = make_non_pad_mask(olens).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            es = es.masked_select(mel_masks)  # Write size
            ps = ps.masked_select(mel_masks)  # Write size
            e_outs = e_outs.masked_select(mel_masks) # Write size
            p_outs = p_outs.masked_select(mel_masks) # Write size
            after_outs = (
                after_outs.masked_select(out_masks) if after_outs is not None else None
            )
            ys = ys.masked_select(out_masks)

        # calculate loss
        before_loss = self.criterion(before_outs, ys)
        after_loss = 0
        if after_outs is not None:
            after_loss = self.criterion(after_outs, ys)
            l1_loss = before_loss + after_loss
        duration_loss = self.duration_criterion(d_outs, ds)
        energy_loss = self.energy_criterion(e_outs, es)
        pitch_loss = self.pitch_criterion(p_outs, ps)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                    duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )

        loss = l1_loss + duration_loss + energy_loss + pitch_loss
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"before_loss": before_loss.item()},
            {"after_loss": after_loss.item()},
            {"duration_loss": duration_loss.item()},
            {"energy_loss": energy_loss.item()},
            {"pitch_loss": pitch_loss.item()},
            {"loss": loss.item()},
        ]

        #self.reporter.report(report_keys)

        return loss, report_keys


    def inference(self, x: torch.Tensor) -> torch.Tensor:
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
        _, outs, _ = self._forward(xs, ilens, is_inference=True)  # (L, odim)

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
        x_masks = make_non_pad_mask(ilens).to(device = next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)


    def _reset_parameters(self, init_type: str, init_enc_alpha: float=1.0, init_dec_alpha: float=1.0):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)
