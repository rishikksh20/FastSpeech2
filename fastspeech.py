#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FastSpeech related modules."""

import logging

import torch
from transformer import TTSPlot
from core.duration_modeling.duration_predictor import DurationPredictor
from core.duration_modeling.duration_predictor import DurationPredictorLoss
from core.energy_predictor.energy_predictor import EnergyPredictor
from core.energy_predictor.energy_predictor import EnergyPredictorLoss
from core.energy_predictor.energy_calculator import energy_to_one_hot
from core.pitch_predictor.pitch_predictor import PitchPredictor
from core.pitch_predictor.pitch_predictor import PitchPredictorLoss
from core.pitch_predictor.pitch_calculator import pitch_to_one_hot
from core.duration_modeling.length_regulator import LengthRegulator
from utils.util import make_non_pad_mask
from utils.util import make_pad_mask
from core.attention import MultiHeadedAttention
from core.embedding import PositionalEncoding
from core.embedding import ScaledPositionalEncoding
from core.encoder import Encoder
from core.initializer import initialize
import hparams as hp
from core.postnet import Postnet


class FeedForwardTransformer(torch.nn.Module):
    """Feed Forward Transformer for TTS a.k.a. FastSpeech.

    This is a module of FastSpeech, feed-forward Transformer with duration predictor described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_, which does not require any auto-regressive
    processing during inference, resulting in fast decoding compared with auto-regressive Transformer.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """


    def __init__(self, idim, odim):
        """Initialize feed-forward Transformer module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.

        """
        # initialize base classes

        torch.nn.Module.__init__(self)

        # fill missing arguments

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.reduction_factor = hp.reduction_factor
        self.use_scaled_pos_enc = hp.use_scaled_pos_enc
        self.use_masking = hp.use_masking

        # TODO(kan-bayashi): support reduction_factor > 1
        if self.reduction_factor != 1:
            raise NotImplementedError("Support only reduction_factor = 1.")

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim,
            embedding_dim=hp.adim,
            padding_idx=padding_idx
        )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=hp.adim,
            attention_heads=hp.aheads,
            linear_units=hp.eunits,
            num_blocks=hp.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            pos_enc_class=pos_enc_class,
            normalize_before=hp.encoder_normalize_before,
            concat_after=hp.encoder_concat_after,
            positionwise_layer_type=hp.positionwise_layer_type,
            positionwise_conv_kernel_size=hp.positionwise_conv_kernel_size
        )

        self.duration_predictor = DurationPredictor(
            idim=hp.adim,
            n_layers=hp.duration_predictor_layers,
            n_chans=hp.duration_predictor_chans,
            kernel_size=hp.duration_predictor_kernel_size,
            dropout_rate=hp.duration_predictor_dropout_rate,
        )

        self.energy_predictor = EnergyPredictor(
            idim=hp.adim,
            n_layers=hp.duration_predictor_layers,
            n_chans=hp.duration_predictor_chans,
            kernel_size=hp.duration_predictor_kernel_size,
            dropout_rate=hp.duration_predictor_dropout_rate,
        )

        self.pitch_predictor = PitchPredictor(
            idim=hp.adim,
            n_layers=hp.duration_predictor_layers,
            n_chans=hp.duration_predictor_chans,
            kernel_size=hp.duration_predictor_kernel_size,
            dropout_rate=hp.duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder because fastspeech's decoder is the same as encoder
        self.decoder = Encoder(
            idim=256,
            attention_dim=256,
            attention_heads=hp.aheads,
            linear_units=hp.dunits,
            num_blocks=hp.dlayers,
            input_layer=None,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            pos_enc_class=pos_enc_class,
            normalize_before=hp.decoder_normalize_before,
            concat_after=hp.decoder_concat_after,
            positionwise_layer_type=hp.positionwise_layer_type,
            positionwise_conv_kernel_size=hp.positionwise_conv_kernel_size
        )

        # define postnet
        self.postnet = (
            None
            if hp.postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=hp.postnet_layers,
                n_chans=hp.postnet_chans,
                n_filts=hp.postnet_filts,
                use_batch_norm=hp.use_batch_norm,
                dropout_rate=hp.postnet_dropout_rate,
            )
        )

        # define final projection
        self.feat_out = torch.nn.Linear(hp.adim, odim * hp.reduction_factor)

        # initialize parameters
        self._reset_parameters(init_type=hp.transformer_init,
                               init_enc_alpha=hp.initial_encoder_alpha,
                               init_dec_alpha=hp.initial_decoder_alpha)


        # define criterions
        self.duration_criterion = DurationPredictorLoss()
        self.energy_criterion = EnergyPredictorLoss()
        self.pitch_criterion = PitchPredictorLoss()
        self.criterion = torch.nn.L1Loss()

    def _forward(self, xs, ilens, ys=None, olens=None, ds=None, es=None, ps=None, is_inference=False):
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)
        # print("Ys :",ys.shape)     torch.Size([32, 868, 80])

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(ilens).to(xs.device)
        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, d_outs, ilens)  # (B, Lmax, adim)
            e_outs = self.energy_predictor.inference(hs)
            p_outs = self.pitch_predictor.inference(hs)
            one_hot_energy = energy_to_one_hot(e_outs)  # (B, Lmax, adim)
            one_hot_pitch = pitch_to_one_hot(p_outs, False)  # (B, Lmax, adim)
        else:
            with torch.no_grad():
                # ds = self.duration_calculator(xs, ilens, ys, olens)  # (B, Tmax)
                one_hot_energy = energy_to_one_hot(es) # (B, Lmax, adim)   torch.Size([32, 868, 256])
                # print("one_hot_energy:", one_hot_energy.shape)
                one_hot_pitch = pitch_to_one_hot(ps)   # (B, Lmax, adim)   torch.Size([32, 868, 256])
                # print("one_hot_pitch:", one_hot_pitch.shape)
            # print("Before Hs:", hs.shape)  torch.Size([32, 121, 256])
            d_outs = self.duration_predictor(hs, d_masks)  # (B, Tmax)
            # print("d_outs:", d_outs.shape)        torch.Size([32, 121])
            hs = self.length_regulator(hs, ds, ilens)  # (B, Lmax, adim)
            # print("After Hs:",hs.shape)  torch.Size([32, 868, 256])
            e_outs = self.energy_predictor(hs)
            # print("e_outs:", e_outs.shape)  torch.Size([32, 868])
            p_outs = self.pitch_predictor(hs)
            # print("p_outs:", p_outs.shape)   torch.Size([32, 868])
        hs = hs + one_hot_pitch 
        hs = hs + one_hot_energy
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

    def forward(self, xs, ilens, ys, olens, ds, es, ps, *args, **kwargs):
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
        xs = xs[:, :max(ilens)]
        ys = ys[:, :max(olens)]

        # forward propagation
        before_outs, after_outs, d_outs, e_outs, p_outs = self._forward(xs, ilens, ys, olens, ds, es, ps, is_inference=False)

        # modifiy mod part of groundtruth
        if hp.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        # apply mask to remove padded part
        if self.use_masking:
            in_masks = make_non_pad_mask(ilens).to(xs.device)
            d_outs = d_outs.masked_select(in_masks)
            ds = ds.masked_select(in_masks)
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            #e_outs = e_outs.masked_select(out_masks) # Write size
            #p_outs = p_outs.masked_select(out_masks) # Write size
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
        if hp.use_weighted_masking:
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

        # report extra information
        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed[-1].alpha.data.item()},
            ]
        #self.reporter.report(report_keys)

        return loss, report_keys

    def calculate_all_attentions(self, xs, ilens, ys, olens, ds, es, ps, *args, **kwargs):
        """Calculate all of the attention weights.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).

        Returns:
            dict: Dict of attention weights and outputs.

        """
        with torch.no_grad():
            # remove unnecessary padded part (for multi-gpus)
            xs = xs[:, :max(ilens)]
            ys = ys[:, :max(olens)]

            # forward propagation
            outs = self._forward(xs, ilens, ys, olens, ds, es, ps, is_inference=False)[0]

        att_ws_dict = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                attn = m.attn.cpu().numpy()
                if "encoder" in name:
                    attn = [a[:, :l, :l] for a, l in zip(attn, ilens.tolist())]
                elif "decoder" in name:
                    if "src" in name:
                        attn = [a[:, :ol, :il] for a, il, ol in zip(attn, ilens.tolist(), olens.tolist())]
                    elif "self" in name:
                        attn = [a[:, :l, :l] for a, l in zip(attn, olens.tolist())]
                    else:
                        logging.warning("unknown attention module: " + name)
                else:
                    logging.warning("unknown attention module: " + name)
                att_ws_dict[name] = attn
        att_ws_dict["predicted_fbank"] = [m[:l].T for m, l in zip(outs.cpu().numpy(), olens.tolist())]

        return att_ws_dict

    def inference(self, x, inference_args, *args, **kwargs):
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

        return outs[0], None, None


    def _source_mask(self, ilens):
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
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)


    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def _transfer_from_teacher(self, transferred_encoder_module):
        if transferred_encoder_module == "all":
            for (n1, p1), (n2, p2) in zip(self.encoder.named_parameters(),
                                          self.teacher.encoder.named_parameters()):
                assert n1 == n2, "It seems that encoder structure is different."
                assert p1.shape == p2.shape, "It seems that encoder size is different."
                p1.data.copy_(p2.data)
        elif transferred_encoder_module == "embed":
            student_shape = self.encoder.embed[0].weight.data.shape
            teacher_shape = self.teacher.encoder.embed[0].weight.data.shape
            assert student_shape == teacher_shape, "It seems that embed dimension is different."
            self.encoder.embed[0].weight.data.copy_(
                self.teacher.encoder.embed[0].weight.data)
        else:
            raise NotImplementedError("Support only all or embed.")

    @property
    def attention_plot_class(self):
        """Return plot class for attention weight plot."""
        return TTSPlot

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training. keys should match what `chainer.reporter` reports.

        If you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ["loss", "l1_loss", "duration_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]

        return plot_keys
