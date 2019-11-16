#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TTS-Transformer related modules."""

import logging

import torch
import torch.nn.functional as F

from core.mask import subsequent_mask
from modules.loss import TransformerLoss
from modules.loss import GuidedMultiHeadAttentionLoss
from utils.util import make_non_pad_mask
from modules.postnet import Postnet
from modules.prenet import Prenet as DecoderPrenet
from modules.prenet import Encoder as EncoderPrenet
from core.attention import MultiHeadedAttention
from core.decoder import Decoder
from core.embedding import PositionalEncoding
from core.embedding import ScaledPositionalEncoding
from core.encoder import Encoder
from core.initializer import initialize
from core.plot import _plot_and_save_attention
from core.plot import PlotAttentionReport
from utils.cli_utils import strtobool
from utils.fill_missing_args import fill_missing_args
import hparams as hp

class TTSPlot(PlotAttentionReport):
    """Attention plot module for TTS-Transformer."""

    def plotfn(self, input_lengths, output_lengths, attn_dict, outdir, suffix="png", savefn=None):
        """Plot multi head attentions.

        Args:
            data (dict): Utts info from json file.
            attn_dict (dict): Multi head attention dict.
                Values should be numpy.ndarray (H, L, T)
            outdir (str): Directory name to save figures.
            suffix (str): Filename suffix including image type (e.g., png).
            savefn (function): Function to save figures.

        """
        import matplotlib.pyplot as plt
        for name, att_ws in attn_dict.items():
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.%s.%s" % (
                    outdir, idx, name, suffix)
                if "fbank" in name:
                    fig = plt.Figure()
                    ax = fig.subplots(1, 1)
                    ax.imshow(att_w, aspect="auto")
                    ax.set_xlabel("frames")
                    ax.set_ylabel("fbank coeff")
                    fig.tight_layout()
                else:
                    fig = _plot_and_save_attention(att_w, filename)
                savefn(fig, filename)


class Transformer(torch.nn.Module):
    """Text-to-Speech Transformer module.

    This is a module of text-to-speech Transformer described in `Neural Speech Synthesis with Transformer Network`_,
    which convert the sequence of characters or phonemes into the sequence of Mel-filterbanks.

    .. _`Neural Speech Synthesis with Transformer Network`:
        https://arxiv.org/pdf/1809.08895.pdf

    """

    @staticmethod
    def add_arguments(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group("core model setting")
        # network structure related
        group.add_argument("--embed-dim", default=512, type=int,
                           help="Dimension of character embedding in encoder prenet")
        group.add_argument("--eprenet-conv-layers", default=3, type=int,
                           help="Number of encoder prenet convolution layers")
        group.add_argument("--eprenet-conv-chans", default=256, type=int,
                           help="Number of encoder prenet convolution channels")
        group.add_argument("--eprenet-conv-filts", default=5, type=int,
                           help="Filter size of encoder prenet convolution")
        group.add_argument("--dprenet-layers", default=2, type=int,
                           help="Number of decoder prenet layers")
        group.add_argument("--dprenet-units", default=256, type=int,
                           help="Number of decoder prenet hidden units")
        group.add_argument("--elayers", default=3, type=int,
                           help="Number of encoder layers")
        group.add_argument("--eunits", default=1536, type=int,
                           help="Number of encoder hidden units")
        group.add_argument("--adim", default=384, type=int,
                           help="Number of attention transformation dimensions")
        group.add_argument("--aheads", default=4, type=int,
                           help="Number of heads for multi head attention")
        group.add_argument("--dlayers", default=3, type=int,
                           help="Number of decoder layers")
        group.add_argument("--dunits", default=1536, type=int,
                           help="Number of decoder hidden units")
        group.add_argument("--postnet-layers", default=5, type=int,
                           help="Number of postnet layers")
        group.add_argument("--postnet-chans", default=256, type=int,
                           help="Number of postnet channels")
        group.add_argument("--postnet-filts", default=5, type=int,
                           help="Filter size of postnet")
        group.add_argument("--use-scaled-pos-enc", default=True, type=strtobool,
                           help="Use trainable scaled positional encoding instead of the fixed scale one.")
        group.add_argument("--use-batch-norm", default=True, type=strtobool,
                           help="Whether to use batch normalization")
        group.add_argument("--encoder-normalize-before", default=False, type=strtobool,
                           help="Whether to apply layer norm before encoder block")
        group.add_argument("--decoder-normalize-before", default=False, type=strtobool,
                           help="Whether to apply layer norm before decoder block")
        group.add_argument("--encoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer's input and output in encoder")
        group.add_argument("--decoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer's input and output in decoder")
        group.add_argument("--reduction-factor", default=1, type=int,
                           help="Reduction factor")
        group.add_argument("--spk-embed-dim", default=None, type=int,
                           help="Number of speaker embedding dimensions")
        group.add_argument("--spk-embed-integration-type", type=str, default="add",
                           choices=["add", "concat"],
                           help="How to integrate speaker embedding")
        # training related
        group.add_argument("--core-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help="How to initialize core parameters")
        group.add_argument("--initial-encoder-alpha", type=float, default=1.0,
                           help="Initial alpha value in encoder's ScaledPositionalEncoding")
        group.add_argument("--initial-decoder-alpha", type=float, default=1.0,
                           help="Initial alpha value in decoder's ScaledPositionalEncoding")
        group.add_argument("--core-lr", default=1.0, type=float,
                           help="Initial value of learning rate")
        group.add_argument("--core-warmup-steps", default=4000, type=int,
                           help="Optimizer warmup steps")
        group.add_argument("--core-enc-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for core encoder except for attention")
        group.add_argument("--core-enc-positional-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for core encoder positional encoding")
        group.add_argument("--core-enc-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for core encoder self-attention")
        group.add_argument("--core-dec-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for core decoder except for attention and pos encoding")
        group.add_argument("--core-dec-positional-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for core decoder positional encoding")
        group.add_argument("--core-dec-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for core decoder self-attention")
        group.add_argument("--core-enc-dec-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for core encoder-decoder attention")
        group.add_argument("--eprenet-dropout-rate", default=0.5, type=float,
                           help="Dropout rate in encoder prenet")
        group.add_argument("--dprenet-dropout-rate", default=0.5, type=float,
                           help="Dropout rate in decoder prenet")
        group.add_argument("--postnet-dropout-rate", default=0.5, type=float,
                           help="Dropout rate in postnet")
        # loss related
        group.add_argument("--use-masking", default=True, type=strtobool,
                           help="Whether to use masking in calculation of loss")
        group.add_argument("--loss-type", default="L1", choices=["L1", "L2", "L1+L2"],
                           help="How to calc loss")
        group.add_argument("--bce-pos-weight", default=5.0, type=float,
                           help="Positive sample weight in BCE calculation (only for use-masking=True)")
        group.add_argument("--use-guided-attn-loss", default=False, type=strtobool,
                           help="Whether to use guided attention loss")
        group.add_argument("--guided-attn-loss-sigma", default=0.4, type=float,
                           help="Sigma in guided attention loss")
        group.add_argument("--guided-attn-loss-lambda", default=1.0, type=float,
                           help="Lambda in guided attention loss")
        group.add_argument("--num-heads-applied-guided-attn", default=2, type=int,
                           help="Number of heads in each layer to be applied guided attention loss"
                                "if set -1, all of the heads will be applied.")
        group.add_argument("--num-layers-applied-guided-attn", default=2, type=int,
                           help="Number of layers to be applied guided attention loss"
                                "if set -1, all of the layers will be applied.")
        group.add_argument("--modules-applied-guided-attn", type=str, nargs="+",
                           default=["encoder-decoder"],
                           help="Module name list to be applied guided attention loss")
        return parser

    @property
    def attention_plot_class(self):
        """Return plot class for attention weight plot."""
        return TTSPlot

    def __init__(self, idim, odim, args=None):
        """Initialize TTS-Transformer module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            args (Namespace, optional):
                - embed_dim (int): Dimension of character embedding.
                - eprenet_conv_layers (int): Number of encoder prenet convolution layers.
                - eprenet_conv_chans (int): Number of encoder prenet convolution channels.
                - eprenet_conv_filts (int): Filter size of encoder prenet convolution.
                - dprenet_layers (int): Number of decoder prenet layers.
                - dprenet_units (int): Number of decoder prenet hidden units.
                - elayers (int): Number of encoder layers.
                - eunits (int): Number of encoder hidden units.
                - adim (int): Number of attention transformation dimensions.
                - aheads (int): Number of heads for multi head attention.
                - dlayers (int): Number of decoder layers.
                - dunits (int): Number of decoder hidden units.
                - postnet_layers (int): Number of postnet layers.
                - postnet_chans (int): Number of postnet channels.
                - postnet_filts (int): Filter size of postnet.
                - use_scaled_pos_enc (bool): Whether to use trainable scaled positional encoding.
                - use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
                - encoder_normalize_before (bool): Whether to perform layer normalization before encoder block.
                - decoder_normalize_before (bool): Whether to perform layer normalization before decoder block.
                - encoder_concat_after (bool): Whether to concatenate attention layer's input and output in encoder.
                - decoder_concat_after (bool): Whether to concatenate attention layer's input and output in decoder.
                - reduction_factor (int): Reduction factor.
                - spk_embed_dim (int): Number of speaker embedding dimenstions.
                - spk_embed_integration_type: How to integrate speaker embedding.
                - transformer_init (float): How to initialize core parameters.
                - transformer_lr (float): Initial value of learning rate.
                - transformer_warmup_steps (int): Optimizer warmup steps.
                - transformer_enc_dropout_rate (float): Dropout rate in encoder except attention & positional encoding.
                - transformer_enc_positional_dropout_rate (float): Dropout rate after encoder positional encoding.
                - transformer_enc_attn_dropout_rate (float): Dropout rate in encoder self-attention module.
                - transformer_dec_dropout_rate (float): Dropout rate in decoder except attention & positional encoding.
                - transformer_dec_positional_dropout_rate (float): Dropout rate after decoder positional encoding.
                - transformer_dec_attn_dropout_rate (float): Dropout rate in deocoder self-attention module.
                - transformer_enc_dec_attn_dropout_rate (float): Dropout rate in encoder-deocoder attention module.
                - eprenet_dropout_rate (float): Dropout rate in encoder prenet.
                - dprenet_dropout_rate (float): Dropout rate in decoder prenet.
                - postnet_dropout_rate (float): Dropout rate in postnet.
                - use_masking (bool): Whether to use masking in calculation of loss.
                - bce_pos_weight (float): Positive sample weight in bce calculation (only for use_masking=true).
                - loss_type (str): How to calculate loss.
                - use_guided_attn_loss (bool): Whether to use guided attention loss.
                - num_heads_applied_guided_attn (int): Number of heads in each layer to apply guided attention loss.
                - num_layers_applied_guided_attn (int): Number of layers to apply guided attention loss.
                - modules_applied_guided_attn (list): List of module names to apply guided attention loss.
                - guided-attn-loss-sigma (float) Sigma in guided attention loss.
                - guided-attn-loss-lambda (float): Lambda in guided attention loss.

        """
        # initialize base classes
        torch.nn.Module.__init__(self)

        # fill missing arguments
        args = fill_missing_args(args, self.add_arguments)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.use_scaled_pos_enc = hp.use_scaled_pos_enc
        self.reduction_factor = hp.reduction_factor
        self.loss_type = "L1"
        self.use_guided_attn_loss = True
        if self.use_guided_attn_loss:
            if hp.num_layers_applied_guided_attn == -1:
                self.num_layers_applied_guided_attn = hp.elayers
            else:
                self.num_layers_applied_guided_attn = hp.num_layers_applied_guided_attn
            if hp.num_heads_applied_guided_attn == -1:
                self.num_heads_applied_guided_attn = hp.aheads
            else:
                self.num_heads_applied_guided_attn = hp.num_heads_applied_guided_attn
            self.modules_applied_guided_attn = hp.modules_applied_guided_attn

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding

        # define core encoder
        # if args.eprenet_conv_layers != 0:
        #     # encoder prenet
        #     encoder_input_layer = torch.nn.Sequential(
        #         EncoderPrenet(
        #             idim=idim,
        #             embed_dim=args.embed_dim,
        #             elayers=0,
        #             econv_layers=args.eprenet_conv_layers,
        #             econv_chans=args.eprenet_conv_chans,
        #             econv_filts=args.eprenet_conv_filts,
        #             use_batch_norm=args.use_batch_norm,
        #             dropout_rate=args.eprenet_dropout_rate,
        #             padding_idx=padding_idx
        #         ),
        #         torch.nn.Linear(args.eprenet_conv_chans, args.adim)
        #     )
        # else:
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
            input_layer=encoder_input_layer,
            dropout_rate=hp.transformer_enc_dropout_rate,
            positional_dropout_rate=hp.transformer_enc_positional_dropout_rate,
            attention_dropout_rate=hp.transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=hp.encoder_normalize_before,
            concat_after=hp.encoder_concat_after
        )



        # define core decoder
        if hp.dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                DecoderPrenet(
                    idim=odim,
                    n_layers=hp.dprenet_layers,
                    n_units=hp.dprenet_units,
                    dropout_rate=hp.dprenet_dropout_rate
                ),
                torch.nn.Linear(hp.dprenet_units, hp.adim)
            )
        else:
            decoder_input_layer = "linear"
        self.decoder = Decoder(
            odim=-1,
            attention_dim=hp.adim,
            attention_heads=hp.aheads,
            linear_units=hp.dunits,
            dropout_rate=hp.transformer_dec_dropout_rate,
            positional_dropout_rate=hp.transformer_dec_positional_dropout_rate,
            self_attention_dropout_rate=hp.transformer_dec_attn_dropout_rate,
            src_attention_dropout_rate=hp.transformer_enc_dec_attn_dropout_rate,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=pos_enc_class,
            normalize_before=hp.decoder_normalize_before,
            concat_after=hp.decoder_concat_after
        )

        # define final projection
        self.feat_out = torch.nn.Linear(hp.adim, odim * hp.reduction_factor)
        self.prob_out = torch.nn.Linear(hp.adim, hp.reduction_factor)

        # define postnet
        self.postnet = None if hp.postnet_layers == 0 else Postnet(
            idim=idim,
            odim=odim,
            n_layers=hp.postnet_layers,
            n_chans=hp.postnet_chans,
            n_filts=hp.postnet_filts,
            use_batch_norm=hp.use_batch_norm,
            dropout_rate=hp.postnet_dropout_rate
        )

        # define loss function
        self.criterion = TransformerLoss(use_masking=hp.use_masking,
                                         bce_pos_weight=hp.bce_pos_weight)
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(
                sigma=0.4,
                alpha=1.0,
            )

        # initialize parameters
        self._reset_parameters(init_type=hp.transformer_init,
                               init_enc_alpha=hp.initial_encoder_alpha,
                               init_dec_alpha=hp.initial_decoder_alpha)

    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def _add_first_frame_and_remove_last_frame(self, ys):
        ys_in = torch.cat([ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1)
        return ys_in

    def forward(self, xs, ilens, ys, labels, olens, spembs=None, *args, **kwargs):
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
        max_ilen = max(ilens)
        max_olen = max(olens)
        if max_ilen != xs.shape[1]:
            xs = xs[:, :max_ilen]
        if max_olen != ys.shape[1]:
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]

        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)



        # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
        if self.reduction_factor > 1:
            ys_in = ys[:, self.reduction_factor - 1::self.reduction_factor]
            olens_in = olens.new([olen // self.reduction_factor for olen in olens])
        else:
            ys_in, olens_in = ys, olens

        # add first zero frame and remove last frame for auto-regressive
        ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

        # forward decoder
        y_masks = self._target_mask(olens_in)
        xy_masks = self._source_to_target_mask(ilens, olens_in)
        zs, _ = self.decoder(ys_in, y_masks, hs, xy_masks)
        # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, Lmax//r, r) -> (B, Lmax//r * r)
        logits = self.prob_out(zs).view(zs.size(0), -1)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]
            labels[:, -1] = 1.0  # make sure at least one frame has 1

        # caluculate loss values
        l1_loss, l2_loss, bce_loss = self.criterion(
            after_outs, before_outs, logits, ys, labels, olens)
        if self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = l2_loss + bce_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss + bce_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"l2_loss": l2_loss.item()},
            {"bce_loss": bce_loss.item()},
            {"loss": loss.item()},
        ]

        # calculate guided attention loss
        if self.use_guided_attn_loss:
            # calculate for encoder
            if "encoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(reversed(range(len(self.encoder.encoders)))):
                    att_ws += [self.encoder.encoders[layer_idx].self_attn.attn[:, :self.num_heads_applied_guided_attn]]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_in, T_in)
                enc_attn_loss = self.attn_criterion(att_ws, ilens, ilens)
                loss = loss + enc_attn_loss
                report_keys += [{"enc_attn_loss": enc_attn_loss.item()}]
            # calculate for decoder
            if "decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(reversed(range(len(self.decoder.decoders)))):
                    att_ws += [self.decoder.decoders[layer_idx].self_attn.attn[:, :self.num_heads_applied_guided_attn]]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_out)
                dec_attn_loss = self.attn_criterion(att_ws, olens_in, olens_in)
                loss = loss + dec_attn_loss
                report_keys += [{"dec_attn_loss": dec_attn_loss.item()}]
            # calculate for encoder-decoder
            if "encoder_decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(reversed(range(len(self.decoder.decoders)))):
                    att_ws += [self.decoder.decoders[layer_idx].src_attn.attn[:, :self.num_heads_applied_guided_attn]]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_in)
                enc_dec_attn_loss = self.attn_criterion(att_ws, ilens, olens_in)
                loss = loss + enc_dec_attn_loss
                report_keys += [{"enc_dec_attn_loss": enc_dec_attn_loss.item()}]

        # report extra information
        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed[-1].alpha.data.item()},
            ]
        #self.reporter.report(report_keys)

        return loss, report_keys

    def inference(self, x, inference_args, *args, **kwargs):
        """Generate the sequence of features given the sequences of characters.

        Args:
            x (Tensor): Input sequence of characters (T,).
            inference_args (Namespace):
                - threshold (float): Threshold in inference.
                - minlenratio (float): Minimum length ratio in inference.
                - maxlenratio (float): Maximum length ratio in inference.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Encoder-decoder (source) attention weights (#layers, #heads, L, T).

        """
        # get options
        threshold = inference_args.threshold
        minlenratio = inference_args.minlenratio
        maxlenratio = inference_args.maxlenratio

        # forward encoder
        xs = x.unsqueeze(0)
        hs, _ = self.encoder(xs, None)



        # set limits of length
        maxlen = int(hs.size(1) * maxlenratio / self.reduction_factor)
        minlen = int(hs.size(1) * minlenratio / self.reduction_factor)

        # initialize
        idx = 0
        ys = hs.new_zeros(1, 1, self.odim)
        outs, probs = [], []

        # forward decoder step-by-step
        while True:
            # update index
            idx += 1

            # calculate output and stop prob at idx-th step
            y_masks = subsequent_mask(idx).unsqueeze(0).to(x.device)
            z = self.decoder.recognize(ys, y_masks, hs)  # (B, adim)
            outs += [self.feat_out(z).view(self.reduction_factor, self.odim)]  # [(r, odim), ...]
            probs += [torch.sigmoid(self.prob_out(z))[0]]  # [(r), ...]

            # update next inputs
            ys = torch.cat((ys, outs[-1][-1].view(1, 1, self.odim)), dim=1)  # (1, idx + 1, odim)

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = torch.cat(outs, dim=0).unsqueeze(0).transpose(1, 2)  # (L, odim) -> (1, L, odim) -> (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                break

        # get attention weights
        att_ws = []
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention) and "src" in name:
                att_ws += [m.attn]
        att_ws = torch.cat(att_ws, dim=0)

        return outs, probs, att_ws

    def calculate_all_attentions(self, xs, ilens, ys, olens,
                                  skip_output=False, keep_tensor=False, *args, **kwargs):
        """Calculate all of the attention weights.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).
            skip_output (bool, optional): Whether to skip calculate the final output.
            keep_tensor (bool, optional): Whether to keep original tensor.

        Returns:
            dict: Dict of attention weights and outputs.

        """
        with torch.no_grad():
            # forward encoder
            x_masks = self._source_mask(ilens)
            hs, _ = self.encoder(xs, x_masks)


            # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
            if self.reduction_factor > 1:
                ys_in = ys[:, self.reduction_factor - 1::self.reduction_factor]
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                ys_in, olens_in = ys, olens

            # add first zero frame and remove last frame for auto-regressive
            ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

            # forward decoder
            y_masks = self._target_mask(olens_in)
            xy_masks = self._source_to_target_mask(ilens, olens_in)
            zs, _ = self.decoder(ys_in, y_masks, hs, xy_masks)

            # calculate final outputs
            if not skip_output:
                before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
                if self.postnet is None:
                    after_outs = before_outs
                else:
                    after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        # modifiy mod part of output lengths due to reduction factor > 1
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])

        # store into dict
        att_ws_dict = dict()
        if keep_tensor:
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention):
                    att_ws_dict[name] = m.attn
            if not skip_output:
                att_ws_dict["before_postnet_fbank"] = before_outs
                att_ws_dict["after_postnet_fbank"] = after_outs
        else:
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention):
                    attn = m.attn.cpu().numpy()
                    if "encoder" in name:
                        attn = [a[:, :l, :l] for a, l in zip(attn, ilens.tolist())]
                    elif "decoder" in name:
                        if "src" in name:
                            attn = [a[:, :ol, :il] for a, il, ol in zip(attn, ilens.tolist(), olens_in.tolist())]
                        elif "self" in name:
                            attn = [a[:, :l, :l] for a, l in zip(attn, olens_in.tolist())]
                        else:
                            logging.warning("unknown attention module: " + name)
                    else:
                        logging.warning("unknown attention module: " + name)
                    att_ws_dict[name] = attn
            if not skip_output:
                before_outs = before_outs.cpu().numpy()
                after_outs = after_outs.cpu().numpy()
                att_ws_dict["before_postnet_fbank"] = [m[:l].T for m, l in zip(before_outs, olens.tolist())]
                att_ws_dict["after_postnet_fbank"] = [m[:l].T for m, l in zip(after_outs, olens.tolist())]

        return att_ws_dict
    #
    # def _integrate_with_spk_embed(self, hs, spembs):
    #     """Integrate speaker embedding with hidden states.
    #
    #     Args:
    #         hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
    #         spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).
    #
    #     Returns:
    #         Tensor: Batch of integrated hidden state sequences (B, Tmax, adim)
    #
    #     """
    #     if self.spk_embed_integration_type == "add":
    #         # apply projection and then add to hidden states
    #         spembs = self.projection(F.normalize(spembs))
    #         hs = hs + spembs.unsqueeze(1)
    #     elif self.spk_embed_integration_type == "concat":
    #         # concat hidden states with spk embeds and then apply projection
    #         spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
    #         hs = self.projection(torch.cat([hs, spembs], dim=-1))
    #     else:
    #         raise NotImplementedError("support only add or concat.")
    #
    #     return hs

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

    def _target_mask(self, olens):
        """Make masks for masked self-attention.

        Examples:
            >>> olens = [5, 3]
            >>> self._target_mask(olens)
            tensor([[[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]],
                    [[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)

        """
        #print("O lens:",olens)
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        # y_masks = torch.mul(y_masks,1)
        # print("y masks ", y_masks)
        # print("s masks ", s_masks)
        # print("y mask",y_masks.size())
        # print("s mask", s_masks.size())
        # print("y mask", y_masks.unsqueeze(-2).size())
        # print("y mask", y_masks.unsqueeze(-1).size())
        # print("S marks : {} - {}".format(y_masks.unsqueeze(-2) & s_masks & y_masks.unsqueeze(-1),type(y_masks.unsqueeze(-2) & s_masks & y_masks.unsqueeze(-1))))
        return y_masks.unsqueeze(-2) & s_masks & y_masks.unsqueeze(-1)

    def _source_to_target_mask(self, ilens, olens):
        """Make masks for encoder-decoder attention.

        Examples:
            >>> ilens = [4, 2]
            >>> olens = [5, 3]
            >>> self._source_to_target_mask(ilens)
            tensor([[[1, 1, 1, 1],
                     [1, 1, 1, 1],
                     [1, 1, 1, 1],
                     [1, 1, 1, 1],
                     [1, 1, 1, 1]],
                    [[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & y_masks.unsqueeze(-1)

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training. keys should match what `chainer.reporter` reports.

        If you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ["loss", "l1_loss", "l2_loss", "bce_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]
        if self.use_guided_attn_loss:
            if "encoder" in self.modules_applied_guided_attn:
                plot_keys += ["enc_attn_loss"]
            if "decoder" in self.modules_applied_guided_attn:
                plot_keys += ["dec_attn_loss"]
            if "encoder-decoder" in self.modules_applied_guided_attn:
                plot_keys += ["enc_dec_attn_loss"]

        return plot_keys
