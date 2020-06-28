import logging

import matplotlib.pyplot as plt
import numpy
import matplotlib
import torch
matplotlib.use('Agg')
import logging
import numpy as np
# matplotlib related
from utils.plot import PlotAttentionReport

def plot_mel(mel, filename):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(np.flip(mel, axis=0), interpolation='nearest', aspect='auto')
    savefig(fig, filename)


def _plot_and_save_attention(att_w, filename):
    # dynamically import matplotlib due to not found error
    from matplotlib.ticker import MaxNLocator
    import os
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)
    w, h = plt.figaspect(1.0 / len(att_w))
    fig = plt.Figure(figsize=(w * 2, h * 2))
    axes = fig.subplots(1, len(att_w))
    if len(att_w) == 1:
        axes = [axes]
    for ax, aw in zip(axes, att_w):
        # plt.subplot(1, len(att_w), h)
        ax.imshow(aw.astype(numpy.float32), aspect="auto")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def savefig(plot, filename):
    plot.savefig(filename)
    plt.clf()


def plot_multi_head_attention(input_lengths, output_lengths, attn_dict, outdir, suffix="png", savefn=savefig):
    """Plot multi head attentions

    :param dict data: utts info from json file
    :param dict[str, torch.Tensor] attn_dict: multi head attention dict.
        values should be torch.Tensor (head, input_length, output_length)
    :param str outdir: dir to save fig
    :param str suffix: filename suffix including image type (e.g., png)
    :param savefn: function to save
    """
    for name, att_ws in attn_dict.items():
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.%s.%s" % (
                outdir, str(idx), name, suffix)
            dec_len = int(output_lengths[idx])
            enc_len = int(input_lengths[idx])
            if "encoder" in name:
                att_w = att_w[:, :enc_len, :enc_len]
            elif "decoder" in name:
                if "self" in name:
                    att_w = att_w[:, :dec_len, :dec_len]
                else:
                    att_w = att_w[:, :dec_len, :enc_len]
            else:
                logging.warning("unknown name for shaping attention")
            fig = _plot_and_save_attention(att_w, filename)
            savefn(fig, filename)


class PlotAttentionReport(PlotAttentionReport):
    def plotfn(self, *args, **kwargs):
        plot_multi_head_attention(*args, **kwargs)

    def __call__(self, step, input_lengths, output_lengths, att_ws):
        #attn_dict = self.get_attention_weights()
        suffix = "ep.{}.png".format(step)
        self.plotfn(input_lengths, output_lengths, att_ws, self.outdir, suffix, savefig)

    #def get_attention_weights(self):
    #    batch = self.converter([self.transform(self.data)], self.device)
    #    if isinstance(batch, tuple):
    #        att_ws = self.att_vis_fn(*batch)
    #    elif isinstance(batch, dict):
    #        att_ws = self.att_vis_fn(**batch)
    #     return att_ws

    def log_attentions(self, logger, step, input_lengths, output_lengths, att_ws):
        def log_fig(plot, filename):
            from os.path import basename
            logger.add_figure(basename(filename), plot, step)
            plt.clf()

        #att_ws = self.get_attention_weights()
        self.plotfn(input_lengths, output_lengths, att_ws, self.outdir, "", log_fig)
