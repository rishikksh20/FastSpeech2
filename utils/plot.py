import matplotlib
import numpy as np
import torch
matplotlib.use('Agg')
import copy
# matplotlib related
import os

class PlotAttentionReport():
    """Plot attention reporter.

    Args:
        att_vis_fn (espnet.nets.*_backend.e2e_asr.E2E.calculate_all_attentions):
            Function of attention visualization.
        data (list[tuple(str, dict[str, list[Any]])]): List json utt key items.
        outdir (str): Directory to save figures.
        converter (espnet.asr.*_backend.asr.CustomConverter): Function to convert data.
        device (int | torch.device): Device.
        reverse (bool): If True, input and output length are reversed.
        ikey (str): Key to access input (for ASR ikey="input", for MT ikey="output".)
        iaxis (int): Dimension to access input (for ASR iaxis=0, for MT iaxis=1.)
        okey (str): Key to access output (for ASR okey="input", MT okay="output".)

    """

    def __init__(self, outdir, device, reverse=False,
                 ikey="input", iaxis=0, okey="output", oaxis=0):
        #self.att_vis_fn = att_vis_fn
        #self.data = copy.deepcopy(data)
        self.outdir = outdir
        #self.converter = converter
        #self.transform = transform
        self.device = device
        self.reverse = reverse
        self.ikey = ikey
        self.iaxis = iaxis
        self.okey = okey
        self.oaxis = oaxis
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, step, att_ws):
        """Plot and save image file of att_ws matrix."""
        #att_ws = self.get_attention_weights()
        if isinstance(att_ws, list):  # multi-encoder case
            num_encs = len(att_ws) - 1
            # atts
            for i in range(num_encs):
                for idx, att_w in enumerate(att_ws[i]):
                    filename = "%s/%s.ep.{}.att%d.png" % (
                        self.outdir, self.data[idx][0], i + 1)
                    att_w = self.get_attention_weight(idx, att_w)
                    np_filename = "%s/%s.ep.{}.att%d.npy" % (
                        self.outdir, self.data[idx][0], i + 1)
                    np.save(np_filename.format(step), att_w)
                    self._plot_and_save_attention(att_w, filename.format(step))
            # han
            for idx, att_w in enumerate(att_ws[num_encs]):
                filename = "%s/%s.ep.{}.han.png" % (
                    self.outdir, self.data[idx][0])
                att_w = self.get_attention_weight(idx, att_w)
                np_filename = "%s/%s.ep.{}.han.npy" % (
                    self.outdir, self.data[idx][0])
                np.save(np_filename.format(step), att_w)
                self._plot_and_save_attention(att_w, filename.format(step), han_mode=True)
        else:
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.ep.{}.png" % (
                    self.outdir, self.data[idx][0])
                att_w = self.get_attention_weight(idx, att_w)
                np_filename = "%s/%s.ep.{}.npy" % (
                    self.outdir, self.data[idx][0])
                np.save(np_filename.format(step), att_w)
                self._plot_and_save_attention(att_w, filename.format(step))

    def log_attentions(self, logger, step, att_ws):
        """Add image files of att_ws matrix to the tensorboard."""
        #att_ws = self.get_attention_weights()
        if isinstance(att_ws, list):  # multi-encoder case
            num_encs = len(att_ws) - 1
            # atts
            for i in range(num_encs):
                for idx, att_w in enumerate(att_ws[i]):
                    att_w = self.get_attention_weight(idx, att_w)
                    plot = self.draw_attention_plot(att_w)
                    logger.add_figure("%s_att%d" % (self.data[idx][0], i + 1), plot.gcf(), step)
                    plot.clf()
            # han
            for idx, att_w in enumerate(att_ws[num_encs]):
                att_w = self.get_attention_weight(idx, att_w)
                plot = self.draw_han_plot(att_w)
                logger.add_figure("%s_han" % (self.data[idx][0]), plot.gcf(), step)
                plot.clf()
        else:
            for idx, att_w in enumerate(att_ws):
                att_w = self.get_attention_weight(idx, att_w)
                plot = self.draw_attention_plot(att_w)
                logger.add_figure("%s" % (self.data[idx][0]), plot.gcf(), step)
                plot.clf()

    def get_attention_weights(self):
        """Return attention weights.

        Returns:
            numpy.ndarray: attention weights.float. Its shape would be
                differ from backend.
                * pytorch-> 1) multi-head case => (B, H, Lmax, Tmax), 2) other case => (B, Lmax, Tmax).
                * chainer-> (B, Lmax, Tmax)

        """
        batch = self.data
        if isinstance(batch, tuple):
            att_ws = self.att_vis_fn(*batch)
        else:
            att_ws = self.att_vis_fn(**batch)
        return att_ws

    def get_attention_weight(self, idx, att_w):
        """Transform attention matrix with regard to self.reverse."""
        if self.reverse:
            dec_len = int(self.data[idx][1][self.ikey][self.iaxis]['shape'][0])
            enc_len = int(self.data[idx][1][self.okey][self.oaxis]['shape'][0])
        else:
            dec_len = int(self.data[idx][1][self.okey][self.oaxis]['shape'][0])
            enc_len = int(self.data[idx][1][self.ikey][self.iaxis]['shape'][0])
        if len(att_w.shape) == 3:
            att_w = att_w[:, :dec_len, :enc_len]
        else:
            att_w = att_w[:dec_len, :enc_len]
        return att_w

    def draw_attention_plot(self, att_w):
        """Plot the att_w matrix.

        Returns:
            matplotlib.pyplot: pyplot object with attention matrix image.

        """
        import matplotlib.pyplot as plt
        att_w = att_w.astype(np.float32)
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        return plt

    def draw_han_plot(self, att_w):
        """Plot the att_w matrix for hierarchical attention.

        Returns:
            matplotlib.pyplot: pyplot object with attention matrix image.

        """
        import matplotlib.pyplot as plt
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                legends = []
                plt.subplot(1, len(att_w), h)
                for i in range(aw.shape[1]):
                    plt.plot(aw[:, i])
                    legends.append('Att{}'.format(i))
                plt.ylim([0, 1.0])
                plt.xlim([0, aw.shape[0]])
                plt.grid(True)
                plt.ylabel("Attention Weight")
                plt.xlabel("Decoder Index")
                plt.legend(legends)
        else:
            legends = []
            for i in range(att_w.shape[1]):
                plt.plot(att_w[:, i])
                legends.append('Att{}'.format(i))
            plt.ylim([0, 1.0])
            plt.xlim([0, att_w.shape[0]])
            plt.grid(True)
            plt.ylabel("Attention Weight")
            plt.xlabel("Decoder Index")
            plt.legend(legends)
        plt.tight_layout()
        return plt

    def _plot_and_save_attention(self, att_w, filename, han_mode=False):
        if han_mode:
            plt = self.draw_han_plot(att_w)
        else:
            plt = self.draw_attention_plot(att_w)
        plt.savefig(filename)
        plt.close()
