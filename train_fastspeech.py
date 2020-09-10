import fastspeech
from tensorboardX import SummaryWriter
import torch
from dataset import dataloader as loader
import logging
import math
import os
import sys
import numpy as np
import configargparse
import random
import tqdm
import time
from evaluation import evaluate
from utils.plot import generate_audio, plot_spectrogram_to_numpy
from core.optimizer import get_std_opt
from utils.util import read_wav_np
from dataset.texts import valid_symbols
from utils.util import get_commit_hash
from utils.hparams import HParam

BATCH_COUNT_CHOICES = ["auto", "seq", "bin", "frame"]
BATCH_SORT_KEY_CHOICES = ["input", "output", "shuffle"]


def train(args, hp, hp_str, logger, vocoder):
    os.makedirs(os.path.join(hp.train.chkpt_dir, args.name), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, args.name), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, args.name, "assets"), exist_ok=True)
    device = torch.device("cuda" if hp.train.ngpu > 0 else "cpu")

    dataloader = loader.get_tts_dataset(hp.data.data_dir, hp.train.batch_size, hp)
    validloader = loader.get_tts_dataset(hp.data.data_dir, 1, hp, True)

    idim = len(valid_symbols)
    odim = hp.audio.num_mels
    model = fastspeech.FeedForwardTransformer(idim, odim, hp)
    # set torch device
    model = model.to(device)
    print("Model is loaded ...")
    githash = get_commit_hash()
    if args.checkpoint_path is not None:
        if os.path.exists(args.checkpoint_path):
            logger.info("Resuming from checkpoint: %s" % args.checkpoint_path)
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            optimizer = get_std_opt(
                model,
                hp.model.adim,
                hp.model.transformer_warmup_steps,
                hp.model.transformer_lr,
            )
            optimizer.load_state_dict(checkpoint["optim"])
            global_step = checkpoint["step"]

            if hp_str != checkpoint["hp_str"]:
                logger.warning(
                    "New hparams is different from checkpoint. Will use new."
                )

            if githash != checkpoint["githash"]:
                logger.warning("Code might be different: git hash is different.")
                logger.warning("%s -> %s" % (checkpoint["githash"], githash))

        else:
            print("Checkpoint does not exixts")
            global_step = 0
            return None
    else:
        print("New Training")
        global_step = 0
        optimizer = get_std_opt(
            model,
            hp.model.adim,
            hp.model.transformer_warmup_steps,
            hp.model.transformer_lr,
        )

    print("Batch Size :", hp.train.batch_size)

    num_params(model)

    os.makedirs(os.path.join(hp.train.log_dir, args.name), exist_ok=True)
    writer = SummaryWriter(os.path.join(hp.train.log_dir, args.name))
    model.train()
    forward_count = 0
    # print(model)
    for epoch in range(hp.train.epochs):
        start = time.time()
        running_loss = 0
        j = 0

        pbar = tqdm.tqdm(dataloader, desc="Loading train data")
        for data in pbar:
            global_step += 1
            x, input_length, y, _, out_length, _, dur, e, p = data
            # x : [batch , num_char], input_length : [batch], y : [batch, T_in, num_mel]
            #             # stop_token : [batch, T_in], out_length : [batch]

            loss, report_dict = model(
                x.cuda(),
                input_length.cuda(),
                y.cuda(),
                out_length.cuda(),
                dur.cuda(),
                e.cuda(),
                p.cuda(),
            )
            loss = loss.mean() / hp.train.accum_grad
            running_loss += loss.item()

            loss.backward()

            # update parameters
            forward_count += 1
            j = j + 1
            if forward_count != hp.train.accum_grad:
                continue
            forward_count = 0
            step = global_step

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hp.train.grad_clip
            )
            logging.debug("grad norm={}".format(grad_norm))
            if math.isnan(grad_norm):
                logging.warning("grad norm is nan. Do not update model.")
            else:
                optimizer.step()
            optimizer.zero_grad()

            if step % hp.train.summary_interval == 0:
                pbar.set_description(
                    "Average Loss %.04f Loss %.04f | step %d"
                    % (running_loss / j, loss.item(), step)
                )

                for r in report_dict:
                    for k, v in r.items():
                        if k is not None and v is not None:
                            if "cupy" in str(type(v)):
                                v = v.get()
                            if "cupy" in str(type(k)):
                                k = k.get()
                            writer.add_scalar("main/{}".format(k), v, step)

            if step % hp.train.validation_step == 0:

                for valid in validloader:
                    x_, input_length_, y_, _, out_length_, ids_, dur_, e_, p_ = valid
                    model.eval()
                    with torch.no_grad():
                        loss_, report_dict_ = model(
                            x_.cuda(),
                            input_length_.cuda(),
                            y_.cuda(),
                            out_length_.cuda(),
                            dur_.cuda(),
                            e_.cuda(),
                            p_.cuda(),
                        )

                        mels_ = model.inference(x_[-1].cuda())  # [T, num_mel]

                    model.train()
                    for r in report_dict_:
                        for k, v in r.items():
                            if k is not None and v is not None:
                                if "cupy" in str(type(v)):
                                    v = v.get()
                                if "cupy" in str(type(k)):
                                    k = k.get()
                                writer.add_scalar("validation/{}".format(k), v, step)

                    mels_ = mels_.T  # Out: [num_mels, T]
                    writer.add_image(
                        "melspectrogram_target_{}".format(ids_[-1]),
                        plot_spectrogram_to_numpy(
                            y_[-1].T.data.cpu().numpy()[:, : out_length_[-1]]
                        ),
                        step,
                        dataformats="HWC",
                    )
                    writer.add_image(
                        "melspectrogram_prediction_{}".format(ids_[-1]),
                        plot_spectrogram_to_numpy(mels_.data.cpu().numpy()),
                        step,
                        dataformats="HWC",
                    )

                    # print(mels.unsqueeze(0).shape)

                    audio = generate_audio(
                        mels_.unsqueeze(0), vocoder
                    )  # selecting the last data point to match mel generated above
                    audio = audio.cpu().float().numpy()
                    audio = audio / (
                        audio.max() - audio.min()
                    )  # get values between -1 and 1

                    writer.add_audio(
                        tag="generated_audio_{}".format(ids_[-1]),
                        snd_tensor=torch.Tensor(audio),
                        global_step=step,
                        sample_rate=hp.audio.sample_rate,
                    )

                    _, target = read_wav_np(
                        hp.data.wav_dir + f"{ids_[-1]}.wav",
                        sample_rate=hp.audio.sample_rate,
                    )

                    writer.add_audio(
                        tag=" target_audio_{}".format(ids_[-1]),
                        snd_tensor=torch.Tensor(target),
                        global_step=step,
                        sample_rate=hp.audio.sample_rate,
                    )

                ##
            if step % hp.train.save_interval == 0:
                avg_p, avg_e, avg_d = evaluate(hp, validloader, model)
                writer.add_scalar("evaluation/Pitch Loss", avg_p, step)
                writer.add_scalar("evaluation/Energy Loss", avg_e, step)
                writer.add_scalar("evaluation/Dur Loss", avg_d, step)
                save_path = os.path.join(
                    hp.train.chkpt_dir,
                    args.name,
                    "{}_fastspeech_{}_{}k_steps.pyt".format(
                        args.name, githash, step // 1000
                    ),
                )

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "step": step,
                        "hp_str": hp_str,
                        "githash": githash,
                    },
                    save_path,
                )
                logger.info("Saved checkpoint to: %s" % save_path)
        print(
            "Time taken for epoch {} is {} sec\n".format(
                epoch + 1, int(time.time() - start)
            )
        )


def num_params(model, print_out=True):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print("Trainable Parameters: %.3fM" % parameters)


def create_gta(args, hp, hp_str, logger):
    os.makedirs(os.path.join(hp.data.data_dir, "gta"), exist_ok=True)
    device = torch.device("cuda" if hp.train.ngpu > 0 else "cpu")

    dataloader = loader.get_tts_dataset(hp.data.data_dir, 1)
    validloader = loader.get_tts_dataset(hp.data.data_dir, 1, True)
    global_step = 0
    idim = len(valid_symbols)
    odim = hp.audio.num_mels
    model = fastspeech.FeedForwardTransformer(idim, odim, args)
    # set torch device
    if os.path.exists(args.checkpoint_path):
        print("\nSynthesis GTA Session...\n")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
    else:
        print("Checkpoint not exixts")
        return None
    model.eval()
    model = model.to(device)
    print("Model is loaded ...")
    print("Batch Size :", hp.train.batch_size)
    num_params(model)
    onlyValidation = False
    if not onlyValidation:
        pbar = tqdm.tqdm(dataloader, desc="Loading train data")
        for data in pbar:
            # start_b = time.time()
            global_step += 1
            x, input_length, y, _, out_length, ids = data
            with torch.no_grad():
                _, gta, _, _, _ = model._forward(
                    x.cuda(), input_length.cuda(), y.cuda(), out_length.cuda()
                )
                # gta = model._forward(x.cuda(), input_length.cuda(), is_inference=False)
            gta = gta.cpu().numpy()

            for j in range(len(ids)):
                mel = gta[j]
                mel = mel.T
                mel = mel[:, : out_length[j]]
                mel = (mel + 4) / 8
                id = ids[j]
                np.save(
                    "{}/{}.npy".format(os.path.join(hp.data.data_dir, "gta"), id),
                    mel,
                    allow_pickle=False,
                )

    pbar = tqdm.tqdm(validloader, desc="Loading Valid data")
    for data in pbar:
        # start_b = time.time()
        global_step += 1
        x, input_length, y, _, out_length, ids = data
        with torch.no_grad():
            gta, _, _ = model._forward(
                x.cuda(), input_length.cuda(), y.cuda(), out_length.cuda()
            )
            # gta = model._forward(x.cuda(), input_length.cuda(), is_inference=True)
        gta = gta.cpu().numpy()

        for j in range(len(ids)):
            print("Actual mel specs : {} = {}".format(ids[j], y[j].shape))
            print("Out length:", out_length[j])
            print("GTA size: {} = {}".format(ids[j], gta[j].shape))
            mel = gta[j]
            mel = mel.T
            mel = mel[:, : out_length[j]]
            mel = (mel + 4) / 8
            print("Mel size: {} = {}".format(ids[j], mel.shape))
            id = ids[j]
            np.save(
                "{}/{}.npy".format(os.path.join(hp.data.data_dir, "gta"), id),
                mel,
                allow_pickle=False,
            )


# define function for plot prob and att_ws
def _plot_and_save(array, figname, figsize=(6, 4), dpi=150):
    import matplotlib.pyplot as plt

    shape = array.shape
    if len(shape) == 1:
        # for eos probability
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(array)
        plt.xlabel("Frame")
        plt.ylabel("Probability")
        plt.ylim([0, 1])
    elif len(shape) == 2:
        # for tacotron 2 attention weights, whose shape is (out_length, in_length)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(array, aspect="auto")
        plt.xlabel("Input")
        plt.ylabel("Output")
    elif len(shape) == 4:
        # for transformer attention weights, whose shape is (#leyers, #heads, out_length, in_length)
        fig = plt.figure(
            figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi
        )
        for idx1, xs in enumerate(array):
            for idx2, x in enumerate(xs, 1):
                plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                plt.imshow(x.cpu().detach().numpy(), aspect="auto")
                plt.xlabel("Input")
                plt.ylabel("Output")
    else:
        raise NotImplementedError("Support only from 1D to 4D array.")
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(figname)):
        # NOTE: exist_ok = True is needed for parallel process decoding
        os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    plt.close()
    return fig


# NOTE: you need this func to generate our sphinx doc
def get_parser():
    """Get parser of training arguments."""
    parser = configargparse.ArgumentParser(
        description="Train a new text-to-speech (TTS) model on one CPU, one or multiple GPUs",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    parser.add_argument(
        "-p",
        "--checkpoint_path",
        type=str,
        default=None,
        help="path of checkpoint pt file to resume training",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="name of the model for logging, saving checkpoint",
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")

    return parser


def main(cmd_args):
    """Run training."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    args = parser.parse_args(cmd_args)

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    # logging info
    os.makedirs(hp.train.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(hp.train.log_dir, "%s-%d.log" % (args.name, time.time()))
            ),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    ngpu = hp.train.ngpu
    logger.info(f"ngpu: {ngpu}")

    # set random seed
    logger.info("random seed = %d" % hp.train.seed)
    random.seed(hp.train.seed)
    np.random.seed(hp.train.seed)

    vocoder = torch.hub.load(
        "seungwonpark/melgan", "melgan"
    )  # load the vocoder for validation

    if hp.train.GTA:
        create_gta(args, hp, hp_str, logger)
    else:
        train(args, hp, hp_str, logger, vocoder)


if __name__ == "__main__":
    main(sys.argv[1:])
