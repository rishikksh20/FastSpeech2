import fastspeech
from tensorboardX import SummaryWriter
import torch
import hparams as hp
from dataset import dataloader as loader
import time
import logging
import math
import os
import configargparse
import random
import numpy as np
from utils.cli_utils import strtobool
import sys
import tqdm
from core.optimizer import get_std_opt

BATCH_COUNT_CHOICES = ["auto", "seq", "bin", "frame"]
BATCH_SORT_KEY_CHOICES = ["input", "output", "shuffle"]


def train(args):
    os.makedirs(hp.chkpt_dir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'img'), exist_ok=True)
    device = torch.device("cuda" if hp.ngpu > 0 else "cpu")

    dataloader = loader.get_tts_dataset(hp.data_dir, hp.batch_size)
    validloader = loader.get_tts_dataset(hp.data_dir, 5, True)
    global_step = 0
    idim = hp.symbol_len
    odim = hp.num_mels
    model = fastspeech.FeedForwardTransformer(idim, odim)
    # set torch device
    if args.resume is not None:
        if os.path.exists(args.resume):
            print('\nSynthesis Session...\n')
            model.load_state_dict(torch.load(args.resume), strict=False)
            optimizer = get_std_opt(model, hp.adim, hp.transformer_warmup_steps, hp.transformer_lr)
            optimizer.load_state_dict(torch.load(args.resume.replace("model", "optim")))
            global_step = optimizer._step
        else:
            print("Checkpoint not exixts")
            return None
    else:
        optimizer = get_std_opt(model, hp.adim, hp.transformer_warmup_steps, hp.transformer_lr)
    model = model.to(device)
    print("Model is loaded ...")
    print("Batch Size :",hp.batch_size)
    num_params(model)
    # Setup an optimizer
    # if args.opt == 'adam':
    #     optimizer = torch.optim.Adam(
    #         model.parameters(), args.lr, eps=args.eps,
    #         weight_decay=args.weight_decay)
    # elif args.opt == 'noam':

    # else:
    #     raise NotImplementedError("unknown optimizer: " + args.opt)

    writer = SummaryWriter(hp.log_dir)
    model.train()
    forward_count = 0
    print(model)
    for epoch in range(hp.epochs):
        start = time.time()
        # dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer,
        #                         drop_last=True, num_workers=16)
        running_loss=0
        j=0

        pbar = tqdm.tqdm(dataloader, desc='Loading train data')
        for data in pbar:
            #start_b = time.time()
            global_step += 1
            x, input_length, y, _, out_length, _, dur, e, p = data
            # x : [batch , num_char], input_length : [batch], y : [batch, T_in, num_mel]
            #             # stop_token : [batch, T_in], out_length : [batch]
            #print("x : ", x.size())
            #print("in : ", input_length.size())
            #print("y : ", y.size())
            #print("stop : ", stop_token.size())
            #print("out : ", out_length.size())
            #print("out length: ", out_length)
            loss, report_dict = model(x.cuda(), input_length.cuda(), y.cuda(), out_length.cuda(), dur.cuda(), e.cuda(), p.cuda())
            loss = loss.mean()/hp.accum_grad
            running_loss += loss.item()

            loss.backward()

            # if hp.tts_clip_grad_norm:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
            #
            # optimizer.step()
            #
            # update parameters
            forward_count += 1
            j = j + 1
            if forward_count != hp.accum_grad:
                continue
            forward_count = 0
            step = global_step
            #

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            logging.debug('grad norm={}'.format(grad_norm))
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.step()
            optimizer.zero_grad()

            if step % hp.summary_interval == 0:
                #torch.cuda.empty_cache()
                            
                pbar.set_description(
                    "Average Loss %.04f Loss %.04f | step %d" % (running_loss / j, loss.item(), step))

                print("Losses :")
                for r in report_dict:
                    for k, v in r.items():
                        if k == 'l1_loss':
                            print("\nL1 loss :", v)    
                        if k == 'duration_loss':
                            print("\nD loss :", v)
                        if k == 'pitch_loss':
                            print("\nP loss :", v)
                        if k == 'energy_loss':
                            print("\nE loss :", v)
                        if k is not None and v is not None:
                            if 'cupy' in str(type(v)):
                                v = v.get()
                            if 'cupy' in str(type(k)):
                                k = k.get()
                            writer.add_scalar("main/{}".format(k), v, step)

            if step % hp.validation_step == 0:
                plot_class = model.attention_plot_class
                plot_fn = plot_class(args.outdir + '/att_ws',device)
                for valid in validloader:
                    x_, input_length_, y_, _, out_length_, ids_, dur_, e_, p_ = valid
                    model.eval()
                    with torch.no_grad(): 
                        loss_, report_dict_ = model(x_.cuda(), input_length_.cuda(), y_.cuda(), out_length_.cuda(), dur_.cuda(), e_.cuda(), p_.cuda())
                    att_ws = model.calculate_all_attentions(x_.cuda(), input_length_.cuda(), y_.cuda(), out_length_.cuda(), dur_.cuda(), e_.cuda(), p_.cuda())
                    model.train()
                    print(" Validation Losses :")
                    for r in report_dict_:
                        for k, v in r.items():
                            if k == 'l1_loss':
                                print("\nL1 loss :", v)
                            if k == 'duration_loss':
                                print("\nD loss :", v)
                            if k == 'pitch_loss':
                                print("\nP loss :", v)
                            if k == 'energy_loss':
                                print("\nE loss :", v)
                            if k is not None and v is not None:
                                if 'cupy' in str(type(v)):
                                    v = v.get()
                                if 'cupy' in str(type(k)):
                                    k = k.get()

                    for r in report_dict_:
                        for k, v in r.items():
                            if k is not None and v is not None:
                                if 'cupy' in str(type(v)):
                                    v = v.get()
                                if 'cupy' in str(type(k)):
                                    k = k.get()
                                writer.add_scalar("validation/{}".format(k), v, step)

                    plot_fn.__call__(step, input_length_, out_length_, att_ws)
                    plot_fn.log_attentions(writer, step, input_length_, out_length_, att_ws)

            if step % hp.save_interval == 0:
                save_path = os.path.join(hp.chkpt_dir, 'checkpoint_model_{}k_steps.pyt'.format(step // 1000))
                optim_path = os.path.join(hp.chkpt_dir, 'checkpoint_optim_{}k_steps.pyt'.format(step // 1000))
                torch.save(model.state_dict(), save_path)
                torch.save(optimizer.state_dict(), optim_path)
                print("Model Saved")
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

def num_params(model, print_out=True):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print('Trainable Parameters: %.3fM' % parameters)

def create_gta(args):
    os.makedirs(os.path.join(hp.data_dir, 'gta'), exist_ok=True)
    device = torch.device("cuda" if hp.ngpu > 0 else "cpu")

    dataloader = loader.get_tts_dataset(hp.data_dir, 1)
    validloader = loader.get_tts_dataset(hp.data_dir, 1, True)
    global_step = 0
    idim = hp.symbol_len
    odim = hp.num_mels
    model = fastspeech.FeedForwardTransformer(idim, odim, args)
    # set torch device
    if os.path.exists(args.resume):
        print('\nSynthesis GTA Session...\n')
        model.load_state_dict(torch.load(args.resume), strict=False)
    else:
        print("Checkpoint not exixts")
        return None
    model.eval()
    model = model.to(device)
    print("Model is loaded ...")
    print("Batch Size :",hp.batch_size)
    num_params(model)
    onlyValidation = False
    if not onlyValidation:
        pbar = tqdm.tqdm(dataloader, desc='Loading train data')
        for data in pbar:
            #start_b = time.time()
            global_step += 1
            x, input_length, y, _, out_length, ids = data
            with torch.no_grad():
                gta, _, _ = model._forward(x.cuda(), input_length.cuda(), y.cuda(), out_length.cuda())
                #gta = model._forward(x.cuda(), input_length.cuda(), is_inference=False)
            gta = gta.cpu().numpy()

            for j in range(len(ids)) :
                mel = gta[j]
                mel = mel.T
                mel = mel[:, :out_length[j]]
                mel = (mel + 4) / 8
                id = ids[j]
                np.save('{}/{}.npy'.format(os.path.join(hp.data_dir, 'gta'), id), mel, allow_pickle=False)

    pbar = tqdm.tqdm(validloader, desc='Loading Valid data')
    for data in pbar:
        #start_b = time.time()
        global_step += 1
        x, input_length, y, _, out_length, ids = data
        with torch.no_grad():
            gta, _, _ = model._forward(x.cuda(), input_length.cuda(), y.cuda(), out_length.cuda())
            #gta = model._forward(x.cuda(), input_length.cuda(), is_inference=True)
        gta = gta.cpu().numpy()

        for j in range(len(ids)) :
            print("Actual mel specs : {} = {}".format(ids[j],y[j].shape))
            print("Out length:",out_length[j])
            print("GTA size: {} = {}".format(ids[j],gta[j].shape))
            mel = gta[j]
            mel = mel.T
            mel = mel[:, :out_length[j]]
            mel = (mel + 4) / 8
            print("Mel size: {} = {}".format(ids[j],mel.shape))
            id = ids[j]
            np.save('{}/{}.npy'.format(os.path.join(hp.data_dir, 'gta'), id), mel, allow_pickle=False)


# define function for plot prob and att_ws
def _plot_and_save(array, figname, figsize=(6, 4), dpi=150):
    import matplotlib.pyplot as plt
    shape = array.shape
    if len(shape) == 1:
        # for eos probability
        fig=plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(array)
        plt.xlabel("Frame")
        plt.ylabel("Probability")
        plt.ylim([0, 1])
    elif len(shape) == 2:
        # for tacotron 2 attention weights, whose shape is (out_length, in_length)
        fig=plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(array, aspect="auto")
        plt.xlabel("Input")
        plt.ylabel("Output")
    elif len(shape) == 4:
        # for transformer attention weights, whose shape is (#leyers, #heads, out_length, in_length)
        fig=plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
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
        description='Train a new text-to-speech (TTS) model on one CPU, one or multiple GPUs',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--ngpu', default=1, type=int,
                        help='Number of GPUs. If not given, use all visible devices')
    parser.add_argument('--backend', default='pytorch', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--resume', '-r', default=None, type=str, nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?',
                        help="Tensorboard log directory path")
    parser.add_argument('--save-interval-epochs', default=1, type=int,
                        help="Save interval epochs")
    parser.add_argument('--report-interval-iters', default=100, type=int,
                        help="Report interval iterations")
    # task related
    parser.add_argument('--train-json', type=str, required=False,
                        help='Filename of training json')
    parser.add_argument('--valid-json', type=str, required=False,
                        help='Filename of validation json')
    # network architecture
    parser.add_argument('--model-module', type=str, default="espnet.nets.pytorch_backend.e2e_tts_tacotron2:Tacotron2",
                        help='model defined module')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batch-sort-key', default='shuffle', type=str,
                        choices=['shuffle', 'output', 'input'], nargs='?',
                        help='Batch sorting key. "shuffle" only work with --batch-count "seq".')
    parser.add_argument('--batch-count', default='auto', choices=BATCH_COUNT_CHOICES,
                        help='How to count batch_size. The default (auto) will find how to count by args.')
    parser.add_argument('--batch-size', '--batch-seqs', '-b', default=0, type=int,
                        help='Maximum seqs in a minibatch (0 to disable)')
    parser.add_argument('--batch-bins', default=0, type=int,
                        help='Maximum bins in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-in', default=0, type=int,
                        help='Maximum input frames in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-out', default=0, type=int,
                        help='Maximum output frames in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-inout', default=0, type=int,
                        help='Maximum input+output frames in a minibatch (0 to disable)')
    parser.add_argument('--maxlen-in', '--batch-seq-maxlen-in', default=100, type=int, metavar='ML',
                        help='When --batch-count=seq, batch size is reduced if the input sequence length > ML.')
    parser.add_argument('--maxlen-out', '--batch-seq-maxlen-out', default=200, type=int, metavar='ML',
                        help='When --batch-count=seq, batch size is reduced if the output sequence length > ML')
    parser.add_argument('--num-iter-processes', default=0, type=int,
                        help='Number of processes of iterator')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    parser.add_argument('--use-speaker-embedding', default=False, type=strtobool,
                        help='Whether to use speaker embedding')
    parser.add_argument('--use-second-target', default=False, type=strtobool,
                        help='Whether to use second target')
    # optimization related
    parser.add_argument('--opt', default='adam', type=str,
                        choices=['adam', 'noam'],
                        help='Optimizer')
    parser.add_argument('--accum-grad', default=1, type=int,
                        help='Number of gradient accumuration')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--eps', default=1e-6, type=float,
                        help='Epsilon for optimizer')
    parser.add_argument('--weight-decay', default=1e-6, type=float,
                        help='Weight decay coefficient for optimizer')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--early-stop-criterion', default='validation/main/loss', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--grad-clip', default=1, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--num-save-attention', default=5, type=int,
                        help='Number of samples of attention to be saved')
    parser.add_argument('--keep-all-data-on-mem', default=False, type=strtobool,
                        help='Whether to keep all data on memory')

    return parser

def main(cmd_args):

    #  Parse Arguments
    # parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    # parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    # parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    # parser.add_argument('--checkpoint', '-c', type=str, help='[string/path] checkpoint file to load weights from')
    # parser.set_defaults(checkpoint=None)
    # args = parser.parse_args()
    #
    # train(args)

    """Run training."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    #model_class = dynamic_import(args.model_module)
    #assert issubclass(model_class, TTSInterface)
    #fastspeech.FeedForwardTransformer.add_arguments(parser)
    args = parser.parse_args(cmd_args)

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    ngpu = hp.ngpu
    logging.info(f"ngpu: {ngpu}")

    # set random seed
    logging.info('random seed = %d' % hp.seed)
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    if hp.GTA:
        create_gta(args)
    else:
        train(args)


if __name__ == "__main__":
    main(sys.argv[1:])
