from utils.hparams import HParam
from dataset.texts import valid_symbols
import utils.fastspeech2_script as fs2
import configargparse
import torch
import sys


def get_parser():

    parser = configargparse.ArgumentParser(
        description='Train a new text-to-speech (TTS) model on one CPU, one or multiple GPUs',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')

    return parser


def main(cmd_args):

    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    args = parser.parse_args(cmd_args)

    hp = HParam(args.config)

    idim = len(valid_symbols)
    odim = hp.audio.num_mels
    model = fs2.FeedForwardTransformer(idim, odim, hp)
    my_script_module = torch.jit.script(model)
    print("Scripting")
    my_script_module.save("{}/{}.pt".format(args.outdir, args.name))
    print("Script done")
    # my_trace_module = torch.jit.trace(model, torch.ones(50).to(dtype=torch.int64))
    # my_trace_module.save("trace_module.pt")


if __name__ == "__main__":
    main(sys.argv[1:])

