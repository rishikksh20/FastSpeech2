from utils.hparams import HParam
from dataset.texts import valid_symbols
import torch

hp = HParam("./configs/default.yaml")

import utils.fastspeech2_script as fs2
idim = len(valid_symbols)
odim = hp.audio.num_mels
model = fs2.FeedForwardTransformer(idim, odim, hp)
my_script_module = torch.jit.script(model)
# torch.jit.trace(model,torch.ones(2, 25))