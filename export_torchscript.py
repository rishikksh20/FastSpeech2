from utils.hparams import HParam
from dataset.texts import valid_symbols
import torch

hp = HParam("./configs/default.yaml")
resume = "./checkpoints/checkpoint_model_156k_steps.pyt"
import utils.fastspeech2_script as fs2
idim = 56
odim = hp.audio.num_mels
model = fs2.FeedForwardTransformer(idim, odim, hp)
#model.load_state_dict(torch.load(resume), strict=False)
my_script_module = torch.jit.script(model)
my_script_module.save("fastspeech_script_n.pt")
#my_script_module.

