from utils.hparams import HParam
from dataset.texts import valid_symbols
import torch

hp = HParam("./configs/default.yaml")
resume = "./checkpoints/first/ts_first_fastspeech_559e919_11k_steps.pyt"
import utils.fastspeech2_script as fs2
idim = len(valid_symbols)
odim = hp.audio.num_mels
model = fs2.FeedForwardTransformer(idim, odim, hp)
model.load_state_dict(torch.load(resume), strict=False)
my_script_module = torch.jit.script(model)
print("Scripting")
my_script_module.save("fastspeech_script_n.pt")
print("Script done")
# my_trace_module = torch.jit.trace(model, torch.ones(50))
# my_trace_module.save("trace_module.pt")

