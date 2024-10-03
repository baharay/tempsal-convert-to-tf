import torch
import torch.onnx
from collections import OrderedDict
from model import PNASBoostedModelMultiLevel
from model import *
model_path = "./checkpoints/multilevel_tempsal.pt"
# Assuming your model structure and state_dict loading as defined in your code
model_vol = PNASVolModellast(time_slices=5, load_weight=0)  # Modify this for time slices
model_vol = torch.nn.DataParallel(model_vol)#.cuda()

# Load the state dict
state_dict = torch.load(model_path,map_location=torch.device('cpu'))
vol_state_dict = OrderedDict()
sal_state_dict = OrderedDict()
smm_state_dict = OrderedDict()

# Populate the OrderedDicts with model weights
for k, v in state_dict.items():
    if 'pnas_vol' in k:
        k = k.replace('pnas_vol.module.', '')
        vol_state_dict[k] = v
    elif 'pnas_sal' in k:
        k = k.replace('pnas_sal.module.', '')
        sal_state_dict[k] = v
    else:
        smm_state_dict[k] = v

# Load the state dicts into the respective models
model_vol.load_state_dict(vol_state_dict)
model_sal = PNASModel(load_weight=0)
model_sal = torch.nn.DataParallel(model_sal)#.cuda()
model_sal.load_state_dict(sal_state_dict, strict=True)

# Set parameters to not require gradients
for param in model_vol.parameters():
    param.requires_grad = False
for param in model_sal.parameters():
    param.requires_grad = False

# Define dummy input based on your model's input shape
dummy_input = torch.randn(1, 3, 255, 255)#.cuda()  # Modify this as needed

# Export the PyTorch model to ONNX
torch.onnx.export(model_vol.module, dummy_input, "model_vol.onnx", export_params=True)
torch.onnx.export(model_sal.module, dummy_input, "model_sal.onnx", export_params=True)
