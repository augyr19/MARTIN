import torch
print(torch.__version__)
print(torch.cuda.is_available())  # should be True
print(torch.version.cuda)         # will show "11.8" or "12.6"
print(torch.cuda.get_device_name(0))
