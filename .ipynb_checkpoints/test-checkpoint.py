import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
a=torch.tensor(1.2)
a.cuda()
print(a.cuda())
from torch.backends import cudnn
print(cudnn.is_available())
print(cudnn.is_acceptable(a.cuda()))
torch.tensor(5.0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
a=torch.tensor(100.)
a
a.to(device)
a
torch.device("cuda:0")
torch.cuda.empty_cache()
%pip config list