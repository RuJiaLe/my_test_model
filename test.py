import torch
from utils.model import Model
import time
from torchvision.transforms import functional as F
from utils.material import count_param

model = Model()

start_time = time.time()

x1 = torch.rand(1, 3, 256, 256)
x2 = torch.rand(1, 3, 256, 256)
x3 = torch.rand(1, 3, 256, 256)
x4 = torch.rand(1, 3, 256, 256)
x5 = torch.rand(1, 3, 256, 256)
x6 = torch.rand(1, 3, 256, 256)
x_s = [x1, x2, x3, x4, x5, x6]

out, stage = model(x_s)

end_time = time.time()

print(f"time_cost: {(end_time - start_time) / 1.0}")

print(f'Model_parameter: {count_param(model) / 1e6}')
