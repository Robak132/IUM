import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from microservice.models.model_b import SimpleNN
from models.train_utils_nn import train_and_extract
from models.utils import load_default_data, calculate_expenses

seed = 213769420
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    train, test, products, users, deliveries = load_default_data()
    targets = calculate_expenses(test, products, users)
    some_net = SimpleNN()
    train_and_extract(some_net, train, users, products, targets)
    torch.save(some_net.state_dict(), "simple_v1")
