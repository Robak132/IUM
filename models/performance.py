import random
import torch
import numpy as np

from microservice.features import extract_users_data
from microservice.models.model_b import SimpleNN
from models.train_utils_nn import train
from models.utils import load_default_data, calculate_expenses

seed = 213769420
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    train_sessions, test_sessions, products, users, deliveries = load_default_data()
    targets = calculate_expenses(test_sessions, products, users)
    some_net = SimpleNN()
    user_data = extract_users_data(train_sessions, users, products)
    train(some_net, user_data, targets)
    torch.save(some_net.state_dict(), "simple_v1")
