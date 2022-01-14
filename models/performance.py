import random
import torch
import numpy as np

from data.utils import load_default_data
from features.build_features import aggregate_users_data, calculate_expenses
from models.neural_networks.NeuralNetworkRegressor import NeuralNetworkRegressor
from models.neural_networks.utils import train


seed = 213769420
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    train_sessions, test_sessions, products, users, deliveries = load_default_data()
    targets = calculate_expenses(test_sessions, products, users)
    some_net = NeuralNetworkRegressor()
    user_data = aggregate_users_data(train_sessions, users, products)
    train(some_net, user_data, targets)
    torch.save(some_net.state_dict(), "parameters/simple_v1")
