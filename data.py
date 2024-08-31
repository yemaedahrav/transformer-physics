import torch
import sys
from util import set_seed
import matplotlib.pyplot as plt

set_seed(12)

def generate_linregdata(num_samples = 5000, sequence_length = 65):
    # Sequence length is the number of cases/points/examples/timesteps on each linear regression (which remains same across linear regressions)
    # Samples is the count of different linear regressions
    tlow, thigh = 0.75, 1
    # Generating a 5000x65 torch tensor with random values in specified ranges
    # 5000 linear regressions of 65 timesteps each
    num_test = num_samples//5
    x_test = torch.empty(num_test, sequence_length).uniform_(-1, 1)  # Fill with values between -1 and 1 initially
    mask = x_test < 0  # Create a mask for negative values
    x_test[mask] = x_test[mask] * (thigh - tlow) - tlow  # Adjust negative values to be between -1 and -0.75
    x_test[~mask] = x_test[~mask] * (thigh - tlow) + tlow  # Adjust positive values to be between 0.75 and 1
    
    w_test = torch.empty(num_test,).uniform_(-1, 1)  # Fill with values between -1 and 1 initially
    mask = w_test < 0  # Create a mask for negative values
    w_test[mask] = w_test[mask] * (thigh - tlow) - tlow  # Adjust negative values to be between -1 and -0.75
    w_test[~mask] = w_test[~mask] * (thigh - tlow) + tlow  # Adjust positive values to be between 0.75 and 1
    y_test = w_test.unsqueeze(-1) * x_test  # Calculate the target values

    x_test_exp = x_test.unsqueeze(2)  # Shape becomes (num_samples, sequence_length, 1)
    y_test_exp = y_test.unsqueeze(2)
    testdata = torch.cat((x_test_exp, y_test_exp), dim=2)
    testdata = testdata.view(num_test, -1)
    
    # the in distribution data is from (-0.75, 0.75)
    x_train = torch.empty(num_samples, sequence_length).uniform_(-0.75, 0.75)
    w_train = torch.empty(num_samples,).uniform_(-0.75, 0.75) 
    y_train = w_train.unsqueeze(-1) * x_train

    x_train_exp = x_train.unsqueeze(2)  # Shape becomes (num_samples, sequence_length, 1)
    y_train_exp = y_train.unsqueeze(2)
    traindata = torch.cat((x_train_exp, y_train_exp), dim=2)
    print("traindata ", traindata.shape)
    sys.exit()
    traindata = traindata.view(num_samples, -1)
    
    # The w_train and the w_test are not really fed to the transformer for training, they are instead used for probing.
    torch.save({
        'x_train': x_train,
        'w_train': w_train,
        'y_train': y_train,
        'traindata': traindata,
        'x_test': x_test,
        'w_test': w_test,
        'y_test': y_test,
        'testdata': testdata

    }, f'data/linreg_data.pth')

if __name__ == '__main__':
    generate_linregdata(5000, 65)
