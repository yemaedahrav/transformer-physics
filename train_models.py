import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import Transformer
from util import set_seed
from config import linreg_config
from data import generate_linregdata


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(config, traindata, testdata, CL=65, loadmodel = False, fname = 'linreg', dir = 'models', batch_size = 64, num_epochs = 1, lr = 0.001):
    set_seed(10)
    if config is None:
        config = linreg_config() 
    base = f'{fname}_{config.n_embd}emb_{config.n_layer}layer'
    filebase = f'{base}_{CL}CL_{num_epochs}epochs_{lr}lr_{batch_size}batch'
    totalbase = f'{dir}/{base}/{filebase}'
    modelfile = f'{totalbase}_model.pth'
    lossesfile = f'{totalbase}_losses.pth'
    print("base: ", base)
    print("filebase: ", filebase)
    print("totalbase: ", totalbase)
    print("modelfile: ",modelfile)
    print("lossesfile: ", lossesfile)

    if base not in os.listdir(dir):
        os.mkdir(f'{dir}/{base}')

    if 'linreg' in fname:
        traindata = traindata.unsqueeze(-1)
        print("Linear Regression Train Data: ", traindata.shape)
        testdata = testdata.unsqueeze(-1)
        print("Linear Regression Test Data: ", testdata.shape)

    # Randomize traindata
    traindata = traindata[torch.randperm(traindata.size()[0])]
    traindata = traindata[:5000,:CL+1,:] # Only use 10 timesteps for transformer predictions. It's shown an ability to learn off of this.
    X, y = traindata[:,:-1,:], traindata[:,1:,:]
    print("X", X.shape)
    print("y", y.shape)
    
    # Train and in distribution test data
    div = int(0.8*len(X))
    X_train, y_train = X[:div], y[:div]
    X_test_in, y_test_in = X[div:], y[div:]
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test_in", X_test_in.shape)
    print("y_test_in", y_test_in.shape)

    testdata = testdata[:,:CL+1,:] # Only use 10 timesteps for transformer predictions. Rest of the data is for ICL experiments
    X_test_out, y_test_out = testdata[:,:-1,:], testdata[:,1:,:]

    # Create DataLoaders (test_in and test_out refer to the data being in-distribution or out-of-distribution, the out-of-distribution data is used for ICL testing)
    train_dataset = TensorDataset(X_train, y_train)
    test_in_dataset = TensorDataset(X_test_in, y_test_in)
    test_out_dataset = TensorDataset(X_test_out, y_test_out)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_in_loader = DataLoader(test_in_dataset, batch_size=batch_size, shuffle=False)
    test_out_loader = DataLoader(test_out_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = Transformer(config).to(device)
    if loadmodel:
        print(f'Loading model from {loadmodel}')
        model.load_state_dict(torch.load(loadmodel, map_location=device))
        for name, param in model.named_parameters():
            param.requires_grad = False 
    model.train()

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track the best model
    best_loss = float('inf')
    best_model_state = None

    # Training loop
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress')
    train_losses = []
    test_in_losses = []
    test_out_losses = []

    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            print("Training: ")
            print("batch_X", batch_X.shape)
            print("batch_y: ", batch_y.shape)
            print("output: ", output.shape)
            if 'linreg' in fname:
                loss = criterion(output[:, 0::2], batch_y[:,0::2])  # We only want the y values from model output.
            else:
                loss = criterion(output, batch_y)    
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        # Calculate Train Loss for the epoch
        epoch_train_loss = total_train_loss / len(train_loader)
        
        # Calculate Test Loss (In Distribution and Out Distribution)
        model.eval()
        total_test_in_loss = 0
        total_test_out_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in test_in_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                print("In Test: ")
                print("batch_X", batch_X.shape)
                print("batch_y: ", batch_y.shape)
                print("output: ", output.shape)
                loss = criterion(output, batch_y)
                total_test_in_loss += loss.item()
            for batch_X, batch_y in test_out_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                print("Out Test: ")
                print("batch_X", batch_X.shape)
                print("batch_y: ", batch_y.shape)
                print("output: ", output.shape)
                loss = criterion(output, batch_y)
                total_test_out_loss += loss.item()

        # Calculate Test Loss for the epoch
        epoch_test_in_loss = total_test_in_loss / len(test_in_loader)
        epoch_test_out_loss = total_test_out_loss / len(test_out_loader)

        # Save intermediate steps
        if epoch % 10 == 0:
            train_losses.append(epoch_train_loss)
            test_in_losses.append(epoch_test_in_loss)
            test_out_losses.append(epoch_test_out_loss)
        if epoch % 100 == 0:    
            torch.save({'train_losses': train_losses, 'test_in_losses': test_in_losses, 'test_out_losses': test_out_losses}, lossesfile)
        if epoch % 500 == 0:
            torch.save(model.state_dict(), f'{totalbase}_model_epoch{epoch}.pth')

        # Update the best model if the current loss is lower
        if epoch_test_in_loss < best_loss:
            best_loss = epoch_test_in_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, modelfile)
            
        # Update progress bar
        epoch_pbar.set_description(f'Epoch {epoch + 1}/{num_epochs}')
        epoch_pbar.set_postfix({'Train Loss': f'{epoch_train_loss:.2e}','Test-In Loss': f'{epoch_test_in_loss:.2e}','Test-Out Loss': f'{epoch_test_out_loss:.2e}'})

    # Save the best model
    torch.save(best_model_state, modelfile)
    torch.save({'train_losses': train_losses, 'test_in_losses': test_in_losses, 'test_out_losses': test_out_losses}, lossesfile)
    
    return model


def train_many(LWtitles, datadict, CL, my_task_id, num_tasks):
    if my_task_id is None:
        my_task_id = int(sys.argv[1])
    if num_tasks is None:
        num_tasks = int(sys.argv[2])
    fnames = LWtitles
    my_fnames = fnames[my_task_id:len(fnames):num_tasks]
    print(my_fnames)
    for L, W, title in my_fnames:
        config = linreg_config()
        config.n_layer = L
        config.n_embd = W
        config.max_seq_length = CL + 1
        train(config, datadict['traindata'], datadict[f'testdata'], fname = title, CL = CL)


def whatmodelstrain(LWtitles):
    num_tasks = len(LWtitles)
    print(num_tasks)
    for my_task_id in range(num_tasks):
        lw = LWtitles[my_task_id:len(LWtitles):num_tasks]
        print(f'{my_task_id}: {lw}')


if __name__ == '__main__':
    
    datadict = torch.load('data/linreg_data.pth')
    print(type(datadict))
    for key in datadict.keys():
        print(key, datadict[key].shape)

    titles = ['linreg']
    #Ls = [1,2,3,4,5]
    #Ws = [2,4,8,16,32]
    Ls = [2]; Ws = [16]
    CL = 10
    LWtitles = []
    for L in Ls:
        for W in Ws:
            for title in titles:
                LWtitles.append((L,W,title))
    whatmodelstrain(LWtitles)
    my_task_id = None
    num_tasks = None
    train_many(LWtitles, datadict, CL, my_task_id, num_tasks)
    sys.exit()
