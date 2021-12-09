import utils
from tqdm import tqdm
import numpy as np
import dataset
from autoencoder import AutoEncoder
import torch
import os
import json

def train_one_epoch(model, loader, opt, loss_func, epoch):
    """
    Trains a model for one epoch

    Parameters
    ----------
    model : torch.nn.Module
        Model class.
    loader : torch.utils.Dataloder
        dataset loader.
    opt : torch.optim
        optimizer.
    loss_func : TYPE
        Loss function.
    epoch : int
        Current epoch.

    Returns
    -------
    None.

    """
    
    training_loss = []
    # ITerate through each batch
    model.train()
    for batch_index, sequence in enumerate(loader):
        
        # Send to device
        sequence = sequence.to('cuda')
        
        # Forward
        opt.zero_grad()
        pred = model(sequence)
        
        # Backprop
        loss = loss_func(pred, sequence)
        training_loss.append(loss.item())
        loss.backward()
        opt.step()
    
    return model, np.mean(training_loss)

def infer(model, loader, loss_func, show_progress=False):
    """
    Infers and returns predictions using the given model and loader

    Parameters
    ----------
    model : torch.nn.Module
        Model class.
    loader : torch.utils.Dataloder
        dataset loader.
    loss_func : TYPE
        Loss function.
    epoch : int
        Current epoch.

    Returns
    -------
    None.

    """
    
    # Holders
    predictions = []
    test_loss = []
    
    if show_progress:
        pbar = tqdm(enumerate(loader), desc='Infering...')
    else:
        pbar = enumerate(loader)
    
    # Iterate through each batch
    model.eval()
    with torch.no_grad():
        
        for batch_index, sequence in pbar:
            
            # Send to device
            sequence = sequence.to('cuda')
            
            # Forward
            pred = model(sequence)
            
            # Grab loss, prediction
            loss = loss_func(pred, sequence)
            test_loss.append(loss.item())
            predictions.append(pred.detach().cpu().tolist())
    
    return test_loss, predictions

def train_AE(record_path, num_train, seqLen, percTest, model_file_path, embeddingDim, loss_name):
    
    # Load data
    data, gts, _, _ = utils.getData(record_path, normalize=False)

    # Split into offline (training), online(testing)
    offline_data, offline_gts, online_data, online_gts = utils.trainSplit(record_path, data, gts, num_train=num_train)
        
    # Normalize data
    offline_data, og_means, og_stdevs = utils.normalizeData(offline_data)
    
    # Create data loaders
    test_index = offline_data.shape[0] - int(percTest * offline_data.shape[0]) 
    train_set = offline_data[:test_index, :]
    test_set = offline_data[test_index:, :]
    train_loader, test_loader, val_loader = dataset.get_dataloaders(train_set, test_set, online_data,
                                                                    og_means, og_stdevs, num_train, 
                                                                    normalize_online=True, seqLen=seqLen,
                                                                    batch_size=8)

    
    # Create model
    model = AutoEncoder(offline_data.shape[1], embeddingDim=embeddingDim).to('cuda')
    num_epochs = 300
    
    # Optimizer, loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if loss_name == 'MSE':
        loss = torch.nn.MSELoss().to('cuda')
    else:
        loss = torch.nn.L1Loss(reduction='sum').to('cuda')
    
    # Holders for loss
    test_loss = []
    train_loss = []
    prev_loss = 500000    
    
    # Train
    for epoch in tqdm(range(num_epochs), desc='Training...'):
        
        # Train for one epoch
        model, training_loss = train_one_epoch(model, train_loader, optimizer, loss, epoch)
        
        # Get test accuracy
        testing_loss, _ = infer(model, test_loader, loss, show_progress=False)
        
        # Save the best one
        if np.mean(testing_loss) < prev_loss:
            prev_loss = np.mean(testing_loss)
            torch.save({'epoch':epoch,
                        'model_dict':model.state_dict(),
                        'opt_dict':optimizer.state_dict(),
                        'loss':testing_loss}, model_file_path)
        
        # Save
        train_loss.append(training_loss)
        test_loss.append(np.mean(testing_loss))


    return

def test_AE(record_path, num_train, seqLen, percTest, model_file_path, embeddingDim, loss_name):
    
    # Load data
    data, gts, _, _ = utils.getData(record_path, normalize=False)

    # Split into offline (training), online(testing)
    num_train = 1900
    offline_data, offline_gts, online_data, online_gts = utils.trainSplit(record_path, data, gts, num_train=num_train)
    

    # Normalize data
    offline_data, og_means, og_stdevs = utils.normalizeData(offline_data)
    
    # Generate anomalies
    clean_data, dirty_data, dirty_gts = utils.generateSmartAnomalies(online_data, og_means, og_stdevs, 3.5, num_anomalies=1000)

    
    # Create data loaders
    test_index = offline_data.shape[0] - int(percTest * offline_data.shape[0]) 
    train_set = offline_data[:test_index, :]
    test_set = offline_data[test_index:, :]
    train_loader, test_loader, val_loader = dataset.get_dataloaders(train_set, test_set, dirty_data,
                                                                    og_means, og_stdevs, num_train, 
                                                                    normalize_online=True, seqLen=seqLen,
                                                                    batch_size=8)

    # Create model
    model = AutoEncoder(offline_data.shape[1], embeddingDim=embeddingDim).to('cuda')
    
    # Load model params
    cp = torch.load(model_file_path)
    model.load_state_dict(cp['model_dict'])    
    
    # loss
    if loss_name == 'MSE':
        loss = torch.nn.MSELoss().to('cuda')
    else:
        loss = torch.nn.L1Loss(reduction='sum').to('cuda')

    
    # Grab the final testing loss for all instances in the training set.
    final_loss, _ = infer(model, train_loader, loss, show_progress=True)
    mean, stdev = np.mean(final_loss), np.std(final_loss)
    print('Mean: ', mean, ' Stdev: ', stdev)

    # Run through dirty set
    loss, predictions = infer(model, val_loader, loss, show_progress=True)
    
    # Generate alarms
    preds = np.zeros((len(predictions), 1))
    thresh = mean + 6*stdev
    preds[loss >= thresh] = 1
    
    # Plot online info
    utils.plotDirtyData(clean_data, dirty_data, dirty_gts, preds)
    
    
    # Print recall, precision
    (recall, fpr) = utils.computeMetrics(preds, dirty_gts)
    
    # Plot ROC curve
    utils.plotROCCurve(dirty_gts, loss, mean, stdev)
    print('\nRecall: ', recall)
    print('FPR: ', fpr)
    return
        

if __name__ == "__main__":

    num_train = 1900
    percTest = 0.1
    embeddingDim=64
    loss = 'L1Loss'
    data_path = "./data/221.csv"
    record_ID = os.path.basename(data_path).split(".")[0]
    model_file_path = os.path.join("./models", f'AE_model_1_seqLen_{loss}_{str(embeddingDim)}Emb_{record_ID}_{str(num_train)}train.pt')
    train_AE(data_path, num_train, 1, percTest, model_file_path, embeddingDim, loss)
    test_AE(data_path, num_train, 1, percTest, model_file_path, embeddingDim, loss)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    