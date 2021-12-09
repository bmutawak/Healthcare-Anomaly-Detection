from os.path import dirname, basename, join, exists
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import csv
from math import sqrt
from sklearn.covariance import MinCovDet
from scipy import stats
from sklearn import metrics
import json

font = {
        'size'   : 20}

matplotlib.rc('font', **font)


def getData(record_path, normalize=False):
    """
    Reads and formats the data specified at record_path.

    Parameters
    ----------
    folder_path : STR
        Path containing record data.
    normalize   : Bool
        Whether to normalize the data or not. Z-score normalization is performed.

    Returns
    -------
    Corresponding matrix of data.

    """
    
    # Make sure path exists
    if not exists(record_path):
        print("ERROR: PATH DOES NOT EXIST")
        return
    
    # Check to see if a parsed pkl is available
    parsed_path = record_path.replace('.csv', '.pkl')
    if exists(parsed_path):
        with open(parsed_path, 'rb') as in_file:
            data = pickle.load(in_file)
    
    else:
        
        # Open the csv
        with open(record_path, 'r') as in_file:
            
            # Create reader
            reader = csv.reader(in_file)
            
            # Grab titles, units
            signal_names = next(reader)
            next(reader)
            
            # Create data 
            data = []
            for row in reader:
                data.append(row)
                
            data = np.array(data, dtype=float)
            
        # Store for later
        with open(parsed_path, 'wb') as out_file:
            pickle.dump(data, out_file)

    means, stdevs = [], []
    # Normalize if needed
    if normalize:
        data, means, stdevs = normalizeData(data)
    
    # The ground truth according to the paper
    gts = np.zeros(data.shape[0])
    gts[3000:3400] = 1
    gts[10600:11300] = 1
    gts[13660:13800] = 1
    gts[19400:19500] = 1
    return data, gts, means, stdevs

def generateAnomalies(data, num_anomalies=1000, anomalous_val=0):
    """
    Randomly ingest anomalies into the dataset. Returns the labels.

    Parameters
    ----------
    data : ndarray
        Input data.
    num_anomalies : int
        How many anomalies to generate.
        
    Returns
    -------
    Untouched data, anomaly data, ground truth.

    """
    
    # Extract data that are non-anomalous
    clean_data = np.vstack((data[3706:10730, :],
                            data[11900:19400, :],
                            data[21200:22640, :],
                            data[22720:28917, :]))
    
    dirty_data = np.copy(clean_data)
    
    # Holders for gts
    gts = np.zeros((dirty_data.shape[0], 1))
    
    # Generate anomalies
    for i in range(num_anomalies):
        
        # Grab a random instance and signal index
        idx = np.random.randint(dirty_data.shape[0])
        signal_idx = np.random.randint(dirty_data.shape[1])

        # Set a random signal to zero
        dirty_data[idx, signal_idx] = anomalous_val
        gts[idx] = 1
        
    return clean_data, dirty_data, gts

def generateSmartAnomalies(data, means, stdevs, num_stds, num_anomalies=1000):
    """
    Randomly ingect anomalies into the dataset. Returns the labels.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Extract data that are non-anomalous
    clean_data = np.vstack((data[3706:10730, :],
                            data[11900:19400, :],
                            data[21200:22640, :],
                            data[22720:28917, :]))
    
    dirty_data = np.copy(clean_data)
    
    # Holders for gts
    gts = np.zeros((dirty_data.shape[0], 1))
    
    # Generate anomalies
    for i in range(num_anomalies):
        
        # Grab a random instance and signal index
        idx = np.random.randint(dirty_data.shape[0])
        signal_idx = np.random.randint(dirty_data.shape[1])

        # Set a random signal to xstdevs
        dirty_data[idx, :] = dirty_data[idx, :] + num_stds*stdevs
        gts[idx] = 1
        
    return clean_data, dirty_data, gts
        
def plotAnomalies(smart_data, dumb_data):
    """
    Plots injected anomalous data
    """
    # Create figure, axs
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    
    # Match colors to the paper
    colors = ['black','blue', 'purple', 'red', 'darkgreen', 'orange', 'lime']
    names = ['BPmean', 'BPsys', 'BPdias', 'HR', 'PULSE', 'RESP', 'SpO2']
    
    # Plot the signals
    for i in range(smart_data.shape[1]):
        axs[1].plot(smart_data[:, i], linewidth=1.25, color=colors[i], label=names[i])
        axs[0].plot(dumb_data[:, i], linewidth=1.25, color=colors[i], label=names[i])
        
    axs[0].set_title('Recording Error Injected Dataset')
    axs[1].set_title('Anomolous Points Injected Dataset')
    axs[1].set_xlabel('Samples')
    axs[0].set_ylabel('Physiological Signals')
    axs[1].set_ylabel('Physiological Signals')
    # plt.grid()
    # plt.
    for line in axs[0].legend(loc="upper left", mode = "expand", ncol = len(names)).get_lines():
        line.set_linewidth(4)
    for line in axs[1].legend(loc="upper left", mode = "expand", ncol = len(names)).get_lines():
        line.set_linewidth(4)
    plt.show()
    return

    
def normalizeData(data, mean_array=None, std_array=None, iteration=-1):
    """
    Performs Z-score normalization on each signal provided.

    Parameters
    ----------
    data : dict
        Input data dict.

    Returns
    -------
    Normalized data.

    """
    
    means, stdevs = [], []

    if mean_array is None:
        
        # Go through each signal
        for i in range(data.shape[1]):
            signal = data[:, i]
            
            # Compute mean, standard deviation
            mean = np.mean(signal)
            std = np.std(signal)
            
            means.append(mean)
            stdevs.append(std)
            
            # Normalize
            data[:, i] = (signal - mean) / std
            
        means = np.array(means)
        stdevs = np.array(stdevs)
        
    else:
        
        means = mean_array
        stdevs = std_array
        # Normalize and update means, stdevs
        for i in range(means.shape[-1]):
            new_mean = means[:, i] + ((data[i] - means[:, i]) / iteration)
            variance = stdevs[:, i] ** 2
            stdevs[:, i] = np.sqrt((1 / (1 + iteration)) * (iteration * variance + (data[i] - new_mean) * (data[i] - new_mean)))
            means[:, i] = new_mean

    
        data = (data - means) / stdevs
            
    return data, means, stdevs

def computeThreshold(data, alpha=0.75, gamma=0.975):
    """
    Computes the SPE threshold as specified in the paper.
    """
    
    # Compute covariance matrix
    cov = np.cov(data.T)
    
    # Compute thetas
    theta_1 = np.trace(cov, offset=1)
    theta_2 = np.trace(cov, offset=2)
    theta_3 = np.trace(cov, offset=3)
    
    # H
    h = (2 * theta_1 * theta_3) / (3 * theta_2**2)
    
    # Za?
    z_a = stats.norm.ppf(gamma)
    
    p_1 = 1 + (z_a*sqrt(2 * theta_2 * (h**2)) / theta_1)
    p_2 = (theta_2 * h * (h - 1)) / (theta_1**2)
    threshold = theta_1 * ((p_1 + p_2)**(1 / h))
    return threshold


def plotData(online_data, lim=20000, title=''):
    
    # Create figure, axs
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    
    # Match colors to the paper
    colors = ['black','blue', 'purple', 'red', 'darkgreen', 'orange', 'lime']
    names = ['BPmean', 'BPsys', 'BPdias', 'HR', 'PULSE', 'RESP', 'SpO2']
    
    # Plot the signals
    for i in range(online_data.shape[1]):
        ax.plot(online_data[:lim, i], linewidth=1.25, color=colors[i], label=names[i])
        
    plt.title(f'Physiological Parameters {title}')
    plt.xlabel('Samples')
    plt.ylabel('Physiological Signals')
    plt.grid()
    # plt.
    for line in plt.legend(loc="upper left", mode = "expand", ncol = len(names)).get_lines():
        line.set_linewidth(4)
    plt.show()
    return

def plotSPE(online_data, spes, preds, gts, thresh=8.2, lim=20000):
    """
    Plots SPEs
    """
    
    # Create figure, axs
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    
    # Match colors to the paper
    colors = ['black','blue', 'purple', 'red', 'darkgreen', 'orange', 'lime']
    
    # Plot the signals
    for i in range(online_data.shape[1]):
        axs[0].plot(online_data[:lim, i], linewidth=1.25, color=colors[i])
    
    # Plot SPEs
    axs[1].plot(spes.reshape(-1, 1), 'r-', linewidth=1.25)
    xs = [0, spes.shape[-1]]
    ys=  [thresh, thresh]
    axs[1].plot(xs, ys, linewidth=1.25 )
    
    # Plot gts
    axs[2].plot(preds, 'r--', linewidth=1.75, label='Predictions', )
    axs[2].plot(gts[:lim], linewidth=1.75, label='GTs')
    
    leg = plt.legend(loc="upper left")
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    
    plt.setp(axs[0], ylabel='Online Signals')
    plt.setp(axs[1], ylabel='Squared Prediction Error')
    plt.setp(axs[2], ylabel='Alarms and Ground Truth')
    plt.show()
    return


def plotDirtyData(clean_data, dirty_data, gts, preds):
    """
    Plots SPEs
    """
    
    # Create figure, axs
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    
    # Match colors to the paper
    colors = ['black','blue', 'purple', 'red', 'darkgreen', 'orange', 'lime']
    
    # Plot the clean and dirty signals
    for i in range(clean_data.shape[1]):
        axs[0].plot(clean_data[:, i], linewidth=1.25, color=colors[i])
        axs[1].plot(dirty_data[:, i], linewidth=1.25, color=colors[i])
    
    
    # Plot gts
    axs[2].plot(gts, linewidth=1.75, label='GTs')
    axs[2].plot(preds, linewidth=1.75, label='Predictions')
    
    leg = plt.legend(loc="upper left")
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    
    plt.setp(axs[0], ylabel='Clean Signals')
    plt.setp(axs[1], ylabel='Dirty Signals')
    plt.setp(axs[2], ylabel='Ground Truth and Predictions')
    plt.show()
    return

def plotROCCurve(gts, loss, mean, stdev):
    """
    Plots SPEs
    """
    
    # Generate points at different thresholds
    pts = []
    for i in np.arange(0, 10, 0.05):
        alarms = np.zeros((len(loss), 1))
        thresh = mean + i*stdev
        alarms[loss >= thresh] = 1
        (recall, fpr) = computeMetrics(alarms, gts)
        pts.append([recall, fpr])
        
    # Sort
    pts = np.array(pts)
    pts = np.sort(pts, axis=0)
    
    
    # Create figure, axs
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    
    # Plot
    ax.plot(pts[:, 1], pts[:, 0], linewidth=1.75, label='ROC')
    
    leg = plt.legend(loc="upper right")
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    ax.set_xscale('log')
    plt.setp(ax, ylabel='Recall')
    plt.setp(ax, xlabel='FPR')
    plt.show()
    return


def plotPredictions(online_data, preds, gts, lim=20000):
    """
    Plots Predictions
    """
    
    # Create figure, axs
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    
    # Match colors to the paper
    colors = ['black','blue', 'purple', 'red', 'darkgreen', 'orange', 'lime']
    
    # Plot the signals
    for i in range(online_data.shape[1]):
        axs[0].plot(online_data[:lim, i], linewidth=1.25, color=colors[i])
    
    # Plot gts
    axs[1].plot(preds, 'r--', linewidth=1.75, label='Predictions', )
    axs[1].plot(gts[:lim], linewidth=1.75, label='GTs')
    
    leg = plt.legend(loc="upper left")
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    
    plt.setp(axs[0], ylabel='Online Signals')
    plt.setp(axs[1], ylabel='Alarms and Ground Truth')
    plt.show()
    return

def plotTrainingLoss(train_loss, test_loss):
    """
    Plots loss
    """
    
    # Create figure, axs
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)

    # Plot the loss
    ax.plot(train_loss, linewidth=1.25, color='blue', label='Training Loss')
    ax.plot(test_loss, linewidth=1.25, color='red', label='Testing Loss')
    
    # Legend, titles, etc.
    leg = plt.legend(loc="upper left")
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    
    plt.setp(ax, ylabel='Loss')
    plt.setp(ax, ylabel='Epoch')
    plt.show()
    return

def plotOnlineLoss(preds, online_data):
    """
    Plots Predictions
    """
    
    # Create figure, axs
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    
    # Match colors to the paper
    colors = ['black','blue', 'purple', 'red', 'darkgreen', 'orange', 'lime']
    
    # Plot the signals
    for i in range(online_data.shape[1]):
        axs[0].plot(online_data[:, i], linewidth=1.25, color=colors[i])
    
    # Plot gts
    axs[1].plot(preds, linewidth=1.75, label='Predictions')
    
    leg = plt.legend(loc="upper left")
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    
    plt.setp(axs[0], ylabel='Input Signals')
    plt.setp(axs[1], ylabel='Predictions')
    plt.show()
    return


def computeMetrics(predictions, gts):
    """
    Computes recall and FPR given the predictions and ground truth
    """
    
    tn, fp, fn, tp = metrics.confusion_matrix(gts, predictions, labels=[0, 1]).ravel()

    return [(tp / (tp + fn)), (fp / (fp + tn))]
    

def computeUV(eigenvalues, k):
    """
    Computes the unexplained variability as a result of choosing the first
    k principle components.
    
    Equation 5. of the article.

    Parameters
    ----------
    eigenvalues : ndarray
        1xM ndarray with M being the number of features.

    Returns
    -------
    UV : float
        unexplained variability.

    """

    return round(np.sum(eigenvalues[k+1:]) / np.sum(eigenvalues), 3)



def computeSPE(data_point, eigenvectors):
    """
    Computes the square prediction error.
    """
    x_hat = np.dot(data_point, eigenvectors.T)
    
    SPE = 0
    for i in range(len(x_hat)):
        SPE += (data_point[:, i] - x_hat[:, i])**2
    return SPE
        
def computeEstimationError(init_vectors, eigenvectors):
    """
    Computes Eigenspace estimation error per the article.
    """
    
    eee = 0
    # vectors = np.matrix(vectors)
    init_vectors = np.matrix(init_vectors)
    for k in range(eigenvectors.shape[0]):
        p1 = np.matmul(eigenvectors[k, :].T, eigenvectors[k, :])
        p2 = np.matmul(init_vectors[k, :].T, init_vectors[k, :])
        eee += (np.linalg.norm(p1 - p2, ord='fro') / np.linalg.norm(p2, ord='fro'))
    return eee
        
def trainSplit(data_path, data, gts, num_train=900):
    """
    Splits the data sequentially into a training and test set.

    """
    
    if '221' in data_path:
        
        return data[:num_train, :], gts[:num_train], data[num_train:, :], gts[num_train:]
    
    elif '226' in data_path:
        return data[60360:60360 + num_train, :], gts[60360:60360 + num_train], np.vstack((data[:60360, :], data[60360+num_train:, :])), np.append(gts[:60360],gts[60360+num_train:])
    
    elif '401' in data_path:
        return data[30000:30000 + num_train, :], gts[30000:30000 + num_train], np.vstack((data[:30000, :], data[30000+num_train:, :])), np.append(gts[:30000],gts[30000+num_train:])
    
    elif '403' in data_path:
        return data[38400:38400 + num_train, :], gts[38400:38400 + num_train], np.vstack((data[:38400, :], data[38400+num_train:, :])), np.append(gts[:38400],gts[38400+num_train:])
    elif '408' in data_path:
        return data[86900:86900 + num_train, :], gts[86900:86900 + num_train], np.vstack((data[:86900, :], data[86900+num_train:, :])), np.append(gts[:86900],gts[86900+num_train:])
            
    
    
def computeStatistics(eees):
    """
    Computes min, max, q1, s3, and median averaged.
    """
    
    # Init to zeros
    mini, maxi, q1, q3, median = 0, 0, 0, 0, 0

    # Loop through all errors
    for eee in eees:
        
        # Compute metrics
        (q1i, q3i, mediani) = np.percentile(eee, [25, 50, 75])
        minii, maxii = min(eee), max(eee)
        mini += minii
        maxi += maxii
        q1 += q1i
        q3 += q3i
        median += mediani
    
    num_iter = len(eees)
    return np.array([mini, maxi, q1, q3, median]) / num_iter 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    