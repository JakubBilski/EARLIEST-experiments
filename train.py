import numpy as np
import torch
import sys
from model import EARLIEST
from dataset import SyntheticTimeSeries, UCRDataset
from torch.utils.data.sampler import SubsetRandomSampler
import utils
from sklearn.metrics import accuracy_score
from itertools import product



UCR_DATASETS = [
    "FiftyWords",
    "Adiac",
    "Beef",
    "CBF",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DiatomSizeReduction",
    "ECG200",
    "ECGFiveDays",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "Fish",
    "GunPoint",
    "Haptics",
    "InlineSkate",
    "ItalyPowerDemand",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "MedicalImages",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "Trace",
    "TwoPatterns",
    "TwoLeadECG",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "WordSynonyms",
    "Yoga"
]

LEARNING_RATES = [0.0001, 0.001, 0.01]
NEPOCHS = [20, 100, 500]
NLAYERS = [1, 3]
NHIDS = [10, 50]

CONFIGURATIONS = list(product(UCR_DATASETS, LEARNING_RATES, NEPOCHS, NLAYERS, NHIDS))

if __name__ == "__main__":
    random_seed = 42
    results_save_path = "./results/"
    batch_size = 100
    rnn_cell = 'LSTM'
    lam = 0.0

    utils.makedirs(results_save_path)

    array_position = int(sys.argv[1])

    dataset, learning_rate, nepochs, nlayers, nhid = CONFIGURATIONS[array_position]
    lr_str = f"{learning_rate}".replace('0.', '')
    results_filename = f"results_{dataset}_{nepochs}_{lr_str}_{nlayers}_{nhid}.txt"
    results_save_path = f"{results_save_path}{results_filename}"
    print(f"Running {results_save_path}")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    exponentials = utils.exponentialDecay(nepochs)

    data = UCRDataset(dataset, 'train')
    train_ix = utils.splitTrainingData(data.nseries)

    train_sampler = SubsetRandomSampler(train_ix)
    real_batch_size = min(batch_size, data.nseries)
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=real_batch_size,
                                               sampler=train_sampler,
                                               drop_last=False)

    model = EARLIEST(data.N_FEATURES, data.N_CLASSES, rnn_cell, nhid, nlayers, lam)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # --- training ---
    training_loss = []
    training_locations = []
    training_predictions = []
    for epoch in range(nepochs):
        model._REWARDS = 0
        model._r_sums = np.zeros(data.ntimesteps).reshape(1, -1)
        model._r_counts = np.zeros(data.ntimesteps).reshape(1, -1)
        model._epsilon = exponentials[epoch]
        loss_sum = 0
        for i, (X, y) in enumerate(train_loader):
            X = torch.transpose(X, 0, 1)
            # --- Forward pass ---
            logits, halting_points = model(X, epoch)
            _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

            training_locations.append(halting_points)
            training_predictions.append(predictions)

            # --- Compute gradients and update weights ---
            optimizer.zero_grad()
            loss = model.computeLoss(logits, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

            # if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, nepochs, i+1, len(train_loader), loss.item()))

        training_loss.append(np.round(loss_sum/len(train_loader), 3))
        scheduler.step()

    data = UCRDataset(dataset, 'test')
    testing_predictions = []
    testing_labels = []
    testing_locations = []
    test_ix = utils.splitTrainingData(data.nseries)
    test_sampler = SubsetRandomSampler(test_ix)
    real_batch_size = min(batch_size, data.nseries)
    test_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=real_batch_size,
                                              sampler=test_sampler,
                                              drop_last=False)
    for i, (X, y) in enumerate(test_loader):
        X = torch.transpose(X, 0, 1)
        logits, halting_points = model(X, test=True)
        _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

        testing_locations.extend(np.repeat(halting_points.numpy(), X.shape[1]))
        testing_predictions.extend(predictions.numpy().reshape(-1, 1))
        testing_labels.extend(y.numpy().reshape(-1, 1))

    # testing_predictions = torch.stack(testing_predictions).numpy().reshape(-1, 1)
    # testing_labels = torch.stack(testing_labels).numpy().reshape(-1, 1)
    # testing_locations = torch.stack(testing_locations).numpy().reshape(-1, 1)

    accuracy = np.round(accuracy_score(testing_labels, testing_predictions), 3)
    earliness = np.round(100.*np.mean(testing_locations), 3)

    print(f"Accuracy: {accuracy}")
    print(f"Earliness: {earliness}")

    f = open(results_save_path, 'w')
    f.writelines([f"{accuracy},{earliness}"])
    f.close()
