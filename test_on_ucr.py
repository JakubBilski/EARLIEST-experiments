from model import EARLIEST
from sktime.datasets import load_UCR_UEA_dataset
import pytorch as torch
import numpy as np


DATASETS = [
    "FiftyWords",
    "Adiac",
    "Beef",
    "CBF",
    "ChlorineConcentration",
    "CinCECGtorso",
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


# --- hyperparameters ---
N_FEATURES = 5 # Number of variables in your data (if we have clinical time series recording both heart rate and blood pressure, this would be a 2-dimensional time series, regardless of the number of timesteps)
N_CLASSES = 3 # Number of classes
HIDDEN_DIM = 50 # Hidden dimension of the RNN
CELL_TYPE = "LSTM" # Use an LSTM as the Recurrent Memory Cell.
NUM_TIMESTEPS = 10 # Number of timesteps in your input series (EARLIEST doesn't need this as input, this is just set to create synthetic series)
BATCH_SIZE = 32 # Pick your batch size
LAMBDA = 0.0 # Set lambda, the emphasis on earliness

# --- defining data and model ---
d = torch.rand((NUM_TIMESTEPS, BATCH_SIZE, N_FEATURES)) # A simple synthetic time series.
X, y = load_UCR_UEA_dataset(name=DATASETS[0], return_X_y=True)
data = torch.tensor(np.asarray(X).astype(np.float32),
                    dtype=torch.float)
labels = torch.tensor(np.array(y).astype(np.int32), dtype=torch.long)

m = EARLIEST(N_FEATURES, N_CLASSES, HIDDEN_DIM, CELL_TYPE, lam=LAMBDA) # Initializing the model


# --- inference ---
# Now we can use m for inference
logits, halting_points = m(d)
_, predictions = torch.max(torch.softmax(logits, 1), 1)

# --- computing loss and gradients ---
# Computing the loss is quite simple:
loss = m.applyLoss(logits, labels)
loss.backward() # Compute all gradients

