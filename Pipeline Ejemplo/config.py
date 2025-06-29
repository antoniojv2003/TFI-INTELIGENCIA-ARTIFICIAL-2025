import os

# Constantes
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3 # Un número pequeño para una ejecución rápida, puede aumentarse
BASE_PATH = 'CheXpert-v1.0-small/'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
VALID_CSV = os.path.join(BASE_PATH, 'valid.csv')
LEARNING_RATE = 0.001

COMPETITION_TASKS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion'
] 