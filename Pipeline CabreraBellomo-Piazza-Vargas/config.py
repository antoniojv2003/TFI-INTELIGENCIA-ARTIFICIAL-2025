import os

# Constantes
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8 # Un número pequeño para una ejecución rápida, puede aumentarse
BASE_PATH = 'CheXpert-v1.0-small/'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
VALID_CSV = os.path.join(BASE_PATH, 'valid.csv')
LEARNING_RATE = 0.001
#cambie el learning rate de 0.001 a 0.005
COMPETITION_TASKS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion'
] 
THRESHOLDS = {
    'Atelectasis':      0.341,
    'Cardiomegaly':     0.067,
    'Consolidation':    0.088,
    'Edema':            0.502,
    'Pleural Effusion': 0.327
}
