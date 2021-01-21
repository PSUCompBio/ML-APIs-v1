import os
from .config import *


def download_models():
    """

    :param client:
    :return:
    """

    if not os.path.exists(ML_MODEL_PATH):
        os.makedirs(ML_MODEL_PATH)

    if not os.path.exists(PATH_TO_SCALER_MPS):
        print("Downloading Scaler MPS")
        s3_client.download_file('nsfcareer-users-data','models/ml/scaler_mps.joblib',PATH_TO_SCALER_MPS)

    if not os.path.exists(PATH_TO_SCALER_X):
        print("Downloading Scaler X")
        s3_client.download_file('nsfcareer-users-data','models/ml/scaler_X.joblib',PATH_TO_SCALER_X)

    if not os.path.exists(MODEL_PATH):
        print("Downloading ML model")
        s3_client.download_file('nsfcareer-users-data','models/ml/initial_model.h5',MODEL_PATH)


