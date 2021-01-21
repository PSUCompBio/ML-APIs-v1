from tensorflow.keras.models import load_model
import numpy as np
from joblib import load
from .config import *
from .download_models import download_models

download_models()


class MPSModel:
    model = load_model(MODEL_PATH)
    scaler_mps = load(PATH_TO_SCALER_MPS)
    scaler_x = load(PATH_TO_SCALER_X)

    @classmethod
    async def predict_mps(cls,data):
        data_trans = cls.scaler_x.transform(data)
        y = cls.model.predict(data_trans)
        y_exp = np.expm1(y)
        y_final = cls.scaler_mps.transform(y_exp)
        return y_final