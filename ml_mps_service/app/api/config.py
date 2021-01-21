import os
import boto3


PATH_PKG = os.path.dirname(os.path.abspath(__file__))
# PATH_PKG = "."
ML_MODEL_PATH = os.path.join(PATH_PKG, "ml_models")
MODEL_PATH = os.path.join(PATH_PKG, "ml_models/model.h5")
PATH_TO_SCALER_MPS = os.path.join(PATH_PKG, "ml_models/scaler_mps.joblib")
PATH_TO_SCALER_X = os.path.join(PATH_PKG, "ml_models/scaler_X.joblib")
PATH_RESULT = os.path.join(PATH_PKG,"result")
PATH_RESULT_FILE = os.path.join(PATH_RESULT,"ml_pred.out")

if not os.path.exists(PATH_RESULT):
    os.makedirs(PATH_RESULT)


session = boto3.Session(
    aws_access_key_id=os.environ["aws_access_key_id"],
    aws_secret_access_key=os.environ["aws_secret_access_key"],
    region_name=os.environ["region_name"])

s3_resource = session.resource('s3')
s3_client = session.client('s3')

folder_name = os.environ["folder"]
bucket = s3_resource.Bucket(folder_name)  # example: energy_market_procesing
