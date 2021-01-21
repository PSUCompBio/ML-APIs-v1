from fastapi import APIRouter, HTTPException
from .schema import MPSData
from .ml import MPSModel
from .preprocess import create_input_feature_table
import json
import numpy as np
from .config import *
from fastapi import BackgroundTasks


mps = APIRouter()


def upload_result(result,s3_path):
    np.savetxt(PATH_RESULT_FILE, result)

    s3_client.upload_file(PATH_RESULT_FILE, folder_name, s3_path)


@mps.post('/mps-single',status_code=201)
async def predict_mps(data:MPSData,background_tasks: BackgroundTasks):
    obj = bucket.Object(
        key=data.simulation_path)  # example: market/zone1/data.csv
    # get the object
    response = obj.get()
    # read the contents of the file
    input_data = response['Body'].read().decode()

    json_data = json.loads(input_data)
    feature = create_input_feature_table(json_data)

    try:
        result = await MPSModel.predict_mps(feature)
        result_path = os.path.join(os.path.dirname(data.simulation_path),os.path.basename(PATH_RESULT_FILE))
        background_tasks.add_task(upload_result,result,result_path)

        return {"message":f"Result file is available at {result_path}", "success":True}
    except Exception as E:
        return {"message":str(E),
                "success":False}
