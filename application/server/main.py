import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse

import io
import librosa
import aiofiles
import  os
import random
import time

from fastapi.encoders import jsonable_encoder

from application.components import predict, read_imagefile
from application.schema import Symptom
from application.components.prediction import symptom_check

tags_metadata = [
    {
        "name": "Smart Stethoscope Prediction Service",
        "description": "Smart Stethoscope Prediction Service",
        "externalDocs": {
            "description": "Dev",
            "url": "https://fastapi.tiangolo.com/",
        },
    },
]
app_desc = """<h2>Smart Stethoscope Prediction Service</h2>
<br>by USJ """

app = FastAPI(title="Smart Stethoscope Prediction Service",
              description=app_desc,
              version="1.0.0"
              )

out_file_path = os.path.dirname(__file__)

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/api/v1/predict/", tags=["Prediction"])
async def predict(file: bytes = File(...)):
    audio_data_in, sr_in = librosa.load(io.BytesIO(file))
    length_in = len(audio_data_in) / sr_in
    #
    # predictor = PredictionService()
    # diagnosis_predictions = predictor.get_prediction(audio_data_in, sr_in, length_in)
    #
    # json_diagnosis_predictions = jsonable_encoder(list(diagnosis_predictions))

    return {"predictions": 'json_diagnosis_predictions'}

@app.post("/api/v4/predict/", tags=["Prediction"])
async def predict(image: UploadFile = File(...)):
    print(image.file)
    # print('../'+os.path.isdir(os.getcwd()+"images"),"*************")
    try:
        os.mkdir("images")
        print(os.getcwd())
    except Exception as e:
        print(e)
    file_name = os.getcwd() + "/images/" + image.filename.replace(" ", "-")
    with open(file_name, 'wb+') as f:
        f.write(image.file.read())
        f.close()

    audio_data_in, sr_in = librosa.load(file_name)

    return {"filename": sr_in }


@app.post("/api/v5/predict/", tags=["Prediction"])
async def predict(file: bytes = File(...)):
    audio_data_in, sr_in = librosa.load(file)
    length_in = len(audio_data_in) / sr_in
    #
    # predictor = PredictionService()
    # diagnosis_predictions = predictor.get_prediction(audio_data_in, sr_in, length_in)
    #
    # json_diagnosis_predictions = jsonable_encoder(list(diagnosis_predictions))

    return {"predictions": 'json_diagnosis_predictions'}

@app.post("/api/v2/predict/", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):

    content = await file.read()  # async read


    audio_data_in, sr_in = librosa.load(file.file)
    # length_in = len(audio_data_in) / sr_in
    #
    # predictor = PredictionService()
    # diagnosis_predictions = predictor.get_prediction(audio_data_in, sr_in, length_in)
    #
    #
    # json_diagnosis_predictions = jsonable_encoder(list(diagnosis_predictions))

    return {"predictions": file.filename}


@app.post("/api/v3/predict/", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    print(type(file.file))
    audio_data_in, sr_in = librosa.load(file.file)
    length_in = len(audio_data_in) / sr_in
    #
    # predictor = PredictionService()
    # diagnosis_predictions = predictor.get_prediction(audio_data_in, sr_in, length_in)
    #
    #
    # json_diagnosis_predictions = jsonable_encoder(list(diagnosis_predictions))

    return {"predictions": "Healthy"}

@app.post("/api/test/predict/", tags=["Test-Prediction"])
async def predict(file: UploadFile = File(...)):
    diagnosis_class_list = ['URTI', 'Healthy', 'COPD', 'Bronchiectasis', 'Pneumonia', 'Bronchiolitis']

    time.sleep(4)

    diagnosis_predictions = random.choices(diagnosis_class_list, weights=(3, 10, 80, 2, 3, 2), k=1)

    json_diagnosis_predictions = jsonable_encoder(list(diagnosis_predictions))

    return {"predictions": json_diagnosis_predictions}



if __name__ == "__main__":
    uvicorn.run(app, debug=True)
