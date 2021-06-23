import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse

import io
import librosa
import aiofiles
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
async def predict(uploaded_file: UploadFile = File(...)):
    # ...
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    return {"info": f"file '{uploaded_file.filename}' saved at '{file_location}'"}


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


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
