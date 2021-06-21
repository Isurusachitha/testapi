import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse

# import io
# import librosa
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
              description="API Documentation",
              version="1.0.0"
              )


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)

    return prediction


@app.post("/api/covid-symptom-check")
def check_risk(symptom: Symptom):
    return symptom_check.get_risk_level(symptom)


@app.post("/api/v1/predict/", tags=["Prediction"])
async def predict(file: bytes = File(...)):
    audio_data_in, sr_in = librosa.load(io.BytesIO(file))
    # length_in = len(audio_data_in) / sr_in
    #
    # predictor = PredictionService()
    # diagnosis_predictions = predictor.get_prediction(audio_data_in, sr_in, length_in)
    #
    #
    # json_diagnosis_predictions = jsonable_encoder(list(diagnosis_predictions))

    return {"predictions": "json_diagnosis_predictions"}

@app.post("/api/v2/predict/", tags=["Prediction"])
async def predict(file: bytes = File(...)):
    # audio_data_in, sr_in = librosa.load(io.BytesIO(file))
    # length_in = len(audio_data_in) / sr_in
    #
    # predictor = PredictionService()
    # diagnosis_predictions = predictor.get_prediction(audio_data_in, sr_in, length_in)
    #
    #
    # json_diagnosis_predictions = jsonable_encoder(list(diagnosis_predictions))

    return {"predictions": "Healthy"}


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
