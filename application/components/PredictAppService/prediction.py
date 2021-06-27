from .treadsafeSingleton import SingletonMeta
from .preprocessor_pipeline import AudioPreProcessor

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import numpy as np
import os

URL_DUAL_PREDICTION_MODEL_PATH = "/content/testapi/application/components/static/ml_models/Diagnosis_Prediction_v5.h5"
URL_DISEASE_CLASS_LABELS_PATH = "/content/testapi/application/components/static/label_encodings/diagnosis_classes_v5.npy"


class PredictionService:

    def __init__(self):
        self.pre_processor = AudioPreProcessor()
        self.prediction_model = load_model(URL_DUAL_PREDICTION_MODEL_PATH)
        self.disease_class_encoder = LabelEncoder()
        self.disease_class_encoder.classes_ = np.load(URL_DISEASE_CLASS_LABELS_PATH, allow_pickle=True)

    def vote_average_logits(self, predictions: np.ndarray) -> np.ndarray:
        """
        Averaging on log-odds of probabilities

        Returns
        -------
          log-odds of probabilities
        """
        logits = np.log(predictions / (1 - predictions))
        avg = np.mean(logits, axis=0)
        predictions = 1 / (1 + np.exp(-avg))
        return predictions

    def label_encoder(self, disease_predictions: np.ndarray) -> str:
        """
          Get Prediction class label

        Parameters
        ----------
        disease_predictions

        Returns
        -------

        """
        average_logits = self.vote_average_logits(disease_predictions)
        label_index = np.argmax(average_logits, axis=0)
        prediction_class_label = self.disease_class_encoder.inverse_transform([label_index])
        return prediction_class_label

    def get_prediction(self, audio_data_in, sr_in, length_in) -> str:
        """
          Get prediction all the frames of audio

        Parameters
        ----------
        audio_path
        sr

        Returns
        -------
           disease class prediction
        """
        feature_engineered_data = self.pre_processor.get_pre_processed_data_dual_model(audio_data_in, sr_in, length_in)
        predictions = self.prediction_model.predict(
            {'mel-spectral-input': feature_engineered_data[0],
             'mel-spectral2-input': feature_engineered_data[1]})

        prediction_class_label = self.label_encoder(predictions)

        return prediction_class_label


class Predictor(metaclass=SingletonMeta):

    def __init__(self):
        self.predict = PredictionService()

    def get_predictor(self):

        return self.predict

# a = PredictionService()
# pred= a.get_prediction('/home/sachitha/Documents/Projects/smarth_sthethoscope/SmartStethoscope/media/101_1b1_Al_sc_Meditron.wav')
#
# print(pred)
