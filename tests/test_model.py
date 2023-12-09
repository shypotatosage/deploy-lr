import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model.model import train_model
from ml_model.predict import predict
import pandas as pd
import pytest

def test_train_model():
    # Add a test for model training
    df_train = pd.DataFrame({
        'anxiety_level': [14, 15, 16],
        'self_esteem': [20, 21, 22],
        'self_esteem' : [20], 
    'mental_health_history' : [0], 
    'depression' : [11], 
    'headache' : [2], 
    'blood_pressure' : [1], 
    'sleep_quality' : [2], 
    'breathing_problem' : [4], 
    'noise_level' : [2], 
    'living_conditions' : [3], 
    'safety' : [3], 
    'basic_needs' : [2], 
    'academic_performance' : [3], 
    'study_load' : [2], 
    'teacher_student_relationship' : [3], 
    'future_career_concerns' : [3], 
    'social_support' : [2], 
    'peer_pressure' : [3], 
    'extracurricular_activities' : [3], 
        'bullying': [2, 3, 2],
        'stress_level': [1, 2, 1]  # Assuming stress level as the target variable
    })

    model = train_model(df_train)
    assert model is not None

def test_predict():
    # Add a test for model prediction
    input_data = pd.DataFrame({
        'anxiety_level' : [14], 
    'self_esteem' : [20], 
    'mental_health_history' : [0], 
    'depression' : [11], 
    'headache' : [2], 
    'blood_pressure' : [1], 
    'sleep_quality' : [2], 
    'breathing_problem' : [4], 
    'noise_level' : [2], 
    'living_conditions' : [3], 
    'safety' : [3], 
    'basic_needs' : [2], 
    'academic_performance' : [3], 
    'study_load' : [2], 
    'teacher_student_relationship' : [3], 
    'future_career_concerns' : [3], 
    'social_support' : [2], 
    'peer_pressure' : [3], 
    'extracurricular_activities' : [3], 
    'bullying' : [2]
    })

    model = train_model(pd.DataFrame())  # You might want to use a trained model here
    prediction = predict(model, input_data)
    assert prediction is not None


