from flask import Flask, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/predict/<anxiety_level>/<self_esteem>/<mental_health_history>/<depression>/<headache>/<blood_pressure>/<sleep_quality>/<breathing_problem>/<noise_level>/<living_conditions>/<safety>/<basic_needs>/<academic_performance>/<study_load>/<teacher_student_relationship>/<future_career_concerns>/<social_support>/<peer_pressure>/<extracurricular_activities>/<bullying>", methods=['GET'])
def hello_world(anxiety_level, self_esteem, mental_health_history, depression, headache, blood_pressure, sleep_quality, breathing_problem, noise_level, living_conditions, safety, basic_needs, academic_performance, study_load, teacher_student_relationship, future_career_concerns, social_support, peer_pressure, extracurricular_activities, bullying):
    input_data = pd.DataFrame({
        'anxiety_level' : [anxiety_level], 
        'self_esteem' : [self_esteem], 
        'mental_health_history' : [mental_health_history], 
        'depression' : [depression], 
        'headache' : [headache], 
        'blood_pressure' : [blood_pressure], 
        'sleep_quality' : [sleep_quality], 
        'breathing_problem' : [breathing_problem], 
        'noise_level' : [noise_level], 
        'living_conditions' : [living_conditions], 
        'safety' : [safety], 
        'basic_needs' : [basic_needs], 
        'academic_performance' : [academic_performance], 
        'study_load' : [study_load], 
        'teacher_student_relationship' : [teacher_student_relationship], 
        'future_career_concerns' : [future_career_concerns], 
        'social_support' : [social_support], 
        'peer_pressure' : [peer_pressure], 
        'extracurricular_activities' : [extracurricular_activities], 
        'bullying' : [bullying]
    })
    
    try:
        predicted = model.predict(input_data)
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
    
    return jsonify(int(predicted[0]))

if __name__ == "__main__":
    app.run()