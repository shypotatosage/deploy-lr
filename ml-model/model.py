import pandas as pd 
from sklearn.preprocessing import RobustScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('StressLevelDataset.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = RobustScaler()
X_train = sc.fit_transform(X_train)

logreg = svm.SVC(kernel='linear')
logreg.fit(X_train, y_train)
pickle.dump(logreg, open('../deploy-lr-project/model.pkl', 'wb'))

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

predicted = logreg.predict(input_data)

print(predicted[0])