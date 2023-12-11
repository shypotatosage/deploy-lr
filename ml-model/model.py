import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('../ml-model/StressLevelDataset.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)

rf = RandomForestClassifier(n_estimators=30)
rf.fit(X_train, y_train)
pickle.dump(rf, open('../deploy-lr-project/model.pkl', 'wb'))