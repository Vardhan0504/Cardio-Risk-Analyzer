import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle 

df = pd.read_csv('heart.csv')
df = df.drop_duplicates()
df['thall'] = df['thall'].replace(0, 2)
df.columns = ['Age' , 'Sex' , 'Chest Pain Type' , 'Resting Blood Pressure' , 'Serum Cholestrol' , 'Fasting Blood Sugar' , 'Resting Electrocardiographic Results' , 'Maximum Heart Rate Achieved' , 'Exercise Induced Angina' , 'ST Depression Induced by Exercise Relative to Rest' , 'The Slope of The Peak Exercise ST Segment' , 'Number of Major Vessels Colored by Flourosopy' , 'Thallium Stress Test' , 'Output']

# Split data
X = df.drop('Output', axis=1)
y = df['Output']



model = RandomForestClassifier()

model.fit(X , y)

pickle.dump(model , open('model.pkl' , 'wb'))

# train_acc, test_acc, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)