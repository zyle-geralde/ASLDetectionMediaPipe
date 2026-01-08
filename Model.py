from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

x = np.load("asl_landmarks_X.npy")
y = np.load("asl_labels_y.npy")

#MODEL TRAINING
#Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Train model
random_classifier_model = RandomForestClassifier(n_estimators=100, max_depth=None,random_state=42)
random_classifier_model.fit(x_train, y_train)

#make prediction
y_pred = random_classifier_model.predict(x_test)

#Model Evaluation

print("Evaluation")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))












