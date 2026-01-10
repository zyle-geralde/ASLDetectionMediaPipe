from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

x_train = np.load("asl_landmarks_X.npy")
y_train = np.load("asl_labels_y.npy")

#MODEL TRAINING
#Split dataset
x_test = np.load("asl_landmarks_Xtest.npy")
y_test = np.load("asl_labels_ytest.npy")

#Train model
random_classifier_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42)
random_classifier_model.fit(x_train, y_train)

#Uncomment if you have not saved the model
joblib.dump(random_classifier_model, "asl_rf_model.pkl")
print("Model saved")

#make prediction
y_pred = random_classifier_model.predict(x_test)

#Model Evaluation
print("Evaluation")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))












