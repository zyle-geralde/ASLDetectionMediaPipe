Dataset: https://www.kaggle.com/datasets/ayuraj/asl-dataset
<div style="text-align:center;">
  
  <h1 style="color:#0e75b6;">🤟 ASL Hand Landmark Detection & Classification</h1>
  <h3>Detects hand landmarks from ASL gestures using MediaPipe and classifies them with Random Forest</h3>
</div>

<div class="section">
  <h2>🌟 Features</h2>
  <ul>
    <li>Hand landmark detection using <b>MediaPipe Hands</b> (21 3D landmarks per hand)</li>
    <li>Data augmentation: rotation, scaling, brightness adjustments</li>
    <li>Random Forest classifier for ASL gesture recognition</li>
    <li>Model evaluation with accuracy & classification report</li>
  </ul>
</div>

<div class="section">
  <h2>🛠️ Technologies Used</h2>
  <ul>
    <li>Python</li>
    <li>OpenCV</li>
    <li>Scikit-Learn</li>
    <li>medipipe</li>
  </ul>
</div>

<div class="section">
  <h2>📂 Dataset Structure</h2>
  <pre>
asl_dataset/
├── a/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── b/
│   ├── img1.jpg
│   └── ...
├── c/
│   └── ...
  </pre>
  <p>Each folder represents a gesture label (A, B, C...) and contains images of that gesture.</p>
</div>

<div class="section">
  <h2>🚀 Usage</h2>
  <h3>1️⃣ Preprocessing & Landmark Extraction</h3>
  <pre><code>python Preprocessing.py</code></pre>
  <ul>
    <li>Loads images from dataset</li>
    <li>Applies augmentations (rotation, scaling, brightness)</li>
    <li>Extracts hand landmarks (21 per hand)</li>
    <li>Saves features and labels as <code>asl_landmarks_X.npy</code> and <code>asl_labels_y.npy</code></li>
  </ul>

  <h3>2️⃣ Train the Random Forest Model</h3>
  <pre><code>python Model.py</code></pre>
  <ul>
    <li>Loads `.npy` files</li>
    <li>Splits data into train/test sets</li>
    <li>Trains Random Forest classifier</li>
    <li>Evaluates performance</li>
    <li>Saves trained model as <code>asl_rf_model.pkl</code></li>
  </ul>
</div>

<div class="section">
  <h2>📊 Model Evaluation</h2>
  <p>Accuracy and classification report per gesture:</p>
  <pre>
Accuracy: x.xx
Classification Report:
              precision    recall  f1-score   support
A               x.xx       x.xx      x.xx      xxx
B               x.xx       x.xx      x.xx      xxx
C               x.xx       x.xx      x.xx      xxx
  </pre>
</div>

<div class="section">
  <h2>🔧 Data Augmentation Techniques</h2>
  <ul>
    <li>Rotation</li>
    <li>Scaling</li>
    <li>Brightness</li>
  </ul>
</div>

<div class="section">
  <h2>📄 License</h2>
  <p>This project is <b>MIT Licensed</b> — free for educational and research purposes.</p>
</div>

