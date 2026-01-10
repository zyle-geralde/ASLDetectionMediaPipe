<div style="text-align:center;">
  
  <h1 style="color:#0e75b6;">🤟 ASL Hand Landmark Detection & Classification</h1>
  <h3>Detects hand landmarks from ASL gestures using MediaPipe and classifies them with Random Forest</h3>
</div>

<div class="section">
  <h2>🌟 Features</h2>
  <ul>
    <li>Hand landmark detection using <b>MediaPipe Hands</b> (21 normalized 3D landmarks per hand)</li>
    <li>Automatic <b>left-to-right hand conversion</b> for consistent landmark representation</li>
    <li>Landmark normalization for <b>position, scale, and depth invariance</b></li>
    <li>Random Forest classifier for ASL alphabet recognition</li>
    <li>Evaluation using accuracy and detailed classification reports</li>
    <li>Real-time ASL recognition from webcam input</li>
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
  <h2>📊 DataSet Used</h2>
  <h4>SignAlphaSet (Mendeley Data)</h4>
  <ul>
    <li>Total of 26,000 images of ASL alphabet, only used 600 images each letter for model training</li>
    <li><span><div>Dataset Link: </div><a>https://data.mendeley.com/datasets/8fmvr9m98w</a></span></li>
  </ul>
  
</div>

<div class="section">
  <h2>📂 Dataset Structure</h2>
  <h4>Training dataset</h4>
  <pre>
SignAlphaSet/
├── A/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── B/
│   ├── img1.jpg
│   └── ...
├── C/
│   └── ...
  </pre>
  <h4>Test dataset</h4>
    <pre>
SignAlphaSet_Sampled/
├── A/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── B/
│   ├── img1.jpg
│   └── ...
├── C/
│   └── ...
  </pre>
  <p>Each folder represents a gesture label (A, B, C...) and contains images of that gesture.</p>
</div>

<div class="section">
  <h2>🚀 Usage</h2>

  <h3>1. Dataset Sampling</h3>
  <pre><code>python TrainTestSplit.py</code></pre>
  <ul>
    <li>Randomly samples a fixed number of images per ASL letter</li>
    <li>Creates a balanced dataset for testing or validation</li>
    <li>Helps prevent data leakage and overfitting</li>
  </ul>

  <h3>2. Preprocessing & Landmark Extraction</h3>
  <pre><code>python Preprocessing.py</code></pre>
  <ul>
    <li>Loads ASL images from the dataset directory</li>
    <li>Detects hands using MediaPipe (21 landmarks per hand)</li>
    <li>Automatically converts <b>left-hand landmarks to right-hand format</b></li>
    <li>Normalizes landmarks by:
      <ul>
        <li>Centering at the wrist</li>
        <li>Scaling by hand size</li>
        <li>Using relative depth (z-axis)</li>
      </ul>
    </li>
    <li>Produces position- and scale-invariant features</li>
    <li>Saves features and labels as <code>asl_landmarks_X.npy</code> and <code>asl_labels_y.npy</code></li>
  </ul>

  <h3>3. Train the Random Forest Model</h3>
  <pre><code>python Model.py</code></pre>
  <ul>
    <li>Loads normalized landmark features</li>
    <li>Trains a Random Forest classifier</li>
    <li>Evaluates performance using a held-out test set</li>
    <li>Saves the trained model as <code>asl_rf_model.pkl</code></li>
  </ul>

  <h3>4. Real-Time ASL Recognition (Webcam)</h3>
  <pre><code>python VideoImplementation.py</code></pre>
  <ul>
    <li>Captures live video from webcam</li>
    <li>Detects hand landmarks in real time</li>
    <li>Applies the same left/right hand conversion and normalization as training</li>
    <li>Predicts ASL letters regardless of hand position or handedness</li>
  </ul>
</div>

<div class="section">
  <h2>📊 Model Evaluation</h2>
  <p>The model is evaluated using a balanced test set with normalized landmarks:</p>
  <pre>
Accuracy: 0.9902

Classification Report:
              precision    recall  f1-score   support

           A       1.00      0.91      0.95       200
           B       1.00      1.00      1.00       198
           C       1.00      1.00      1.00       190
           D       1.00      1.00      1.00       200
           E       1.00      1.00      1.00       200
           F       1.00      1.00      1.00       200
           G       1.00      1.00      1.00       200
           H       1.00      1.00      1.00       200
           I       1.00      0.92      0.96       200
           J       1.00      1.00      1.00       200
           K       1.00      1.00      1.00       200
           L       1.00      1.00      1.00       200
           M       0.99      0.98      0.99       188
           N       1.00      1.00      1.00       200
           O       1.00      1.00      1.00       141
           P       1.00      1.00      1.00       200
           Q       1.00      1.00      1.00       200
           R       1.00      0.95      0.98       200
           S       0.98      0.99      0.99       200
           T       0.91      1.00      0.95       200
           U       0.96      1.00      0.98       200
           V       1.00      1.00      1.00       200
           W       1.00      1.00      1.00       200
           X       1.00      1.00      1.00       200
           Y       0.92      1.00      0.96       200
           Z       1.00      1.00      1.00       200

    accuracy                           0.99      5117
   macro avg       0.99      0.99      0.99      5117
weighted avg       0.99      0.99      0.99      5117
  </pre>
  <p>
    High accuracy is achieved due to landmark normalization and consistent
    left/right hand representation across training and inference.
  </p>
</div>

<div class="section">
  <h2>📄 License</h2>
  <p>This project is <b>MIT Licensed</b> — free for educational and research purposes.</p>
</div>

