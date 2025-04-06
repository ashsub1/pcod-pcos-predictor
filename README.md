PCOD/PCOS Predictor
A machine learning-powered application that predicts the likelihood of Polycystic Ovarian Disease (PCOD) and Polycystic Ovarian Syndrome (PCOS) based on patient health parameters. This tool is designed to assist early diagnosis and guide individuals toward medical consultation.

🚀 Features
Predicts both PCOD and PCOS conditions individually.

Pre-trained ML models using relevant clinical features.

Simple Python-based interface (app.py).

Serialized models for direct inference using .pkl files.

Easily customizable for integration with web or desktop apps.

🧠 Machine Learning Workflow
Data Collection & Cleaning
Medical data relevant to PCOD and PCOS symptoms was collected and cleaned for model training.

Feature Selection
Important clinical and physiological features were selected and stored in:

pcod_features.pkl

pcos_features.pkl

Model Training
Separate models were trained for PCOD and PCOS. Algorithms used may include Logistic Regression, Random Forest, or similar classification models (details inside training notebook if available).

Serialization
Trained models were saved using Python’s pickle module:

pcod_model.pkl

pcos_model (2).pkl

Prediction Interface
The app.py script loads the models and accepts user input to predict PCOD/PCOS likelihood.

🗂️ Repository Structure
bash
Copy
Edit
├── app.py                   # Main script to run prediction
├── pcod_model.pkl           # Trained model for PCOD prediction
├── pcos_model (2).pkl       # Trained model for PCOS prediction
├── pcod_features.pkl        # Features used for PCOD model
├── pcos_features.pkl        # Features used for PCOS model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
