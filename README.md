# Computer Network Intrusion Detection Using Deep Learning (ANN)

### Introduction
This project aims to develop a deep learning-based intrusion detection system for computer networks using Artificial Neural Networks (ANN). The goal is to detect and classify different types of network intrusions or attacks, such as denial of service (DoS), unauthorized access (U2R), and probing (Probe), to enhance the security of computer networks.

### Dataset
The project utilizes the NSL-KDD dataset, which is an updated version of the widely used KDD Cup 1999 dataset for network intrusion detection. The dataset contains a large number of network connection records with various features, including protocol type, service, flags, duration, and more. Each connection record is labeled as either normal or belonging to a specific attack category.

### Methodology
1. **Data Preprocessing:** The dataset is preprocessed to handle missing values, encode categorical features, and normalize numerical features.
2. **Feature Selection:** Relevant features are selected to reduce dimensionality and improve model performance.
3. **Model Training:** An Artificial Neural Network (ANN) model is trained on the preprocessed dataset using TensorFlow/Keras. The model is optimized to classify network connections into normal or attack categories.
4. **Model Evaluation:** The trained model is evaluated using performance metrics such as accuracy, precision, recall, and F1-score on a separate test dataset to assess its effectiveness in detecting network intrusions.

### Results
The project presents the results of the trained ANN model in terms of its performance metrics on the test dataset. The model's ability to accurately classify network connections and detect different types of intrusions is analyzed and discussed.

### Usage
1. Clone the repository:
   ```
   git clone https://github.com/your_username/Computer_Network_Intrusion_Detection.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script to preprocess the data, train the model, and evaluate its performance:
   ```
   python main.py
   ```
