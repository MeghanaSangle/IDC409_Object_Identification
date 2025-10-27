# OBJECT IDENTIFICATION MODEL
A Python-based machine learning project for object identification using image feature extraction, CNN and OpenCV to classify and detect geometric shapes in images.   

# Overview
- This project builds and trains a **Convolutional Neural Network (CNN)** to identify and classify multiple geometric shapes from images.  
- It includes:
  - **Custom CNN:** A sequential CNN model defined for 64x64 grayscale image classification.
  - **Data Handling:** Uses Keras's `ImageDataGenerator` for data scaling, splitting (training/validation), and flow management.
  - **Contour Detection:** Includes an independent function using **OpenCV** to perform object and contour detection on external images.
  - **Comprehensive Evaluation:** Generates a detailed classification report, confusion matrix, and plots for loss/accuracy.
  - **Model Persistence:** Saves the trained model (`.keras` format) for later use.
  
# Tech Stack
- **Libraries:** TensorFlow, Keras, NumPy, Pandas, OpenCV, Matplotlib, Seaborn, Scikit-learn
  * **Deep Learning:** TensorFlow, Keras  
  * **Data Handling:** NumPy, Pandas  
  * **Visualization:** Matplotlib, Seaborn  
  * **Computer Vision:** OpenCV  
  * **Evaluation Metrics:** Scikit-learn   
- **Dataset:** Custom dataset stored in Google Drive (`/shapes_dataset`)

# Model Architecture
| Layer Type | Description |
|-------------|--------------|
| **Conv2D** | 16 filters, 3×3 kernel, ReLU activation |
| **MaxPooling2D** | 2×2 pool size |
| **Conv2D** | 32 filters, 3×3 kernel, ReLU activation |
| **MaxPooling2D** | 2×2 pool size |
| **Conv2D** | 64 filters, 3×3 kernel, ReLU activation |
| **MaxPooling2D** | 2×2 pool size |
| **Flatten** | Converts 2D feature maps to 1D vector |
| **Dense** | 128 neurons, ReLU activation |
| **Dropout** | 0.3 to prevent overfitting |
| **Output Layer** | Softmax activation for multi-class classification |


# Installation & Setup
- Generated a synthetic training dataset comprising 300 images per shape category designed specifically to introduce complexity, including random changes in size, rotation, and realistic boundary wear or damage.
- Install all the necessary libraries, ensuring compatibility for the TensorFlow version used.
- Mount the Google Drive if using Google Colab :  from google.colab import
                                                  drive.mount('/content/drive')
- Ensure your dataset is correctly placed in your Google Drive at the specified path.
    
# Dataset Structure 
- Ensure that the dataset (`shapes_dataset/`) is organized into subdirectories for each shape class.

# Project Structure

```text
├── shapes_dataset/                  # Dataset folders
├── shape_classifier_cnn.keras       # Trained CNN model
├── shapes_results/                  # Output folder for results
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── misclassified_samples.csv
│   └── misclassified_examples.png
├── main.py / train_model.py         # Training and evaluation script
└── README.md                        # Project documentation
```

# Training
- Run the training and evaluation code (`train_model.py `to Colab or `python train_model.py`)
- After training:
  * The trained model is saved as `shape_classifier_cnn.keras`
  * Evaluation outputs are saved to `/shapes_results/`
  * Plots for accuracy/loss vs epochs are displayed automatically
 
# Results & Evaluation
- `Predicted: Circle (98.45%)`\
  `Validation accuracy: 94.27%`
- Metrics are generated : 
  * Training vs validation accuracy/loss curves
  * Confusion matrix visualization
  * Precision, Recall, and F1-score per class
  * Misclassified samples report with confidence levels
