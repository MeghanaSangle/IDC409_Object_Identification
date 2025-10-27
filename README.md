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

    
# Training Dataset
- The link to the training dataset used: https://drive.google.com/drive/folders/1dMQ95eQCP_F1Kr5xR8oxYC1oDdCN0xUH?usp=drive_link
- The link to the saved results: https://drive.google.com/drive/folders/1EjjJSjihbziVgNeMTK2fmw3UhO9Dz6nX?usp=drive_link

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
