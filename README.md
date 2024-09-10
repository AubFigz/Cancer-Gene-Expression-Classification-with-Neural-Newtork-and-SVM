Download data.csv here, please:

https://drive.google.com/drive/folders/1t2bmOHzVUtW9L5g_Vklsoxm0y7hRvdZO?usp=sharing

Gene Analysis with Neural Networks and Support Vector Machine Models
Project Overview
This project, titled Gene Analysis with Neural and Support Vector Machine Models, aims to analyze gene expression data using machine learning models, specifically a Neural Network (NN) and Support Vector Machine (SVM). The dataset consists of gene expression values from various cancer types, and the goal is to classify these samples into different classes (e.g., PRAD, LUAD, BRCA, KIRC, COAD). The project leverages PCA (Principal Component Analysis) for dimensionality reduction and two machine learning models: a multi-layer neural network and a support vector machine. The results are evaluated using metrics like accuracy, confusion matrices, and classification reports.

Key Features
Data Preprocessing:

Reads gene expression data and labels, maps class labels to numerical values, and splits the data into training and test sets.
Standardizes the data using StandardScaler for normalization.
Dimensionality Reduction with PCA:

Performs Principal Component Analysis (PCA) to reduce the dimensionality of the gene expression data, retaining only the most informative features.
Plots the explained variance ratio to visualize how much variance is captured by each principal component.
Neural Network (NN) Model:

Builds and trains a multi-layer neural network with L2 regularization using Keras.
The model is trained on the PCA-transformed data and evaluated on a test set.
Prints the confusion matrix, classification report, and accuracy results for performance evaluation.
Support Vector Machine (SVM) Model:

Implements an SVM with a linear kernel to classify the PCA-transformed gene expression data.
Uses cross-validation to evaluate the model and prints the confusion matrix and classification report for performance assessment.
Plotting and Visualization:

Combines and visualizes results, such as the PCA explained variance, neural network loss over epochs, and confusion matrices for both models.
Requirements
To run this project, you need the following Python libraries:

Keras (for Neural Networks):

Copy code
pip install keras
Scikit-learn (for SVM, PCA, and metrics):

Copy code
pip install scikit-learn
Pandas (for data handling):

Copy code
pip install pandas
Matplotlib and Seaborn (for plotting):

Copy code
pip install matplotlib seaborn
NumPy (for numerical computations):

Copy code
pip install numpy
Google Colab: If using Google Colab, ensure that your Google Drive is mounted for data file access.

Project Structure
Modules and Classes
DataProcessor Class:

Reads and preprocesses the gene expression data and class labels.
Splits the data into training and test sets, and standardizes it for model input.
PCAAnalyzer Class:

Performs Principal Component Analysis (PCA) on the standardized gene expression data.
Reduces the data's dimensionality to a specified number of components for more efficient model training and evaluation.
NNModel Class:

Implements a fully connected neural network using Keras.
Trains and evaluates the neural network on the PCA-transformed data.
Prints a classification report and generates a confusion matrix.
SVMModel Class:

Implements a support vector machine (SVM) with a linear kernel using scikit-learn.
Trains the SVM on the PCA-transformed data.
Evaluates the model using cross-validation and generates a confusion matrix.
Plotter Class:

Plots combined results, including PCA explained variance, neural network loss, and confusion matrices for both models.
Key Methods
DataProcessor._prepare_data(): Prepares the data by mapping class labels to numerical values, splitting the data into training and test sets, and scaling the features.

PCAAnalyzer._perform_pca(): Performs dimensionality reduction on the scaled data, retaining a specified number of components.

NNModel.train(): Trains the neural network model on the PCA-transformed training data.

SVMModel.train(): Trains the SVM model and evaluates its accuracy using cross-validation.

Plotter.plot_combined_results(): Plots the results, including PCA explained variance, neural network loss curves, and confusion matrices for both models.

Usage
Steps to Run the Project:
Prepare Data:

Ensure that the gene expression data (data.csv) and the corresponding labels (labels.csv) are available in the same directory or accessible via Google Drive.
Run the Script:

Mount Google Drive if using Colab:
python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Set the data_path and labels_path to the appropriate locations.
Instantiate and use the classes to process data, run PCA, and train the models.
python
Copy code
data_path = 'drive/My Drive/ColabProjects/data.csv'
labels_path = 'drive/My Drive/ColabProjects/labels.csv'

data_processor = DataProcessor(data_path, labels_path)
pca_analyzer = PCAAnalyzer(data_processor)

# Train Neural Network model
nn_model = NNModel(pca_analyzer, data_processor.y_train, data_processor.y_test, l2_coeff=0.001)
nn_model.train()
nn_model.evaluate()

# Train SVM model
svm_model = SVMModel(pca_analyzer, data_processor.y_train, data_processor.y_test)
svm_model.train()
svm_model.evaluate()

# Print results
nn_model.print_results()
svm_model.print_results()

# Plot combined results
Plotter.plot_combined_results(pca_analyzer.pca, nn_model.history, nn_model, svm_model)
Example Output:
Neural Network Model:

yaml
Copy code
NNModel Test Loss: 0.0231
NNModel Test Accuracy: 93.45%
NNModel Classification Report:
  PRAD      0.94    0.92    0.93       34
  LUAD      0.90    0.93    0.91       27
  BRCA      0.95    0.96    0.95       24
  KIRC      0.92    0.91    0.91       22
  COAD      0.94    0.94    0.94       29
SVM Model:

yaml
Copy code
SVMModel Test Accuracy: 91.23%
SVMModel Classification Report:
  PRAD      0.92    0.91    0.91       34
  LUAD      0.88    0.92    0.90       27
  BRCA      0.90    0.89    0.89       24
  KIRC      0.89    0.91    0.90       22
  COAD      0.91    0.89    0.90       29
Data Visualization:
The script generates plots that show:

PCA Explained Variance.
Neural Network Loss Curve (Training and Validation Loss).
Confusion Matrices for both Neural Network and SVM models.
Conclusion
This project demonstrates how to process gene expression data and classify samples using machine learning models. The combination of PCA for dimensionality reduction and the use of a neural network and SVM models enables robust gene classification with strong accuracy. The results provide insights into the performance of both models through confusion matrices and classification reports.

Future Enhancements
Hyperparameter Tuning: Implement grid search or use Optuna for hyperparameter optimization of the neural network and SVM models.

Model Ensemble: Combine the predictions of the neural network and SVM models using an ensemble method to improve accuracy.

Additional Models: Implement and compare other machine learning models such as Random Forests or Gradient Boosting.
