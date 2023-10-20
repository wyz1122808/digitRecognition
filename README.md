# digitRecognition


# Handwritten Digit Classification with AlexNet

This repository provides a deep learning model to classify handwritten digits using the popular AlexNet architecture.

## Libraries Used:
- numpy
- pandas
- matplotlib
- sklearn
- keras (from tensorflow)

## Dataset:

The dataset used for this project consists of handwritten digits and is split into training and test sets. These datasets can be downloaded using the following links:

- [Train Data](https://drive.google.com/uc?id=1zDelOAtkwghIf-5eNX5XerEGp2viEY60)
- [Test Data](https://drive.google.com/uc?id=1SXCGAzKAKG8RMTe_TSmESdyEFxiUx6d-)

## How the Code Works:

1. **Importing Necessary Libraries**: All the required packages for data manipulation, visualization, and modeling are imported.

2. **Loading the Data**: The datasets are loaded directly from Google Drive using the `gdown` command.

3. **Data Exploration**: Basic EDA is performed to check the shape of the datasets and visualize the distribution of different labels.

4. **Preprocessing**: The images in the dataset are reshaped to the required format and labels are separated from the training dataset.

5. **Model**: AlexNet, a convolutional neural network architecture, is used. This network contains multiple convolutional layers, max-pooling layers, and dense layers. Dropout is also used to prevent overfitting.

6. **Training**: The model is trained using the training dataset with early stopping as a callback to prevent overfitting. Training performance is also evaluated on a validation set.

7. **Summary**: After training, a summary of the model architecture is printed.

## Results:

The model shows promising results with a high validation accuracy after a certain number of epochs.

## To Run the Code:

Ensure that you have all the mentioned libraries installed. Clone the repository and run the provided code in a Jupyter notebook or Google Colab environment. Ensure you have internet access to download the datasets directly.

## Future Work:

- Fine-tuning the model with different architectures or hyperparameters for improved accuracy.
- Exploring data augmentation techniques.
- Implementing more visualization tools for better data understanding.
