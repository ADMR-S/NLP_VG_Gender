# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score  # For model evaluation
from sklearn.preprocessing import MinMaxScaler  # For data scaling
import matplotlib.pyplot as plt  # For plotting graphs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras import regularizers  # For regularization
from scikeras.wrappers import KerasRegressor  # For using Keras with scikit-learn
import joblib  # For saving the model
import sys

data_file_name = 'data_clean.csv'
window_size = 5  # Time-series window (how many days are taken into account for each prediction)
shuffle = False  # Whether to shuffle the data before splitting
recall = False  # Whether to recall the last best parameters for the same dataset
loss = 'mean_squared_error'  # Loss function for the model

#-------------------PLOTS-------------------

plt.rcParams['figure.figsize'] = (14, 7) #set global matplotlib plot size

def plot_loss(epochs, loss, val_loss, filename):
    plt.figure()
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss During Training')
    plt.legend()
    plt.savefig(filename)  # Save the plot as a PNG file

def plot_predictions(data, y, y_pred, testingOrTraining):
    plt.figure()
    plt.scatter(data.index[:len(y)], y, label='Original ' + testingOrTraining, s=10)
    plt.scatter(data.index[:len(y)], y_pred, label='Predicted ' + testingOrTraining, s=10)
    plt.xlabel('Date')
    plt.ylabel('Scaled Price')
    plt.title('BTC Price Prediction - ' + testingOrTraining + ' Set')
    plt.legend()
    plt.xticks(data.index[:len(y)][::365], data.index[:len(y)][::365].strftime('%Y'), rotation=90)  # Show only years at the beginning of each year
    plt.savefig(f'{data_file_name}_price_prediction_{testingOrTraining}_shuffle{shuffle}_window_size{window_size}.png')  # Save the plot as a PNG file

def plot_predictions_merged(data, y_train, y_pred_train, y_test, y_pred_test, filename):
    plt.figure()
    plt.plot(data.index[:len(y_train)], y_train, label='Original Train')
    plt.plot(data.index[:len(y_train)], y_pred_train, label='Predicted Train')
    plt.plot(data.index[len(y_train):len(y_train) + len(y_test)], y_test, label='Original Test')
    plt.plot(data.index[len(y_train):len(y_train) + len(y_test)], y_pred_test, label='Predicted Test')
    plt.xlabel('Date')
    plt.ylabel('Scaled Price')
    plt.title('BTC Price Prediction - Training and Testing Sets')
    plt.legend()
    plt.xticks(data.index[::365], data.index[::365].strftime('%Y'), rotation=90)  # Show only years at the beginning of each year
    plt.savefig(filename)  # Save the plot as a PNG file

def plot_mse_vs_r2(mse_train, mse_test, r2_train, r2_test) :
    # Plot MSE against R2 score
    plt.figure()
    plt.scatter(mse_train, r2_train, label='Training')
    plt.scatter(mse_test, r2_test, label='Testing')
    plt.xlabel('MSE')
    plt.ylabel('R2 Score')
    plt.title('MSE vs R2 Score')
    plt.legend()
    plt.savefig(f'{data_file_name}_mse_vs_r2_shuffle{shuffle}_window_size{window_size}.png')  # Save the plot as a PNG file

#--------------------FUNCTIONS--------------------
# Create time windows
def create_time_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Build the LSTM model with initial values that might be changed by gridSearchCV
def build_model(optimizer='adam', activation='tanh', lstm_units=50, dropout_rate=0.2, dense_layers=1):
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(units=lstm_units, activation=activation),
        Dropout(dropout_rate)
    ])
    for _ in range(dense_layers):
        model.add(Dense(1, kernel_regularizer=regularizers.L2(0.01)))
    model.compile(optimizer=optimizer, loss=loss)
    return model

def save_metrics(mse_train, mse_test, r2_train, r2_test, accuracy, precision, recall, filename):
    with open(filename, 'w') as f:
        f.write(f'Training MSE: {mse_train}\n')
        f.write(f'Testing MSE: {mse_test}\n')
        f.write(f'Training R2: {r2_train}\n')
        f.write(f'Testing R2: {r2_test}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')

def find_and_save_best_params_with_gridSearchCV(model, X_train, y_train, X_val, y_val) :

    # Define the hyperparameters for GridSearchCV
    params = {
        'model__optimizer': ['adam', 'rmsprop'],
        'model__activation': ['sigmoid', 'tanh'],
        'model__lstm_units': [50, 100],
        'model__dropout_rate': [0.1, 0.2, 0.3],
        'model__dense_layers': [1, 3, 5],
        'batch_size': [24, 36 ,48]
    }
    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=params, verbose=0)
    grid_search.fit(X_train, y_train, validation_data=(X_val, y_val))

    
    # Get the best model
    best_model = grid_search.best_estimator_

    # Save the best model parameters
    joblib.dump(grid_search.best_params_, 'best_model_params.pkl')

    # Display the best model parameters
    print("Best model parameters:", grid_search.best_params_)
    
    return best_model

def recall_previous_best_model_and_plot_loss(model, X_train, y_train, X_val, y_val) :
    # Load the best model parameters
    best_params = joblib.load('best_model_params.pkl')

    # Set the parameters to the estimator
    model.set_params(**best_params)


    # Redirect system output to a file
    sys.stdout = open('training_output.txt', 'w')

    # Train the model with the best parameters
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=30, verbose=1)

    # Restore system output
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    # Read the training output file and extract loss and validation loss
    epochs = []
    loss = []
    val_loss = []

    with open('training_output.txt', 'r') as file:
        for line in file:
            if 'Epoch' in line:
                epoch = int(line.split('/')[0].split(' ')[-1])
                epochs.append(epoch)
            if 'loss:' in line and 'val_loss:' in line:
                loss_value = float(line.split('loss: ')[1].split(' - ')[0])
                val_loss_value = float(line.split('val_loss: ')[1])
                loss.append(loss_value)
                val_loss.append(val_loss_value)

    # Plot the loss and validation loss evolution over epochs
    plot_loss(epochs, loss, val_loss, f'{data_file_name}_loss_plot_shuffle{shuffle}_window_size{window_size}.png')

def display_confusion_matrix_for_direction(y_test_classes, y_pred_test_classes):
    # Display the confusion matrix for direction prediction
    cm = confusion_matrix(y_test_classes, y_pred_test_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f'{data_file_name}_confusion_matrix_shuffle{shuffle}_window_size{window_size}.png')  # Save the confusion matrix as a PNG file

def main(recall) : 
    
    # Load the prepared data
    data = pd.read_csv(data_file_name)

    # Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Calculate the change for the day (Open - Adj Close)
    data['Change'] = data['Open'] - data['Close']

    # Rescale the data into the range [0, 1] in case it's not done
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['Open', 'High', 'Low', 'Close', 'Change']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Change']])

    X, y = create_time_windows(data['Close'].values, window_size)

    # Split the data into training and testing sets without shuffling (1/3 of the data is used for testing)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, shuffle=shuffle, random_state=42)

    # Split the remaining data into training and validation sets (20% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, shuffle=shuffle, random_state=42)

    # Wrap the model using KerasRegressor
    model = KerasRegressor(model=build_model, epochs=30, batch_size=64, verbose=1)

    if recall:
        recall_previous_best_model_and_plot_loss(model, X_train, y_train, X_val, y_val)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    else:
        best_model = find_and_save_best_params_with_gridSearchCV(model, X_train, y_train, X_val, y_val)
        # Make predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

    # Plot the original and predicted values for training set
    plot_predictions(data, y_train, y_pred_train, 'Training')


    # Plot the original and predicted values for testing set
    plot_predictions(data, y_test, y_pred_test, 'Testing')

    # Plot the original and predicted values for both training and testing sets on the same chart
    plot_predictions_merged(data, y_train, y_pred_train, y_test, y_pred_test, f'{data_file_name}_price_prediction_train_test_shuffle{shuffle}_window_size{window_size}.png')

    # Evaluate the model
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    y_test_diff = np.diff(y_test, prepend=y_test[0])
    y_pred_test_diff = np.diff(y_pred_test, prepend=y_pred_test[0])
    y_test_classes = np.where(y_test_diff > 0, 1, 0)
    y_pred_test_classes = np.where(y_pred_test_diff > 0, 1, 0)

    display_confusion_matrix_for_direction(y_test_classes, y_pred_test_classes)

    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(y_test_classes, y_pred_test_classes)
    precision = precision_score(y_test_classes, y_pred_test_classes)
    recall = recall_score(y_test_classes, y_pred_test_classes)

    # Print MSE, R2, accuracy, precision, and recall scores to a file
    save_metrics(mse_train, mse_test, r2_train, r2_test, accuracy, precision, recall, f'{data_file_name}_mse_r2_scores_shuffle{shuffle}_window_size{window_size}.txt')

    print(f'Training MSE: {mse_train}, Testing MSE: {mse_test}')
    print(f'Training R2: {r2_train}, Testing R2: {r2_test}')
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

    # Plot MSE against R2 score
    plot_mse_vs_r2(mse_train, mse_test, r2_train, r2_test)

main(recall)