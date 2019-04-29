# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries

import pandas as pd
import os


def churningPrediction(path):
        
    customer_file_path = os.path.join(path, 'CustomerAccounts.csv')
    # Importing the dataset
    dataset = pd.read_csv(customer_file_path)
    X = dataset.iloc[:, 3:13].values
    
    
    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    
    
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    
    # Importing the already trained model!
    from keras.models import load_model
    model_file_path = os.path.join(path, 'model/churning_model.h5')
    model = load_model(model_file_path)
    
    
    #Making the prediction 
    y_pred = model.predict(X)
    y_predrounded = (y_pred > 0.5)
    
    
    #Saving probability in a dataframe
    yprob_df = pd.DataFrame(y_pred)
    yprob_df.columns = ['Probabilities']
    #yprob_df.to_csv('Probabilities.csv', index = False)
    
    
    #Saving prediction in a dataframe
    ypredrounded_df = pd.DataFrame(y_predrounded)
    ypredrounded_df.columns = ['Predictions']
    #ypredrounded_df.to_csv('Predictions.csv', index = False)
    
    
    dataset['Probabilities'] = yprob_df['Probabilities']
    dataset['Predictions'] = ypredrounded_df['Predictions']
    
    finalpath = os.path.join(path, 'Customer_file_with_predictions.csv')
    
    dataset.to_csv(finalpath, index=False)
    os.remove(customer_file_path) 
    

    
    















