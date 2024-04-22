# Development of a NLP model for Sentiment Analysis using LSTMs

## 1. Project Description
This project focuses on the development of a NLP model for sentiment analysis task. The developed model uses LSTMs and RNNs architecture. 

The dataset used is the Amazon product reviews dataset publicly available from the following S3 location: 

"s3://dlai-practical-data-science/data/raw/womens_clothing_ecommerce_reviews.csv"

NOTE: Since the data preparation and balancing the dataset contains many steps that rquire more explanation, A data preparation notebook is also added. So, in case you need more explanation or details about the data preparation you could go through the notebook. Otherwise, you could just simply run the data_praparation.py script. 

A sample model is trained on the data and is saved in the "models" folder. The LSTM model subdirectory in this folder, contains the pytorch model artifact, the model training info and the classification report of the model on the test set. The sample trained model shows a 83.3% overall accuracy on the test set. However, higher accuracies could be achived by hyperparameter tuning. 
The specific details of the trained model are as follows and the data used could be found in the data directory: 

- Size of the embedding vector for each token : 20

- Size of the lstm output : 20
- Whether to run a bidirectional LSTM : True
- Number of LSTM layers : 1
- LSTM dropout : 0
- Learning rate for trianing the model : 0.001
- Number of epochs to run : 100
- Threshold for positive and negative labels : 0.5
- Number of batches to use for each parameter update : 256

## 2. Tech Stack
 - Python
 - Pytorch
 - AWS CLI

## 3. How to run the project: 
Before running this project. please consider the following points: 
- Install the project packages using the requirements.txt file.
- Make sure you have AWS CLI installed on your machine.
- To process the data, train the model and then make some inferences from the model, run the <b>main.py</b> script from within the src directory. 
<b>NOTE: you must run the main script from within the src directory, many of the scripts use relative paths which could lead to errors</b>

A list of the input parameters could be viewed in the main.py script.

## 4. Project File Struture:

- src: Contains project codes and scripts. 
    - <b>main.py</b>: Preprocesses the data, trains a model, saves the model and then uses the last trained model to make some predictions. 
    - <b>data_preparation.py</b>: Preprocesses the data based on the inputs. 
    - <b>data_preparation_notebook.py</b>: A notebook with more explanations about the data preparation script. 
    - <b>inferecne.py</b>: Could be used to make predictions using the last trained model. 
    - <b>training.py</b>: Script for training the model. 

- <b>models</b>: Model artifacts are saved in this directory. Each trained model is saved as a subdirectory. Each subdirectory contains the saved Pytorch model object, Information about the training, such as losses and accuracies during epochs and a JSON file containing the classification report of the model on the test set. 

- <b>data</b>: Contains the raw data and processed data for training and inference of the model. 

- <b>requirements.txt</b>: The requirements file containing the required packages for using the project.    

