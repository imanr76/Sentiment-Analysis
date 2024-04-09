from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download("punkt")
nltk.download('wordnet')
from data_preparation import process_data
from training import train_model
from inference import predict

#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    # Defining the reuqired parameters
    max_len = 500
    train_size = 0.8
    validation_size = 0.15
    test_size = 0.05
    embed_dim = 10
    lstm_size = 5
    bidirectional = True
    num_layers = 1
    dropout = 0
    learning_rate = 0.01
    epochs = 100
    threshold = 0.5
    
    print("\nData processing started\n")
    process_data(max_len, train_size, validation_size, test_size)
    
    print("Data processing finished, started training the model\n")
    report, model = train_model(embed_dim, lstm_size, bidirectional, num_layers, dropout, learning_rate, epochs, threshold)
    
    print("\nModel trained and saved, predicting some reviews\n")
    reviews =  ["I hate it", "it was too loose" ,"This is a nice computer, I love using it and i will recommend you to buy it as soon as possible"]
    predict(model, word_tokenize, WordNetLemmatizer(), reviews, threshold, max_len)
