import torch
import os 
from training import sentiment 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download("punkt")
nltk.download('wordnet')
from data_preparation import process_reviews


def predict(model, tokenizer, lemmatizer, reviews, threshold, max_len):
    """
    Accepts some reviews nd returns their corresponding sentiments. 

    Parameters
    ----------
    model : obj
        PyTorch model.
    tokenizer : obj
        Word tokenizer.
    vocabulary : obj
        Vocabulary object.
    lemmatizer : obj
        Lemmantizer object.
    reviews : TYPE
         A list of reviews whose sentiment we want to predict.
    threshold : float
        Threshold for determining positive and negative values based on the model output.
    max_len : int
        Maximum allowed length of the input sequence.

    Returns
    -------
    predictions : TYPE
        DESCRIPTION.

    """
    with torch.no_grad():
        model.to("cpu")
        model.eval()
        # Reading the vocabulary 
        vocabulary = torch.load("./../data/vocabulary.pth")
        predictions = []
        for review in reviews:
            review_processed = process_reviews(review, tokenizer, lemmatizer, vocabulary, max_len)
            prediction = torch.where(model.sigmoid(model(torch.tensor(review_processed).reshape(1, -1))) >= threshold, torch.tensor(1), torch.tensor(0))
            if prediction == 1:
                predictions.append("Positive")
            else:
                predictions.append("Negative")
            print(f"Review : '{review}'\nPredicted Sentiment : {predictions[-1]}\n")
    return predictions

#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    
    # Setting the threshold for positive and negative labels
    threshold = 0.5
    max_len = 500
    model = torch.load(os.path.join("./../models/" + os.listdir("./../models")[-1], "model.pth"))
    # model.to("cpu")
    reviews =  ["i hated it", "it was too loose" ,"This is a nice computer, I love using it and i will recommend you to buy it as soon as possible"]

    predictions = predict(model, word_tokenize, WordNetLemmatizer(), reviews, threshold, max_len)