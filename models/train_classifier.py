# import libraries
import sys
import re
import pickle
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """Load data from SQLite database.
        Args: 
            database_filepath: SQLite database file path and name                    
        Returns: 
            X: the column of messages used to predict categories
            Y: 36 category features to predict based on messages
            category_names: names of the 36 category features
    """
    dialectobject = 'sqlite:///'
    do_db = dialectobject + database_filepath
    engine = create_engine(do_db)
    df = pd.read_sql("SELECT * FROM MessageCats", engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Normalize, clean, lemmatize, and tokenize text
        Args: 
            text: text to be normalized, cleaned, and tokenized                   
        Returns: 
            clean_tokens: cleaned and tokenized text
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
   
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:        
        # lemmatize and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Machine Learning pipeline model
        Args: 
            None                   
        Returns: 
            cv: tuned model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # tune pipeline with best parameters found in GridSearchCV
    params_custom = {'clf__estimator__min_samples_split': 2, 
                     'vect__max_df': 0.75, 'vect__min_df': 0.01, 
                     'vect__ngram_range': (1, 1)}
    
    pipeline.set_params(**params_custom)
    
    # parameters for GridSearchCV
    parameters = {
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__min_df': (0.01, 0.1, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__min_samples_split': [2, 3, 4],
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
   
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Report f1 score, precision and recall for each category
        Args: 
            model: the NLP/MLP Pipeline model 
            X_test: test split of the messages
            Y_test: test split of the categories
            category_names: names of the 36 category features
        Returns: 
            None
    """
    Y_pred = model.predict(X_test)
    Y_pred2 = pd.DataFrame(Y_pred, columns = category_names)
    
    for category in category_names:
        categoryDF = pd.DataFrame(
            classification_report(Y_test[category], Y_pred2[category], zero_division=0,output_dict=True)
        )
        print(category)
        print('    Accuracy: {}%  Precision: {}%  Recall: {}%\n'.format(
            categoryDF.loc['f1-score','weighted avg'].round(4),
            categoryDF.loc['precision','weighted avg'].round(4),
            categoryDF.loc['recall','weighted avg'].round(4))
             )


def save_model(model, model_filepath):
    """ Export model as a pickle file        
        Args: 
            model: the model to be exported 
            model_filepath: the file name and path to export the model
        Returns: 
            None
    """    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()