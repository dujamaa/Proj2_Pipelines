# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to import the messages and categories csv data sets.
        Args: 
            messages_filepath: location of messages csv dataset file
            categories_filepath: location of categories csv dataset file        
        Returns: 
            df: data frame of merged messages and category data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    return df


def clean_data(df):
    """Function to clean data by creating 36 individual category columns
        with numeric values, and removing duplicates        
        Args: 
            df: data frame of merged messages and category data        
        Returns: 
            df: cleaned data frame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories
    category_colnames = row.str.slice(stop=-2)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # set each value to be the last character of the string
    # convert column from string to numeric
    for column in categories:    
        categories[column] = categories[column].str.slice(start=-1)    
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1, join='inner', sort=False)

    # drop duplicates
    df.drop_duplicates(inplace=True)    
    
    return df


def save_data(df, database_filename):
    """Function to load data frame as table in SQLite database        
        Args: 
            df: cleaned merged message and category data frame 
            database_filename: location of SQLite database
        Returns: 
            None
    """  
    dialectobject = 'sqlite:///'
    do_db = dialectobject + database_filename
    engine = create_engine(do_db)
    df.to_sql('MessageCats', con = engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()