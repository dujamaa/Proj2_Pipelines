# Disaster Response Pipeline Project

### Table of Contents
1. [Project Motivation](#Project Motivation)
2. [Installation](#Installation)
3. [File Descriptions](#File-Descriptions)
4. [Instructions](#Instructions)
5. [Results](#Results)
6. [Acknowledgements](#Acknowledgements)

### Project Motivation
This project is being completed as one of the requirements for Udacity's Data Science Nanodegree program.  The objective of this project is to develop an ETL pipeline from disaster data provided by [Figure Eight](https://www.figure-eight.com/) that feeds into a machine learning pipeline to build a model for a web app that classifies disaster messages.    

### Installation
This project uses the following libraries for the Anaconda distribution of Python version 3.6+:
* re (Regular expression operations)
* numpy
* pandas
* pickle
* nltk (Natural Language Tool Kit)
* sklearn
* sqlalchemy

### File Descriptions


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Results

### Acknowledgements
* [Udacity](https://www.udacity.com/) is acknowledged for assigning this project.
* [Figure Eight](https://www.figure-eight.com/) is acknowledged for providing the data for this project.
