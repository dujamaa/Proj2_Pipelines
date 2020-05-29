# Disaster Response Pipeline Project

### Table of Contents
1. [Project Motivation](#Project-Motivation)
2. [Installation](#Installation)
3. [File Descriptions](#File-Descriptions)
4. [Instructions](#Instructions)
5. [Results](#Results)
6. [Acknowledgements](#Acknowledgements)

### Project Motivation
This project is being completed as part of the requirements for Udacity's Data Science Nanodegree program.  The objective of this project is to develop an ETL pipeline from disaster data provided by [Figure Eight](https://www.figure-eight.com/) that feeds into a machine learning pipeline to build a model for a web app that classifies disaster messages.    

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
* app - folder containing files for the web app
    * templates - folder containing html code for the web app
        * master.html - main page of web app
        * go.html - classification result page of web app
    * run.py - flask file that run web app
* data - folder containing data files and ETL pipeline
    * process_data.py - ETL pipeline python code
    * disaster_categories.csv - data to process
    * disaster_messages.csv - data to process
    * DisasterResponse.db - database containing clean data
* models - folder containing machine learning files
    * train_classifier.py - machine learning pipeline python code
    * classifier.pkl - final model exported as a pickle file
* ETL Pipeline Preparation.ipynb - Jupyter notebook to perform preliminary coding and ananlysis for the ETL pipeline process_data.py file.
* ML Pipeline Preparation.ipynb - Jupyter notebook to perform preliminary coding and exploratory analysis before refactoring into the train_classifier.py file.

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
Running process_data.py and train_classifier.py will print results to the terminal screen.  The screen below will be displayed for 5 to 10 minutes while the model is building and training:
![Figure1](https://raw.githubusercontent.com/dujamaa/Proj2_Pipelines/master/images/screenshot1.png)

After the model has completed training, the f1 score, precision, and recall for each output category is printed to the terminal screen and the model is saved as shown in the screenshot below:
![Figure2](https://raw.githubusercontent.com/dujamaa/Proj2_Pipelines/master/images/screenshot2.png)

After the web app starts by running run.py is run in the app directory, the web app will display the following three visualizations: 1) Distribution of Message Genres, 2) Distribution of Message Genres with no Categorization, and 3) Distribution of Messages per Category.
![Figure3](https://raw.githubusercontent.com/dujamaa/Proj2_Pipelines/master/images/screenshot3.png)
![Figure4](https://raw.githubusercontent.com/dujamaa/Proj2_Pipelines/master/images/screenshot4.png)

New messages can be entered in the text box to classify the message using the model as shown in the screenshots below:
![Figure5](https://raw.githubusercontent.com/dujamaa/Proj2_Pipelines/master/images/screenshot5.png)
![Figure6](https://raw.githubusercontent.com/dujamaa/Proj2_Pipelines/master/images/screenshot6.png)

### Acknowledgements
* [Udacity](https://www.udacity.com/) is acknowledged for assigning this project.
* [Figure Eight](https://www.figure-eight.com/) is acknowledged for providing the data for this project.
