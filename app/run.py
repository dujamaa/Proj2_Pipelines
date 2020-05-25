import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCats', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Count the number of messages in each category
    category_counts = df.astype(bool).sum(axis = 0)
    category_counts.drop(['id','message','original','genre'], inplace = True)
    category_names = list(category_counts.index)
    
    # Count the number of messages with no categoy by genre
    no_category = df[(df.related == 0) & (df.request == 0) & (df.offer == 0) & 
                 (df.aid_related == 0) & (df.medical_help == 0) & (df.medical_products == 0) & 
                 (df.search_and_rescue == 0) & (df.security == 0) & (df.military == 0) & 
                 (df.child_alone == 0) & (df.water == 0) & (df.food == 0) & (df.shelter == 0) & 
                 (df.clothing == 0) & (df.money == 0) & (df.missing_people == 0) & 
                 (df.refugees == 0) & (df.death == 0) & (df.other_aid == 0) & 
                 (df.infrastructure_related == 0) & (df.transport == 0) & 
                 (df.buildings == 0) & (df.electricity == 0) & (df.tools == 0) & 
                 (df.hospitals == 0) & (df.shops == 0) & (df.aid_centers == 0) & 
                 (df.other_infrastructure == 0) & (df.weather_related == 0) & 
                 (df.floods == 0) & (df.storm == 0) & (df.fire == 0) & (df.earthquake == 0) & 
                 (df.cold == 0) & (df.other_weather == 0) & (df.direct_report == 0) 
                ]
    no_category_count = no_category.groupby('genre').count()['message']
    no_ctgry_genre_names = list(no_category_count.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data':[
                Bar(
                    x=no_ctgry_genre_names,
                    y=no_category_count
                )
            ],
            'layout':{
                'title': 'Distribution of Message Genres with no Categorization',
                'yaxis':{
                    'title':"Count"
                },
                'xaxis':{
                    'title':"Genre"
                }
            }    
        },
        {
            'data':[
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout':{
                'title': 'Distribution of Messages per Category',
                'yaxis':{
                    'title':"Count"
                },
                'xaxis':{
                    'title':"Category"
                }
            }    
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()