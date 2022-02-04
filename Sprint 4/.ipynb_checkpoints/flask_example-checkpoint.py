from flask import Flask, request,jsonify 
from sklearn.metrics.pairwise import linear_kernel
import pickle
import pandas as pd
from flask_cors import CORS

# tdidf.pkl
tfidf_matrix = pickle.load(open("tfidfvector.pkl","rb"))
# titles.pkl
titles = pickle.load(open("titles.pkl","rb"))
# indices.pkl
indices = pickle.load(open("indices.pkl","rb"))


def recommendations(title):
    print(title)
    if isinstance(indices[title], pd.core.series.Series):
        idx = indices[title][0]
    else:
        idx = indices[title]


    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix)
    sim_scores = sim_scores.reshape(sim_scores.size)
    scores_with_index = pd.Series(sim_scores,index=titles.index)
    scores_with_index = scores_with_index.sort_values(ascending=False)
    scores_with_index = scores_with_index[1:21]
    only_index = scores_with_index.index

    return titles.iloc[only_index].values



app=Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return '<h1>Flask is running</h1>'

@app.route('/recom',methods=['POST'])
def movie_recommend():
    content = request.data.decode("utf-8")
    print(content)
    # print(type(content.decode("utf-8")))
    # print("chi mai ka")
    result = recommendations(content.strip('\"'))
    result = result.tolist()
    
    return jsonify(result)






if __name__=='__main__':
    app.run()





# from sklearn.metrics.pairwise import linear_kernel
# import pickle
# import pandas as pd

# # tdidf.pkl
# tfidf_matrix = pickle.load(open("tfidfvector.pkl","rb"))
# # titles.pkl
# titles = pickle.load(open("titles.pkl","rb"))
# # indices.pkl
# indices = pickle.load(open("indices.pkl","rb"))

# def genre_recommendations(title):
#     if isinstance(indices[title], pd.core.series.Series):
#         idx = indices[title][0]
#     else:
#         idx = indices[title]


#     sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix)
#     sim_scores = sim_scores.reshape(sim_scores.size)
#     scores_with_index = pd.Series(sim_scores,index=titles.index)
#     scores_with_index = scores_with_index.sort_values(ascending=False)
#     scores_with_index = scores_with_index[1:21]
#     only_index = scores_with_index.index

#     return titles.iloc[only_index]