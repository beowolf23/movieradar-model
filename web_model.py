import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the CountVectorizer
with open('count_vectorizer.pkl', 'rb') as f:
    count = pickle.load(f)

# Load the Cosine Similarity matrix
with open('cosine_similarity.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Load the DataFrame and indices
df2 = pd.read_pickle('movies_df.pkl')
with open('indices.pkl', 'rb') as f:
    indices = pickle.load(f)
    
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2[['id', 'original_title']].iloc[movie_indices].to_dict('records')

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    recommendations = get_recommendations(title)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080, debug=True)
