from flask import Flask, jsonify, request
import pickle
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the data and preprocess it
df = pd.read_csv('dataset.csv')
df = df.drop_duplicates().dropna()

product_desc = df['Product Description'].apply(lambda x: x.lower() if type(x) != float else x)
product_desc = product_desc.str.replace('[^\w\s]', '')
product_desc = product_desc.fillna('')
product_desc = product_desc.apply(
    lambda x: ' '.join([word for word in str(x).split() if word not in stopwords.words('english')]))
product_desc = product_desc.apply(lambda x: word_tokenize(str(x)))
lemmatizer = WordNetLemmatizer()
product_desc = product_desc.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
product_desc = product_desc.apply(lambda x: " ".join(x))

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
product_desc = vectorizer.fit_transform(product_desc)

sparse_matrix = csr_matrix(product_desc)

S_eff = df['Side effect'].apply(lambda x: x.lower() if type(x) != float else x)
S_eff = S_eff.str.replace('[^\w\s]', '')
S_eff = S_eff.str.replace('\d+', '')
S_eff = S_eff.fillna('')
S_eff = S_eff.apply(lambda x: ' '.join([word for word in str(x).split() if word not in stopwords.words('english')]))
S_eff = S_eff.apply(lambda x: word_tokenize(str(x)))
lemmatizer = WordNetLemmatizer()
S_eff = S_eff.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
S_eff = S_eff.apply(lambda x: " ".join(x))

S_eff_vectorizer = TfidfVectorizer()
S_eff_vectorizer.fit(S_eff)
S_eff_vectorized = S_eff_vectorizer.transform(S_eff)
S_eff_dense = csr_matrix(S_eff_vectorized).toarray()


dosage_pattern = r'\d+mg'
df['Dosage'] = df['Active salt'].apply(lambda x: re.findall(dosage_pattern, x))
df['Dosage_str'] = df['Dosage'].apply(lambda x: ','.join(x))
dosage_vectorizer = TfidfVectorizer()
dosage_vectorized = dosage_vectorizer.fit_transform(df['Dosage_str'])
dosage_dense = dosage_vectorized.toarray()

active_salt_vectorizer = TfidfVectorizer()
active_salt_vectorized = active_salt_vectorizer.fit_transform(df['Active salt'])
active_salt_dense = active_salt_vectorized.toarray()

y = df['Med_Name']
X = np.concatenate((active_salt_dense, dosage_dense, sparse_matrix.toarray(), S_eff_dense), axis=1)

# Load the trained model
model = pickle.load(open('model1.pkl', 'rb'))

med_indices = pd.Series(df.index, index=df['Med_Name']).drop_duplicates()
med_name_to_idx = dict(zip(med_indices.index, med_indices.values))

idx_to_med_name = list(df['Med_Name'])

@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def predict():
    medName = request.form.get('medName')

    med_idx = med_name_to_idx[medName]
    cos_sim_scores = cosine_similarity(X[med_idx:med_idx + 1], X)
    similar_med_indices = np.argsort(cos_sim_scores)[0][::-1][1:11]
    similar_meds = [idx_to_med_name[idx] for idx in similar_med_indices]

    result = {'medName': medName, 'similar_meds': similar_meds}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

