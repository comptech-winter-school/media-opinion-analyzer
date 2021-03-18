import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
from PIL import Image

import plotly.express as px
import pickle

import umap
import plotly.figure_factory as ff
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(
page_title="Ruler - Opinion analysis tool",
page_icon="📏",
initial_sidebar_state="expanded",
)


image = Image.open('logo.jpg')

st.image(image,width=500)

st.markdown("<h2 style='text-align: left; font-size:200%'>Opinion analysis tool</h1>", unsafe_allow_html=True)

#st.write("blah blah blah text")

input_text = st.text_input('Text for analysis', '') #We need to build a wall!


selected_model = st.selectbox(
    'Which model would you like to use?',
     ["Doc2Vec", "Sentence-BERT", "Universal Sentence Encoder"])

if selected_model == "Doc2Vec":
    all_df = pickle.load(open("df_doc2vec.pickle", "rb"))
    all_df = all_df.rename(columns={"embedding":"vec"})
elif selected_model == "Sentence-BERT":
    all_df = pickle.load(open("df_sbert.pickle", "rb"))
    all_df = all_df.rename(columns={"embedding":"vec"})
elif selected_model == "Universal Sentence Encoder":
    all_df = pickle.load(open("df_use.pickle", "rb")) 


st.write(all_df.head())

loaded_model = pickle.load(open("USE_model.pickle", 'rb'))

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Compute a representation for each message, showing various lengths supported.
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed([input_text]))

use_text_vec = message_embeddings[0]
result = loaded_model.predict([use_text_vec])

if selected_model == "Universal Sentence Encoder":
    all_df = all_df.append({"body": input_text, "vec": use_text_vec, "who": 2},ignore_index=True)

#result[0]

if result[0] == 0:
    st.markdown("Most likely this text is **pro-Trump**")
else:
    st.markdown("Most likely this text is **pro-Biden**")

expander = st.beta_expander("FAQ")
expander.write("""
## How does this works?
This app is geting the proximity of the vector representation of the text to the reference vectors of the topic and determine the polarity of the statements of the text 
""")

#st.button('Say hello')
st.write("Data visualization")

scatter_type = st.radio("Select plot type", ("2D scatter", "3D scatter"))

if scatter_type == "2D scatter":
    
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.0, metric='cosine')
    embedding = reducer.fit_transform(all_df["vec"].tolist())
    all_df["x"] = embedding[:, 0]
    all_df["y"] = embedding[:, 1]
    
    #st.write(all_df.head())

    fig = px.scatter(all_df, x="x", y="y", color="who", hover_name='body',opacity=0.3)
    st.write(fig)

elif scatter_type == "3D scatter":
    
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.0, metric='cosine', n_components=3)
    embedding = reducer.fit_transform(all_df["vec"].tolist())
    all_df["x"] = embedding[:, 0]
    all_df["y"] = embedding[:, 1]
    all_df["z"] = embedding[:, 2]
    
    #st.write(all_df.head())

    fig = px.scatter_3d(all_df, x="x", y="y", z="z", color="who", hover_name='body',opacity= (0.3))
    st.write(fig)


trump_sim2 = []
biden_sim2 = []
tb_sim2 = []


from scipy.spatial.distance import cosine
def cosine_similarity(a,b):
  return (1 - cosine(a,b))

sample_value = 10000

for i in range(sample_value):
  trump_sample_1 = all_df.loc[all_df['who'] == 1]["vec"].sample().values[0]
  trump_sample_2 = all_df.loc[all_df['who'] == 1]["vec"].sample().values[0]
  trump_sim2.append(cosine_similarity(trump_sample_1, trump_sample_2))

  biden_sample_1 = all_df.loc[all_df['who'] == 0]["vec"].sample().values[0]
  biden_sample_2 = all_df.loc[all_df['who'] == 0]["vec"].sample().values[0]
  biden_sim2.append(cosine_similarity(biden_sample_1, biden_sample_2))

  #tb_sim2.append(cosine_similarity(biden_sample_1, trump_sample_1))


fig = ff.create_distplot([trump_sim2, biden_sim2], ["Trump","Biden"], bin_size=[.02, .02])
st.plotly_chart(fig, use_container_width=True)

st.markdown("<h2 style='text-align: center; font-size:100%'>CompTech2021 winter school</h1>", unsafe_allow_html=True)