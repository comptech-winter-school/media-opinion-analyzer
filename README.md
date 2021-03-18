# media-opinion-analyzer

The main purpose is to help scientists from all over the world to estimate and analyze social opinion from social-media comments. The idea is to use text embedding algorithms to vectorize comments which then we can use for clustering, classification, dynamic analyzation and similarity comparison with reference text.
The approach is tested on data from the Reddit platform.

<a href="https://colab.research.google.com/drive/1scGdPdq4bS1DFhphSRsL4GoHbfyMJ0gQ?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Repository structure

| Folder           | Description                     |
| :-------------------- | :------------------------------------------------- |
| preprocessing | preprocessing data downloaded from Reddit |
| webapp | streamlit web_app |
| sBert| testing sBert: vectorization, classification, cos_sim, clustering |
| doc2vec | testing doc2vec: vectorization, classification, cos_sim |
| USE| testing USE: vectorization, classification, cos_sim |


## How to use

Run streamlit app in webapp folder. 
- Install streamlit library 
- Set environmet. Libraries listed in requirements.txt
- Put three tables ("df_doc2vec", "df_sbert", "df_use") in pickle format to the same folder with my_app.py.        
  Main columns names: body, vec, who.         
  In 'body' column comments with type string, in vec - embeddings, in who biden(1) or trump(0) type int.
- Run streamlit app: ```streamlit run my_app.py```              
  more info here https://docs.streamlit.io/en/stable/streamlit_configuration.html


## Summary 
Results for sBert, doc2vec, USE
1. similarity to reference text: 'We should build the wall!'
2. classification to the right party
3. clusters
4. dynamic