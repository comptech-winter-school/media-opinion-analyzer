# media-opinion-analyzer
This project was born during CompTech2021 winter-school.

Online social media has firmly entered our lives. According to facebook humanity spending more time in social media and the audience of social media is growing. Network often become a platform for discussion of various phenomena and events.

On the other hand, sociologists have devoted more than a decade to the development of models for the transformation of opinions. These models are a mathematical description of the principles and patterns observed in a real environment. With the help of big data, research on the interaction of opinions on social media is already emerging. For example, the presence of echoes of rooms on Twitter and Facebook has been experimentally confirmed.

As we know, mathematical models work with numbers, and opinions are expressed by text. Scientists need to go for different tricks to elicit opinions from social media data. The metric is usually the number of likes or self-reported data (e.g. “on a scale of 1 to 10, describe how much you like yaoi”).


## Purpose

The main purpose is to help scientists from all over the world to estimate and analyze social opinion from social-media comments. The idea is to use text embedding algorithms to vectorize comments which then we use for clustering, classification, dynamic analyzation and similarity comparison with reference text.
The approach is tested on data from the Reddit platform.


## Repository structure

| File name              | Desctiption                     |
| :-------------------- | :------------------------------------------------- |
| preprocessing | preprocessing data downloaded from Reddit |
| webapp | streamlit web_app |
| sBert| testing sBert: vectorization, classification, cos_sim, clustering |
| doc2vec | testing doc2vec: vectorization, classification, cos_sim |
| USE| testing USE: vectorization, classification, cos_sim, clustering |
| LanguageModel.ipynb | pipeline. Input: table with vectors of a chosen model. Output: cos_sim, opinion_dynamic, clusters, classification |


## How to use

There are 2 ways to check results:
1. Use LanguageModel.ipynb. 
   In "/content/drive/My Drive/weights/" tables in pickle format.
   Columns: created_utc, author, body, embedding
   
2. Run streamlit app in webapp folder. 
   - Install streamlit library 
   - Set environmet. Libraries listed in requirements.txt
   - Put three tables ("df_doc2vec", "df_sbert", "df_use") in pickle format to the same folder with my_app.py
     Columns names: body, vec, who. 
     In 'body' comments type string, in vec - embeddings, in who biden(1) or trump(0) type int.
   - Run streamlit app: ```streamlit run my_app.py [-- script args]```
     more info here https://docs.streamlit.io/en/stable/streamlit_configuration.html


## Summary 
Resulats of sBert, doc2vec, USE
1. similarity to reference text: 'We should build the wall!'
2. classification to the right party
3. clusters
4. dynamic