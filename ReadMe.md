# Query Search
A python library for generating and using sentence embeddings for semantic search.

Developed using the [SentenceTransformer](https://www.sbert.net/) framework which was built to generate state-of-the-art text and image embeddings using Siamese BERT Networks (read paper [here](https://arxiv.org/abs/1908.10084)).

The library uses a bi-encoder transformer model for generating embeddings (or feature vectors) which is good enough on it's own. If you need a more powerful model, the library is also capable of further performing retrieval and re-ranking using a more powerful cross encoder model.

The library also supports a Bring-Your-Own model design pattern where you can provide your own bi-encoder and cross encoder models.

# Use Case
This library was developed to power semantic search engines for small to medium sized datasets. 

A good use case would be to deploy a semantic search engine and integrate it with an FAQ or knowledge base section for your website to improve user experience.

# To-Do
I'm currently building a no-code application where users will be able to provide their data and we'll handle training and deployment. Watch this space!!!

# Usage
I've given examples on how to train and save embeddings in `train.py` and how to generate predictions with the embeddings in `inference.py`.

I also provided a colab notebook `Query Search.ipynb` where there's a complete example from training to generating predictions.

The dataset used is a stackoverflow dataset containing titles and IDs of stackoverflow questions related to python. You can download the CSV file [here](https://drive.google.com/file/d/1--LZrZlh28VxFginMRABA4gvzBMzbaOj/view?usp=sharing)

# Thank you
Your contributions and feedback are more than welcome!!!