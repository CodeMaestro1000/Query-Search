from search_model import SearchModel
import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/SO_df.csv")

indexes = list(df.index)
passages = {}
passages['title'] = []
passages['id'] = []

for title, id in zip(df['title'][:300000], indexes[:300000]): # use first 300k rows of data
    passages['title'].append(title)
    passages['id'].append(id)

# If you like, you can also limit the number of passages you want to use
print("Passages:", len(passages['title']))

my_search_model = SearchModel(passages['title'], name='my_search_model')

if __name__ == '__main__':
    my_search_model.train()
    my_search_model.save('my_embeddings.pt')
