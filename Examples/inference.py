from search_model import SearchModel

saved_model = SearchModel()
saved_model.load('my_embeddings.pt', is_cpu=False) # Using GPU

query = "Hello world in python"

print("Without retrieve and re-ranking: ")
results = saved_model.predict(query)
for data in results[:5]:
    print(f"Text: {data['text']}, Score: {data['score']}")

print("Using retrieve and re-ranking: ")
results = saved_model.predict(query, re_rank=True)
for data in results[:5]:
    print(f"Text: {data['text']}, Score: {data['score']}")