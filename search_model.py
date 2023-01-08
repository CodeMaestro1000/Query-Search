from typing import List
import torch
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from exceptions import NoCrossEncoder, NoDataAvailable

class SearchModel:
    def __init__(
        self, 
        corpus_passages: List[str] = [],  
        bi_encoder: str = '',
        use_cross_encoder: bool = False, 
        cross_encoder: str = '', 
        top_k=32, 
        name: str = 'semantic_search_model'
        ) -> None:
        """
        corpus_passages: A list of all the text you wish to generate embeddings for
        bi_encoder: A string representing a path to a SentenceTransformer bi-encoder model
        use_cross_encoder: Bool to select whether a cross encoder should be added to the model.
        cross_encoder: A string representing a path to a SentenceTransformer cross-encoder model. A cross encoder
        won't be used if use_cross_encoder is False.
        top_k: The number of passages we want to retrieve
        name: Name of your search model
        """
        self._corpus_passages = corpus_passages
        self._top_k = top_k
        self._name = name
        
        if bi_encoder == '':
            self._bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        else:
            self._bi_encoder = SentenceTransformer(bi_encoder)
        
        if use_cross_encoder:
            if cross_encoder == '':
                self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            else:
                self._cross_encoder = CrossEncoder(cross_encoder)
        else:
            self._cross_encoder = None
        
        self._corpus_embeddings = None

    def train(self) -> None:
        """
        Use bi-encoder to generate embeddings for the entire corpus
        """
        if not self._corpus_passages:
            raise NoDataAvailable(f"Corpus for model: {self._name} is empty")
        if not torch.cuda.is_available():
            logging.warning("Warning: No GPU found. Please use GPU as running on CPU may take a lot of time")
        
        self._bi_encoder.max_seq_length = 256  # Number of passages we want to retrieve with the encoder
        # compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
        self._corpus_embeddings = self._bi_encoder.encode(self._corpus_passages, convert_to_tensor=True, show_progress_bar=True)

    def load(self, embeddings_path: str, is_cpu: bool = False) -> None:
        """
        Load embeddings from pt file
        
        Use is_cpu = True if GPU is not available
        """
        if is_cpu:
            self._corpus_embeddings = torch.load(embeddings_path, map_location=torch.device('cpu'))
        else: # if using GPU
            self._corpus_embeddings = torch.load(embeddings_path)

    def save(self, filename: str) -> None:
        """Save embeddings in torch format (.pt) extension"""
        torch.save(self._corpus_embeddings, filename)

    def predict(self, query, re_rank: bool = False, is_cpu: bool = False) -> List[dict]:
        """
        Run semantic search on a single query
        
        Use is_cpu = True if GPU is not available
        """
        if not self._corpus_passages:
            raise NoDataAvailable(f"Corpus for model: {self._name} is empty")
        
        results = []
        if re_rank and self._cross_encoder == None:
            raise NoCrossEncoder(f"No cross encoder specified for model: {self._name}")
        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = self._bi_encoder.encode(query, convert_to_tensor=True)
        if not is_cpu: # when using GPU
            question_embedding = question_embedding.cuda()
        
        hits = util.semantic_search(question_embedding, self._corpus_embeddings, top_k=self._top_k)
        hits = hits[0]  # Get the hits for the first query

        if not re_rank:
            # sort scores in descending order
            hits = sorted(hits, key=lambda x: x['score'], reverse=True) 
            for hit in hits:
                data = {}
                data['score'] = hit['score']
                data['text'] = self._corpus_passages[hit['corpus_id']]
                results.append(data)
        else:
            cross_inp = []
            for hit in hits:
                text = self._corpus_passages[hit['corpus_id']]
                cross_inp.append([query, text])
            cross_scores = self._cross_encoder.predict(cross_inp)

            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]
            
            # sort cross encoder scores in descending order
            hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        
            for hit in hits:
                data = {}
                data['score'] = hit['cross-score']
                data['text'] = self._corpus_passages[hit['corpus_id']]
                results.append(data)
        
        return results
            