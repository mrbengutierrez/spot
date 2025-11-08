```
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model_st = None
def _get_st_model():
    global _model_st
    if _model_st is None:
        _model_st = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_st

def sentence_adj_similarity_coherence(text: str) -> float:
    """
    Returns average cosine similarity between adjacent sentences.
    Higher = more locally coherent.
    """
    # trivial sentence split – replace with a better splitter if you like
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    if len(sentences) < 2:
        return 1.0  # single sentence is trivially "coherent"
    
    model = _get_st_model()
    embs = model.encode(sentences)
    
    sims = []
    for i in range(len(embs) - 1):
        sim = cosine_similarity([embs[i]], [embs[i+1]])[0][0]
        sims.append(sim)
    
    return sum(sims) / len(sims)

```

```
# pip install spacy
# python -m spacy download en_core_web_sm
import spacy

_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def entity_overlap_coherence(text: str) -> float:
    """
    Fractional overlap of entities/noun-chunks between adjacent sentences.
    Returns value in [0,1]. Higher = more shared discourse entities.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    # split by sentences from spaCy
    sents = list(doc.sents)
    if len(sents) < 2:
        return 1.0
    
    overlaps = []
    for i in range(len(sents) - 1):
        s1 = sents[i]
        s2 = sents[i+1]
        
        # collect surface forms of entities + noun chunks
        def collect_units(span):
            ents = {e.text.lower().strip() for e in span.ents}
            # add noun chunks to be less brittle
            nch = {nc.text.lower().strip() for nc in span.noun_chunks}
            return ents.union(nch)
        
        e1 = collect_units(s1)
        e2 = collect_units(s2)
        
        if not e1 and not e2:
            overlaps.append(1.0)  # two generic sentences; don't punish
            continue
        inter = len(e1.intersection(e2))
        union = len(e1.union(e2))
        overlaps.append(inter / union)
    
    return sum(overlaps) / len(overlaps)

```

```
# pip install transformers torch
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math

_bart_tokenizer = None
_bart_model = None
def _get_bart():
    global _bart_tokenizer, _bart_model
    if _bart_model is None:
        model_name = "facebook/bart-large-cnn"
        _bart_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _bart_tokenizer, _bart_model

def bartstyle_source_summary_coherence(source: str, summary: str) -> float:
    """
    Returns negative log-likelihood per token of the summary given the source.
    Lower is better. We invert to make 'higher is better'.
    """
    tokenizer, model = _get_bart()
    inputs = tokenizer(source, return_tensors="pt", truncation=True, max_length=1024)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summary, return_tensors="pt", truncation=True, max_length=256).input_ids
    labels[labels == tokenizer.pad_token_id] = -100  # ignore padding in loss
    
    with torch.no_grad():
        loss = model(**inputs, labels=labels).loss.item()
    # convert to a bounded-ish score: higher is better
    score = math.exp(-loss)  # just a monotonic transform
    return score
```

```
from typing import List
import torch

def compute_nsp_coherence(summary: str,
                          model_name: str = "bert-base-uncased") -> float:
    """
    Uses BERT's Next Sentence Prediction head to score adjacent sentences.
    Returns average P(is_next).
    """
    from transformers import BertTokenizer, BertForNextSentencePrediction

    sents = sentence_split(summary)
    if len(sents) < 2:
        return 1.0

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForNextSentencePrediction.from_pretrained(model_name)
    model.eval()

    probs = []
    for a, b in zip(sents[:-1], sents[1:]):
        encoding = tokenizer(a, b, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**encoding).logits  # shape [1,2]
        # logits[0,0] = is_next, logits[0,1] = not_next for BERT NSP
        is_next_prob = torch.softmax(logits, dim=1)[0, 0].item()
        probs.append(is_next_prob)

    return sum(probs) / len(probs)
```

```
from typing import List
import numpy as np

def sentence_split(text: str) -> List[str]:
    # super simple splitter; swap for nltk or spacy in real use
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s]

def compute_adjacent_embedding_coherence(summary: str,
                                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    """
    Returns mean cosine similarity between adjacent sentences.
    Higher = more locally coherent.
    """
    from sentence_transformers import SentenceTransformer
    from numpy.linalg import norm

    sents = sentence_split(summary)
    if len(sents) < 2:
        return 1.0  # trivially coherent

    model = SentenceTransformer(model_name)
    embeddings = model.encode(sents)

    sims = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i+1]
        cos = float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))
        sims.append(cos)

    return float(np.mean(sims))

```
