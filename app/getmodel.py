# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
# model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
# tokenizer.save_pretrained('./local_directory/')
# model.save_pretrained('./local_directory/')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

path = '../models'
model.save(path=path, model_name='sbert')