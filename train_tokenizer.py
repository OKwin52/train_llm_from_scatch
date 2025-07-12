from tokenizers import models, Tokenizer, decoders, pre_tokenizers, trainers
import os
import json
from typing import Iterable
def read_data(path: str) -> Iterable[str]: 
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            yield line['text']

# BPE 
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
# Special tokens
special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
# initialize the trainer
trainer = trainers.BpeTrainer(
    vocab_size=6400,
    special_tokens=special_tokens,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
# read data
texts = read_data("llm_from_scratch\dataset\pretrain_hq.jsonl")
tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.decoder = decoders.ByteLevel()

tokenizer_dir = "./tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))