import numpy
from rwkvt.dataset.binidx import MMapIndexedDataset
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from rwkvt.dataset.binidx import MMapIndexedDataset

tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

dataset = MMapIndexedDataset("data/message_cache")
print(f"Total documents: {len(dataset)}")

with open("data/dump.txt", "wb") as f:
    for i in range(0, len(dataset), 100):
        tokens: numpy.ndarray = dataset[i].astype(int)
        tokens = numpy.where(tokens == 65530, 62, tokens)
        tokens = numpy.where(tokens == 24, 261, tokens)
        string = tokenizer.decodeBytes(tokens.tolist())
        print(f"\nSample {i+1}:")
        print("Decoded text:", string.decode(errors="ignore"))
        print("Token IDs:", tokens)
        f.write(string)
        f.write(b"\n\n" + b"=" * 80 + b"\n\n")
