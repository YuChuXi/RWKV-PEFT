from rwkvt.dataset.binidx import MMapIndexedDataset


from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from rwkvt.dataset.binidx import MMapIndexedDataset

tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

dataset = MMapIndexedDataset('data/message_cache')
print(f'Total documents: {len(dataset)}')

for i in range(0, 50000, 1000):
    tokens = dataset[i].astype(int)
    print(f'\nSample {i+1}:')
    print('Token IDs:', tokens)
    print('Decoded text:', tokenizer.decode(tokens.tolist()))
    print('Token IDs:', tokens)