import torch

#hyperparams
block_size = 8 # Token size / Recall size
batch_size = 32 # Number of simultaneous training points
epochs = 10000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
embd_size = 32
# ----------

with open("input.txt") as f:
    text = f.read()

# get characters from text
chars = list(set(text))
vocab_size = len(chars)

# create mappings, decoder, and encoder for characters to indices and vice-versa
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n_split = int(0.9*len(data))
train_data = data[:n_split]
val_data = data[n_split:]

# create batches either for testing or validation
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[bix: bix + block_size] for bix in ix]).to(device)
    y = torch.stack([data[bix + 1 : bix + block_size + 1] for bix in ix]).to(device)

    return x, y
