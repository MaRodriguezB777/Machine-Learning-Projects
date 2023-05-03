import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparams
block_size = 8 # Token size / Recall size
batch_size = 32 # Number of simultaneous training points
epochs = 10000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
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

# Estimate Loss throughout epochs
@torch.no_grad()
def estimate_loss():
    est = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            xb, yb = get_batch(split)

            _, loss = m(xb, yb)
            losses[iter] = loss.item()

        est[split] = losses.mean(dim=0)
    m.train()
    return est

# Defining model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Each token is embedded into a size of number of vocab letters
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and target both of dimensions (B, T) where B is batch size and T is time / token dimension
        preds = self.token_embedding_table(idx) # (B, T, C)

        if targets == None:
            loss = None
        else:
            B, T, C = preds.size()
            preds = preds.view(B*T, C) # Says to make each entry in 2d space its own entry and get the corresponding embedding table.
            targets = targets.view(B*T) # corresponds to the actual target entry (the index in C dimension of preds) that we see. 
            loss = F.cross_entropy(preds, targets)
            
        return preds, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            preds, _ = self(idx) # (B, T, C)
            
            # only care about last prediction
            preds = preds[:,-1, :] # (B, C)

            probs = F.softmax(preds, dim=1) # (B, C)

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) / One prediction per batch
            idx = torch.cat((idx, idx_next), dim=1) # Want to add this prediction to end of each batch
        return idx

    def sample(self, max_new_tokens):
        return self.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens)[0].tolist()

# Model and optimizer setup
m = BigramLanguageModel()
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
m.to(device)

# Training the model
for epoch in range(epochs):

    xb, yb = get_batch("train")

    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step = {epoch}: train_loss = {losses['train']}, val_loss = {losses['val']}")

    preds, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.sample(1000)))