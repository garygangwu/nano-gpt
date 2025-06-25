import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import glob
import argparse

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PositionalCoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model=512, nhead=8, dim_feedforward=1024, nlayers=4, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'

        # Positional encoding helps the model capture the order of tokens
        self.pos_encoder = PositionalCoding(d_model, dropout)

        # Transformer Encoder Stack
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)

        # Token embedding: maps tokens to vectors of size d_model
        self.encoder = nn.Embedding(ntoken, d_model)

        # Add a sequence pooling layer
        self.sequence_pooling = nn.Sequential(
            nn.Linear(d_model, d_model),  # Transform each position
            nn.ReLU(),
            nn.Linear(d_model, d_model),  # Combine information
            nn.ReLU(),
            nn.Linear(d_model, d_model)   # Final transformation
        )

        # Output layer maps back to vocabulary size for prediction
        self.decoder = nn.Linear(d_model, ntoken)

        self.d_model = d_model

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # Initialize embedding and decoder weights uniformly
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        # print("!!src", src.shape)
        # Embedding + Scaling (standard practice for transformer)
        src = self.encoder(src) * math.sqrt(self.d_model)
        # print("!!src2", src.shape)

        # Add positional encoding
        src = self.pos_encoder(src)
        # print("!!src3", src.shape)

        src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)

        # Transformer Decoder processing
        output = self.transformer_decoder(src, src, src_mask)
        # print("!!output", output.shape)

        output = self.sequence_pooling(output)
        # print("!!output_pooled", output.shape)

        # Project to vocabulary logits
        output = self.decoder(output)
        # print("!!output_decoder", output.shape)

        return output



raw_text_list = []
for filename in glob.glob('data/*.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text_list.append(f.read())

raw_text = ''.join(raw_text_list)

chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

#print("chars:", chars)
#print("vocab_size:", vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

train_data_list = []
val_data_list = []
for raw_text in raw_text_list:
    data = torch.tensor(encode(raw_text), dtype=torch.long).to(device)
    n = int(0.8 * len(data))
    train_data_list.append(data[:n])
    val_data_list.append(data[n:])

train_data = torch.cat(train_data_list)
val_data = torch.cat(val_data_list)


batch_size = 256
block_size = 128

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Train or load the transformer model')
parser.add_argument('--training', action='store_true', default=False, help='Train the model if specified, otherwise load from file')
# Default model path constant
DEFAULT_MODEL_PATH = 'model_data/transformer_model.pth'
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to load the model from')

args = parser.parse_args()

if args.training:
    # Training mode
    os.makedirs('intermediate_data', exist_ok=True)
    os.makedirs('model_data', exist_ok=True)
    model = TransformerModel(vocab_size)
    model = model.to(device)  # Move model to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 3000

    for epoch in range(epochs):
        total_loss = 0
        input_data, target_data = get_batch('train')
        # Data is already on GPU from get_batch function

        output = model(input_data)

        B, T, C = output.shape
        output = output.view(B*T, C)
        target_data = target_data.view(B*T)

        loss = F.cross_entropy(output, target_data)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Track the best validation loss and save the best model separately
        if epoch == 0:
            best_val_loss = float('inf')

        if (epoch+1) % 200 == 0:
            # Evaluate on validation data
            model.eval()
            with torch.no_grad():
                val_input, val_target = get_batch('val')
                val_output = model(val_input)
                Bv, Tv, Cv = val_output.shape
                val_output = val_output.view(Bv*Tv, Cv)
                val_target = val_target.view(Bv*Tv)
                val_loss = F.cross_entropy(val_output, val_target)
            model.train()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            torch.save(model.state_dict(), f'intermediate_data/transformer_model_epoch_{epoch+1}.pth')
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'intermediate_data/transformer_model_best.pth')
                print(f"New best model saved with Val Loss: {best_val_loss:.4f}")



    # Save the model after training
    torch.save(model.state_dict(), DEFAULT_MODEL_PATH)
    print(f"Model saved to {DEFAULT_MODEL_PATH}")

else:
    # Load mode
    model = TransformerModel(vocab_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)  # Move model to GPU
    print(f"Model loaded from {args.model_path}")


test_sentence = "开始"

eval_data = torch.tensor(encode(test_sentence), dtype=torch.long).to(device)
print(eval_data)

model.eval()
with torch.no_grad():
    for _ in range(2000):
        #print("input : |", decode(eval_data.tolist()), "|")
        eval_data_unsqeeuze = eval_data[-block_size:].unsqueeze(0)

        # print("input", eval_data_unsqeeuze)
        output = model(eval_data_unsqeeuze)
        # print("output", output)
        # print("output", output.shape)
        # print("output_last", output[-1, -1])
        # print("output_last", output[-1, -1].shape)
        probs = output[-1, -1].softmax(dim=0)
        next_token = torch.multinomial(probs, num_samples=1)
        eval_data = torch.cat((eval_data, next_token), dim=-1)

        #print("output: |", decode(eval_data.tolist()), "|")

print(decode(eval_data.tolist()))




