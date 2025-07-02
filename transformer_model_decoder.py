import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import glob
import argparse

MODEL_PARAMS = {
    'd_model': 256,
    'nhead': 4,
    'dim_feedforward': 512,
    'nlayers': 3,
    'dropout': 0.4,
    'lr': 5e-5,
    'epochs': 10000,
    'batch_size': 512,
    'block_size': 128,
    'max_length': 2000,
    'temperature': 1.05
}

class PositionalEncoding(nn.Module):
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


# Transformer Decoder Model class
class TransformerModel(nn.Module):
    def __init__(self,
                 ntokens,
                 d_model=MODEL_PARAMS['d_model'],
                 nhead=MODEL_PARAMS['nhead'],
                 dim_feedforward=MODEL_PARAMS['dim_feedforward'],
                 num_layers=MODEL_PARAMS['nlayers'],
                 dropout=MODEL_PARAMS['dropout']):
         super().__init__()
         self.model_type = 'Transformer'
         self.pos_encoder = PositionalEncoding(d_model)
         self.embedding = nn.Embedding(ntokens, d_model)

         decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
         self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

         self.fc_out = nn.Linear(d_model, ntokens)

         self.d_model = d_model
         self._init_weights()

    def _init_weights(self):
        # Explicit weight initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_emb = self.pos_encoder(tgt_emb)
        # The TransformerDecoder requires a non-None memory argument.
        # For a decoder-only model, we can pass tgt_emb as both tgt and memory.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
        output = self.transformer_decoder(tgt_emb, memory=tgt_emb, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output

# Label smoothing loss
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)

        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Tokenizer:
    def __init__(self, raw_text):
        self.chars = sorted(list(set(raw_text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])


class DataManager:
    def __init__(self, raw_text_list, tokenizer, device,
                 batch_size=MODEL_PARAMS['batch_size'], block_size=MODEL_PARAMS['block_size']):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        self.train_data_list = []
        self.val_data_list = []
        for raw_text in raw_text_list:
            data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long).to(self.device)
            n = int(0.8 * len(data))
            self.train_data_list.append(data[:n])
            self.val_data_list.append(data[n:])

        self.train_data = torch.cat(self.train_data_list)
        self.val_data = torch.cat(self.val_data_list)

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y


def generate_text(model, start_sequence, tokenizer, max_length=100, temperature=1.0):
    model.eval()
    generated_sequence = start_sequence[:]

    for _ in range(max_length):
        input_tensor = generated_sequence.unsqueeze(0)  # shape: [1, seq_len]
        output_logits = model(input_tensor)

        # Take the logits for the last token
        logits = output_logits[:, -1, :] / temperature

        # Apply softmax and sample
        probabilities = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1).item()

        # Append token
        next_token_tensor = torch.tensor([next_token_id], dtype=torch.long).to(input_tensor.device)
        generated_sequence = torch.cat([generated_sequence, next_token_tensor])

    return tokenizer.decode(generated_sequence.tolist())


def print_model_params_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

raw_text_list = []
for filename in glob.glob('data/*.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text_list.append(f.read())

raw_text = ''.join(raw_text_list)

# Instantiate the tokenizer
tokenizer = Tokenizer(raw_text)

# Instantiate DataManager
data_manager = DataManager(raw_text_list, tokenizer, device, batch_size=256, block_size=128)

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Train or load the transformer model')
parser.add_argument('--training', action='store_true', default=False, help='Train the model if specified, otherwise load from file')
# Default model path constant
DEFAULT_MODEL_PATH = 'model_data/transformer_model.pth'
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to load the model from')
parser.add_argument('--start_sentence', type=str, default="林黛玉", help='Start sentence for generation')

args = parser.parse_args()

if args.training:
    # Training mode
    os.makedirs('intermediate_data', exist_ok=True)
    os.makedirs('model_data', exist_ok=True)
    model = TransformerModel(tokenizer.vocab_size)
    model = model.to(device)  # Move model to GPU
    print_model_params_size(model)
    criterion = LabelSmoothingLoss(smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_PARAMS['lr'])

    model.train()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5  # Stop if no improvement for 5 validation checks

    for epoch in range(MODEL_PARAMS['epochs']):
        total_loss = 0
        input_data, target_data = data_manager.get_batch('train')
        # Data is already on GPU from get_batch function

        output = model(input_data)

        B, T, C = output.shape
        output = output.view(B*T, C)
        target_data = target_data.view(B*T)

        loss = criterion(output, target_data)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
        optimizer.step()

        if (epoch+1) % 100 == 0:
            # Evaluate on validation data
            model.eval()
            with torch.no_grad():
                val_input, val_target = data_manager.get_batch('val')
                val_output = model(val_input)
                Bv, Tv, Cv = val_output.shape
                val_output = val_output.view(Bv*Tv, Cv)
                val_target = val_target.view(Bv*Tv)
                val_loss = criterion(val_output, val_target)
            model.train()
            print(f"Epoch {epoch+1}/{MODEL_PARAMS['epochs']}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            torch.save(model.state_dict(), f'intermediate_data/transformer_model_epoch_{epoch+1}.pth')
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'intermediate_data/transformer_model_best.pth')
                print(f"New best model saved with Val Loss: {best_val_loss:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement in val loss for {epochs_no_improve} validation checks.")
                if epochs_no_improve >= patience:
                    print(f"Validation loss did not improve for {patience} consecutive checks. Early stopping.")
                    break

    # Copy the best model to be the final model
    import shutil
    shutil.copy('intermediate_data/transformer_model_best.pth', DEFAULT_MODEL_PATH)
    print(f"Best model copied to {DEFAULT_MODEL_PATH}")

else:
    # Load mode
    model = TransformerModel(tokenizer.vocab_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)  # Move model to GPU
    print(f"Model loaded from {args.model_path}")
    print_model_params_size(model)


eval_data = torch.tensor(tokenizer.encode(args.start_sentence + "\n"), dtype=torch.long).to(device)
print(eval_data)

generated_text = generate_text(model, eval_data, tokenizer,
                               max_length=MODEL_PARAMS['max_length'],
                               temperature=MODEL_PARAMS['temperature'])
print(generated_text)




