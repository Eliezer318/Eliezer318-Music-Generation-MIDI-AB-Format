import torch
from torch import nn
from typing import Tuple, Callable, Any
import math

from data import device


def token_to_indices(word2idx: dict, tokens: str) -> torch.Tensor:
    return torch.tensor([word2idx[token] for token in tokens.split(' ')], device=device).long()


def weights_init_normal(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, y ** -0.5)
        m.bias.data.fill_(0)


class BaseModel(nn.Module):
    def __init__(self, vocab_size: int, input_size: int, hidden_size=90, num_layers: int = 2, dropout: float = 0.1):
        super(BaseModel, self).__init__()
        self.words_embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.seq = nn.Sequential(nn.Linear(hidden_size, vocab_size), nn.ReLU(), nn.Linear(vocab_size, vocab_size))
        self.apply(weights_init_normal)

    def forward(self, words_indices: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, Tuple]:
        embedded = self.words_embedding(words_indices)
        output, hidden = self.lstm(embedded) if hidden is None else self.lstm(embedded, hidden)
        output = self.seq(output.squeeze(0))  # [seq_len, vocab_size]
        return output, hidden

    @torch.no_grad()
    def sample(self, word_map: tuple, start_tokens='<s>', max_length=800, T=1.) -> str:
        self.eval()
        idx2word, word2idx = word_map
        out_text = start_tokens
        hs = None
        start_tokens = token_to_indices(word2idx, start_tokens).unsqueeze(1).unsqueeze(2)
        for token in start_tokens[-1][:-1]:
            _, hs = self.forward(token, hs)
        embedded = start_tokens[-1]
        for i in range(max_length):
            y, hs = self.forward(embedded, hs)
            new_token = idx2word[(y[-1] / T).softmax(0).multinomial(num_samples=1).item()]
            out_text += " " + new_token
            embedded = token_to_indices(word2idx, new_token).unsqueeze(1)
            if new_token == '</s>':
                break
        return out_text.replace("<s> ", "").replace(" </s>", "").replace('\n ', '\n')


class AdvancedModel(nn.Module):

    def __init__(self, ntokens: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super(AdvancedModel, self).__init__()
        self.ntokens = ntokens
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntokens)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = src.T
        src_mask = generate_square_subsequent_mask(src.shape[0]).to(device)
        src = self.encoder(src) * (self.d_model ** 0.5)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output.view(-1, self.ntokens), None

    @torch.no_grad()
    def sample(self, word_map: tuple, start_tokens='<s>', max_length=800, T=1.) -> str:
        self.eval()
        idx2word, word2idx = word_map
        out_text = start_tokens
        start_tokens = token_to_indices(word2idx, start_tokens).unsqueeze(1)
        for i in range(max_length):
            y, _ = self.forward(start_tokens)
            new_token = idx2word[(y[-1] / T).softmax(0).multinomial(num_samples=1).item()]
            if new_token == '</s>':
                break
            out_text += " " + new_token
            embedded = token_to_indices(word2idx, out_text).unsqueeze(0)
            start_tokens = embedded
        out_text = ' '.join(out_text.split(' ')[1:]).replace('\n ', '\n')
        return out_text[1:] if out_text[0] == ' ' else out_text

    @torch.no_grad()
    def sample(self, word_map: tuple, start_tokens='<s>', max_length=800, T=1., remove_first=True) -> str:
        self.eval()
        idx2word, word2idx = word_map
        out_text = start_tokens
        start_tokens = token_to_indices(word2idx, start_tokens).unsqueeze(1)
        for i in range(max_length):
            y, _ = self.forward(start_tokens)
            new_token = idx2word[(y[-1] / T).softmax(0).multinomial(num_samples=1).item()]
            if new_token == '</s>':
                break
            out_text += " " + new_token
            embedded = token_to_indices(word2idx, out_text).unsqueeze(0)
            start_tokens = embedded
        out_text = ' '.join(out_text.split(' ')[int(remove_first):]).replace('\n ', '\n')
        return out_text[1:] if out_text[0] == ' ' else out_text

    @torch.no_grad()
    def extend_new_music(self, word_map: tuple, start_tokens: str, every=4, T=1.5):
        self.eval()
        first_n: Callable[[str, str], str] = lambda string, idx: ' '.join(string.split(' ')[: idx])
        last_n: Callable[[str, str], str] = lambda string, idx: ' '.join(string.split(' ')[idx:])
        initials = first_n(start_tokens, 3)
        start_tokens = last_n(start_tokens, 3)
        n = len(start_tokens.split(' '))
        for i in range(3, n - 5, every):
            start_tokens = (self.sample(word_map, first_n(start_tokens, i), 1, T, False) + " " + last_n(start_tokens, i)).replace('\n', '\n ').replace(' <s>', '').replace(' </s>', '')
        return initials + " " + start_tokens


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(0)])
