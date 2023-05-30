import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, sigma=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_dim) * sigma)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * sigma)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, X, hidden_state=None):
        # X shape: (len_sequence, batch_size, embedding_dim)
        if hidden_state is None:
            hidden_state = torch.zeros((X.shape[1], self.hidden_dim)).to(X.device)
        
        hidden_states = []
        for X_t in X:
            hidden_state = torch.tanh(X_t @ self.W_xh + hidden_state @ self.W_hh + self.b_h)
            hidden_states.append(hidden_state)

        return torch.stack(hidden_states), hidden_state
    

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, sigma=0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.W_e = nn.Parameter(torch.randn(vocab_size, embedding_dim) * sigma)

        self.rnn = RNN(embedding_dim, hidden_dim)
        self.W_hq = nn.Parameter(torch.randn(hidden_dim, vocab_size) * sigma)
        self.b_q = nn.Parameter(torch.zeros(vocab_size))

    def embedding(self, X):
        one_hot = F.one_hot(X, self.vocab_size)
        embeddings = one_hot.type(torch.float32) @ self.W_e
        return embeddings

    def forward(self, X, hidden_state=None):
        embeddings = self.embedding(X)
        hidden_states, hidden_state = self.rnn(embeddings, hidden_state)

        outputs = []
        for state in hidden_states:
            output = state @ self.W_hq + self.b_q
            outputs.append(output) 
        
        return torch.stack(outputs)
    
    def generate(self, idxs, max_seq_len=10):
        embeddings = self.embedding(idxs)

        _, hidden_state = self.rnn(embeddings, None)

        for _ in range(max_seq_len):
            logits = hidden_state @ self.W_hq + self.b_q
            predict_idxs = torch.argmax(logits, dim=1, keepdim=True)

            idxs = torch.cat((idxs, predict_idxs))

            predict_embeddings = self.embedding(predict_idxs)
            _, hidden_state = self.rnn(predict_embeddings, hidden_state)
            
        return idxs


if __name__ == '__main__':
    batch_size = 4
    embedding_dim = 512
    hidden_dim = 256
    len_sequence = 64
    vocab_size = 10000

    # rnn = RNN(embedding_dim, hidden_dim)
    # X = torch.ones((len_sequence, batch_size, embedding_dim))
    # hidden_states, hidden_state = rnn(X)
    # print(hidden_states.shape)

    model = RNNLM(vocab_size, embedding_dim, hidden_dim)
    # X = torch.ones((len_sequence, batch_size), dtype=torch.int64)
    # outputs = model(X)
    # print(outputs.shape)


    idxs = torch.randint(0, vocab_size, (len_sequence, ), dtype=torch.int64)
    idxs = torch.unsqueeze(idxs, -1)
    idxs = model.generate(idxs)
    print(torch.squeeze(idxs))