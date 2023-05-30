import torch
import torch.nn as nn

from torcheval.metrics.functional.text import perplexity

from model import RNNLM


def train(args):
    train_dataloader = None
    val_dataloader = None
    
    model = RNNLM(args.vocal_size, args.embedding_dim, args.hidden_sim)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learing_rate)
    criterion = nn.CrossEntropyLoss()

    iter = 0
    for epoch in range(args.epochs):
        model.train()
        for X, y in train_dataloader:
            X = X.to(args.device)
            y = y.to(args.device)

            logits = model(X)
            loss = criterion(logits.reshape(-1, args.vocal_size), y.reshape(-1, ))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters())
            optimizer.step()

            iter += 1

            if iter % 10 == 0:
                train_loss, train_perplexity = 0, 0
                model.eval()
                with torch.no_grad():
                    for X, y in train_dataloader:
                        X = X.to(args.device)
                        y = y.to(args.device)

                        logits = model(X)
                        loss = criterion(logits.reshape(-1, args.vocal_size), y.reshape(-1, ))
                        train_loss += loss.item()
                        train_perplexity += perplexity(logits.permute(1, 0, 2), y).item()
                
                train_loss /= len(train_dataloader)
                train_perplexity /= len(train_dataloader)
                print(f'iter {iter} - loss: {train_loss:.4f} - perplexity: {train_perplexity:.4f}')
                        

            if iter % 100 == 0:
                val_loss, val_perplexity = 0, 0
                model.eval()
                with torch.no_grad():
                    for X, y in val_dataloader:
                        X = X.to(args.device)
                        y = y.to(args.device)

                        logits = model(X)
                        loss = criterion(logits.reshape(-1, args.vocal_size), y.reshape(-1, ))
                        val_loss += loss.item()
                        val_perplexity += perplexity(logits.permute(1, 0, 2), y).item()
                
                    val_loss /= len(val_dataloader)
                    val_perplexity /= len(val_dataloader)
                    print(f'validation - iter {iter} - loss: {val_loss:.4f} - perplexity: {val_perplexity:.4f}')




    pass

class Args():
    def __init__(self) -> None:
        self.vocal_size = 100
        self.len_sequence = 50
        self.embedding_dim = 32
        self.hidden_dim = 16

        self.learing_rate = 1e-4

        self.epochs = 3
        self.device = 'cpu'



if __name__ == '__main__':
    args = Args()
    train(args)