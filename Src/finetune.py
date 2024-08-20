import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.cuda.amp import GradScaler, autocast
from text_generator import TextGenerator
from dataset import PersianWikipediaDataset

def finetune_model(model, dataset, tokenizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True))
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            targets = batch['input_ids'].to(device)

            optimizer.zero_grad()
            hidden = (torch.zeros(num_layers, inputs.size(0), hidden_dim).to(device),
                      torch.zeros(num_layers, inputs.size(0), hidden_dim).to(device))

            inputs = inputs[:, :-1]
            targets = targets[:, 1:].reshape(-1)

            with autocast():
                output, hidden = model(inputs, hidden)
                output = output.view(-1, vocab_size)
                loss = criterion(output, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')

    torch.save(model.state_dict(), 'finetuned_text_generator.pth')

if __name__ == "__main__":
    embedding_dim = 64
    hidden_dim = 128
    num_layers = 1
    vocab_size = tokenizer.vocab_size

    model = TextGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)
    num_articles = 200
    dataset = PersianWikipediaDataset(num_articles=num_articles)
    finetune_model(model, dataset, tokenizer, num_epochs=10)
