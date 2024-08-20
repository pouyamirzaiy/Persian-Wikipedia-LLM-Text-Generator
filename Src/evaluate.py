import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from text_generator import TextGenerator
from dataset import PersianWikipediaDataset
from datasets import load_metric

def evaluate_model(model, dataset, tokenizer, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True))

    total_loss = 0
    total_tokens = 0
    rouge_metric = load_metric("rouge")
    bleu_metric = load_metric("bleu")
    meteor_metric = load_metric("meteor")

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            targets = batch['input_ids'].to(device)

            hidden = (torch.zeros(num_layers, inputs.size(0), hidden_dim).to(device),
                      torch.zeros(num_layers, inputs.size(0), hidden_dim).to(device))

            inputs = inputs[:, :-1]
            targets = targets[:, 1:].reshape(-1)

            with autocast():
                output, hidden = model(inputs, hidden)
                output = output.view(-1, vocab_size)
                loss = criterion(output, targets)

            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

            generated_texts = tokenizer.batch_decode(torch.argmax(output, dim=-1), skip_special_tokens=True)
            reference_texts = tokenizer.batch_decode(targets, skip_special_tokens=True)

            rouge_metric.add_batch(predictions=generated_texts, references=reference_texts)
            bleu_metric.add_batch(predictions=[text.split() for text in generated_texts], references=[[text.split()] for text in reference_texts])
            meteor_metric.add_batch(predictions=generated_texts, references=reference_texts)

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    rouge_results = rouge_metric.compute()
    bleu_results = bleu_metric.compute()
    meteor_results = meteor_metric.compute()

    return perplexity, rouge_results, bleu_results, meteor_results

if __name__ == "__main__":
    embedding_dim = 64
    hidden_dim = 128
    num_layers = 1
    vocab_size = tokenizer.vocab_size

    model = TextGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)
    num_articles = 200
    dataset = PersianWikipediaDataset(num_articles=num_articles)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('finetuned_text_generator.pth'))
    model.to(device)

    perplexity, rouge_results, bleu_results, meteor_results = evaluate_model(model, dataset, tokenizer, device)

    print(f'Perplexity: {perplexity}')
    print(f'ROUGE Results: {rouge_results}')
    print(f'BLEU Results: {bleu_results}')
    print(f'METEOR Results: {meteor_results}')
