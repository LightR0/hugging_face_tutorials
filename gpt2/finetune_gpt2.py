import os
import time
import torch
import argparse
import numpy as np
from rouge import Rouge
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="gpt2通用中文模型", type=str, help='')
    parser.add_argument('--vocab_path', default="gpt2通用中文模型/vocab.txt", type=str, help='')
    parser.add_argument('--save_model_path', default="save_model", type=str, help='')
    parser.add_argument('--final_model_path', default="final_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=64, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=3, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    return parser.parse_args()


def load_model(model_path, vocab_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = BertTokenizer(vocab_file=vocab_path)
    return model, tokenizer


def rouge(not_ignore, shift_labels, preds):
    main_rouge = Rouge()
    true_length = [w.sum() for w in not_ignore.float()]
    rouge_labels = []
    rouge_predicts = []
    for idx, tmp_len in enumerate(true_length):
        tmp_labels = shift_labels[idx][:int(tmp_len)]
        rouge_labels.append(" ".join([str(w) for w in tmp_labels.tolist()]))
        tmp_pred = preds[idx][:int(tmp_len)]
        rouge_predicts.append(" ".join([str(w) for w in tmp_pred.tolist()]))
    rouge_score = main_rouge.get_scores(rouge_predicts, rouge_labels, avg=True)
    return rouge_score


def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(tokenizer.pad_token_id)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    rouge_score = rouge(not_ignore, shift_labels, preds)
    return loss, accuracy, rouge_score


def collate_fn(batch):
    input_ids = []
    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def data_loader(args, train_data_path, tokenizer, shuffle):
    data_list = []
    with open(train_data_path, 'rb') as f:
        data = f.read().decode("utf-8")
        train_data = data.split("\n")
        print("数据总行数:{}".format(len(train_data)))
        for text in tqdm(train_data):
            text_split = text.split("\t")
            if len(text_split) != 3:
                continue
            product_word, title, wenan = text_split
            title_ids = tokenizer.encode(title)
            wenan_ids = tokenizer.encode(wenan)
            inputs_ids = title_ids + wenan_ids[1:]
            data_list.append(inputs_ids)
    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader


def train(args, model, dataloader):
    num_training_steps = args.epochs * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    batch_steps = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            # loss = outputs.loss
            loss, acc, rouge_score = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}, rouge-1 {}, rouge-2 {}, rouge-l {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc,
                    rouge_score["rouge-1"]['f'],
                    rouge_score["rouge-2"]["f"],
                    rouge_score["rouge-l"]["f"]))

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.save_model_path)
    # torch.save(model, os.path.join(args.final_model_path, 'gpt2_WenAn.pth'))


def evaluate(dataloader, args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, _ = load_model(args.save_model_path, args.vocab_path)
    model.to(device)
    model.eval()
    loss_list, acc_list, rouge_1_list, rouge_2_list, rouge_l_list = [], [], [], [], []
    batch_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            loss, acc, rouge_score = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss_list.append(float(loss))
            acc_list.append(float(acc))
            rouge_1_list.append(float(rouge_score["rouge-1"]['f']))
            rouge_2_list.append(float(rouge_score["rouge-2"]['f']))
            rouge_l_list.append(float(rouge_score["rouge-l"]['f']))
            print("eval batch {}/{}, loss {}, accuracy {}, rouge-1 {}, rouge-2 {}, rouge-l {}".format(
                batch_steps,
                len(dataloader),
                loss, acc,
                rouge_score["rouge-1"]['f'],
                rouge_score["rouge-2"]["f"],
                rouge_score["rouge-l"]["f"]))
    print("loss: {},".format(np.mean(loss_list)),
          "accuracy: {}.".format(np.mean(acc_list)),
          "rouge-1: {},".format(np.mean(rouge_1_list)),
          "rouge-2: {},".format(np.mean(rouge_2_list)),
          "rouge-l: {}".format(np.mean(rouge_l_list)))
    


def predict(args, text="美丽时分雪绒花美白面膜美白提亮，均匀肤色贴片面膜"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, _ = load_model(args.save_model_path, args.vocab_path)
    model.to(device)
    model.eval()
    time1 = time.time()
    max_length = 30
    input_ids = []
    input_ids.extend(tokenizer.encode(text))
    wenan = ""
    for i in range(max_length):
        input_tensor = torch.tensor([input_ids])
        inputs = {"input_ids": input_tensor.to(device)}
        outputs = model(**inputs)
        logits = outputs.logits
        last_token_id = int(np.argmax(logits[0][-1].detach().to('cpu').numpy()))
        if last_token_id == tokenizer.sep_token_id:
            break
        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_ids.append(last_token_id)
        wenan += last_token
    print("time cost: {}".format(time.time()-time1))
    print(text)
    print(wenan)
    
    
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    model, tokenizer = load_model(args.model_path, args.vocab_path)
    train_dataloader = data_loader(args, args.train_raw_path, tokenizer=tokenizer, shuffle=True)
    eval_dataloader = data_loader(args, args.eval_raw_path, tokenizer=tokenizer, shuffle=False)
    train(args, model, train_dataloader)
    evaluate(eval_dataloader, args=args)

    
