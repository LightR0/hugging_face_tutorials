import time
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from torch.optim import Adam
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model_path', default="bert_pretrain_model", type=str, help='')
    parser.add_argument('--vocab_path', default="bert_pretrain_model/vocab.txt", type=str, help='')
    parser.add_argument('--save_model_path', default="save_model", type=str, help='')
    parser.add_argument('--label_data_path', default="cate_dict.txt", type=str, help='')
    parser.add_argument('--final_model_path', default="final_model", type=str, help='')
    parser.add_argument('--train_data_path', default='train_data.txt', type=str, help='')
    parser.add_argument('--eval_data_path', default='test_data.txt', type=str, help='')
    parser.add_argument('--max_sequence_length', default=60, type=int, required=False, help='')
    parser.add_argument('--batch_size', default=256, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=3, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=3e-5, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=100, type=int, required=False, help='print log steps')
    return parser.parse_args()


class MyDataset(Dataset):
    def __init__(self, text_list, label_list, tokenizer, max_sequence_len, device):
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_list = label_list
        self.len = len(label_list)
        for text in tqdm(text_list):
            text = text[:max_sequence_len - 2]
            title_ids = tokenizer.encode_plus(text, padding='max_length', max_length=max_sequence_len)
            self.input_ids.append(title_ids['input_ids'])
            self.token_type_ids.append(title_ids["token_type_ids"])
            self.attention_mask.append(title_ids["attention_mask"])

    def __getitem__(self, index):
        tmp_input_ids = self.input_ids[index]
        tmp_token_type_ids = self.token_type_ids[index]
        tmp_attention_mask = self.attention_mask[index]
        tmp_label = self.label_list[index]
        output = {"input_ids": torch.tensor(tmp_input_ids).to(device),
                  "token_type_ids": torch.tensor(tmp_token_type_ids).to(device),
                  "attention_mask": torch.tensor(tmp_attention_mask).to(device)}
        return output, tmp_label

    def __len__(self):
        return self.len


def load_label(data_path):
    label_id_name = {}
    label_name_id = {}
    with open(data_path, "r") as f:
        for line in f:
            line_split = line.strip().split("\t")
            cate_name, cate_id = line_split
            label_id_name[int(cate_id)] = cate_name
            label_name_id[cate_name] = int(cate_id)
    return label_name_id, label_id_name


def data_loader(data_path, tokenizer, label_name_id, max_sequence_len, batch_size, shuffle, device):
    text_list = []
    label_list = []
    cnt = 0
    with open(data_path, 'rb') as f:
        data = f.read().decode("utf-8")
        train_data = data.split("\n")
        print("数据总行数:{}".format(len(train_data)))
        for text in train_data:
            text_split = text.split("\t")
            if len(text_split) != 2 and len(text_split) <= 5:
                continue
            cate, title = text_split
            text_list.append(title)
            label_list.append(label_name_id[cate])
            cnt += 1
    #             if cnt == 500000:
    #                 break
    dataset = MyDataset(text_list, label_list, tokenizer, max_sequence_len, device)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)

    return dataloader


def load_model(args, num_labels):
    model = BertForSequenceClassification.from_pretrained(args.pretrain_model_path,
                                                          num_labels=num_labels)
    tokenizer = BertTokenizer(args.vocab_path)
    return model, tokenizer


def compute_acc(logits, label):
    predicted_class_id = torch.tensor([w.argmax().item() for w in logits])
    return float((predicted_class_id == label).float().sum()) / label.shape[0]


def train(args, model, dataloader, device):
    num_training_steps = args.epochs * len(dataloader)
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    model.to(device)
    model.train()
    batch_steps = 0
    for epoch in range(args.epochs):
        for batch, label in dataloader:
            batch_steps += 1
            outputs = model(**batch, labels=label.to(device))
            loss = outputs.loss
            logits = outputs.logits
            acc = compute_acc(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}, acc {}".format(
                    epoch + 1, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss,
                    acc))

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.save_model_path)


def evaluate(args, dataloader, num_labels, device):
    model = BertForSequenceClassification.from_pretrained(args.save_model_path, num_labels=num_labels)
    model.to(device)
    model.eval()
    loss_list = []
    acc_list = []
    with torch.no_grad():
        for batch, label in dataloader:
            outputs = model(**batch, labels=label.to(device))
            loss = outputs.loss
            logits = outputs.logits
            acc = compute_acc(logits, label)
            loss_list.append(float(loss))
            acc_list.append(float(acc))
    print("loss: {},".format(np.mean(loss_list)),
          "accuracy: {}.".format(np.mean(acc_list)))


def predict(args, device, text, tokenizer, label_dict):
    model = BertForSequenceClassification.from_pretrained(args.save_model_path, num_labels=len(label_dict.keys()))
    model.to(device)
    model.eval()
    time_start = time.time()
    with torch.no_grad():
        text = text[:args.max_sequence_length - 2]
        inputs = tokenizer.encode_plus(text,
                                       padding='max_length',
                                       max_length=args.max_sequence_length,
                                       return_tensors="pt")
        inputs = {key: torch.tensor(value).to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        print("predict time cost {}".format(time.time() - time_start))
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        predicted_class_name = label_dict[predicted_class_id]
    print("title {}".format(text))
    print("predict category {}".format(predicted_class_name))


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    label_name_id, label_id_name = load_label(args.label_data_path)
    model, tokenizer = load_model(args, num_labels=len(label_name_id.keys()))
    print(get_parameter_number(model))
    train_dataloader = data_loader(args.train_data_path,
                                   tokenizer,
                                   label_name_id,
                                   args.max_sequence_length,
                                   args.batch_size,
                                   True,
                                   device)
    time_start = time.time()
    train(args, model, train_dataloader, device)
    print("train time cost {}".format(time.time() - time_start))
    eval_dataloader = data_loader(args.eval_data_path,
                                  tokenizer,
                                  label_name_id,
                                  args.max_sequence_length,
                                  args.batch_size,
                                  False,
                                  device)
    time_start = time.time()
    evaluate(args, eval_dataloader, num_labels=len(label_name_id.keys()), device=device)
    print("eval time cost {}".format(time.time() - time_start))

    text = "DICIHOZZ9小白鞋女2018秋季新品韩版百搭休闲时尚复古帆布鞋女 墨绿 36"
    predict(args, device, text, tokenizer, label_id_name)
