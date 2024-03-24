from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import RobertaModel, RobertaTokenizer
from transformers import BertTokenizer,BertTokenizerFast
import torch
import os

def get_dataloader(task:str, model_checkpoint:str,dataloader_drop_last:bool=True, shuffle:bool=True,
                   batch_size:int=16, dataloader_num_workers:int=2, dataloader_pin_memory:bool=True,tokenizer=None,only_train=False):

    task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True)
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    validation_name = 'validation'

    if task == "mnli":
        validation_name = "validation_matched"
    if task == "mnli-mm":
        validation_name = "validation_mismatched"

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    train_dataset=dataset['train']
    validation_dataset=dataset[validation_name]
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    validation_dataset = validation_dataset.map(preprocess_function, batched=True)

    columns_to_return = ['input_ids', 'label', 'attention_mask','token_type_ids']
    train_dataset.set_format(type='torch', columns=columns_to_return)
    validation_dataset.set_format(type='torch', columns=columns_to_return)


    
    print(train_dataset)
    print(validation_dataset)
    train_dataloader = DataLoader(
                    train_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    collate_fn=data_collator,
                    # drop_last=dataloader_drop_last,
                    num_workers=dataloader_num_workers,
                    pin_memory=dataloader_pin_memory,
    )
    if only_train:
        return train_dataloader
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=data_collator,
        # drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
    )
    
    return train_dataloader,validation_dataloader