import torch
from transformers import BertModel,BertForQuestionAnswering,BertForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from datasets import load_metric
from transformers import RobertaModel, RobertaTokenizer

def get_metrics(task:str):
    task_to_metric = {
    "cola": ["matthews_correlation", None],
    "sst2": ["accuracy", None],
    "mrpc": ["f1", "accuracy"],
    "stsb": ["pearsonr", 'spearmanr'],
    "qqp": ["f1", "accuracy"],
    "mnli": ["accuracy", None],
    "mnli-mm": ["accuracy", None],
    "qnli": ["accuracy", None],
    "rte": ["accuracy", None],
    "wnli": ["accuracy", None],
    'squad':['exact_match','f1']
    }
    metric = load_metric(task_to_metric[task][0])
    metric_1 = load_metric(task_to_metric[task][1]) if task_to_metric[task][1] else None
    return metric, metric_1

def compute_metrics(predictions, references, metric):
    if f"{metric.__class__.__name__ }" == 'Pearsonr' or f"{metric.__class__.__name__ }" == 'Spearmanr':
        predictions = predictions
    else:
        predictions = torch.argmax(predictions, dim=1)


    return metric.compute(predictions=predictions, references=references)

class BERTClassifierModel(torch.nn.Module):
    def __init__(self, BERT_model, num_labels, task=None):
        super(BERTClassifierModel, self).__init__()
        self.task = task
        self.bert = BERT_model
        self.linear = torch.nn.Linear(768, num_labels)
        self.loss = torch.nn.CrossEntropyLoss()
        setattr(self.linear, "is_classifier", True)

        self.metric, self.metric_1 = get_metrics(task)

    def forward(self, input_ids, attention_mask,token_type_ids=None):
        if token_type_ids is None:
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        # linear_output = self.linear(bert_output['last_hidden_state'][:,0,:].view(-1,768))
        linear_output=self.linear(bert_output.pooler_output)


        # if self.task == "stsb":
        #     linear_output = torch.clip((self.sigmoid(linear_output) * 5), min=0.0, max=5.0)
        output = {"hidden_layer": bert_output, "logits":linear_output}

        return output

    def compute_metrics(self, predictions, references):
        metric, metric_1 = None, None
        if self.metric is not None: 
            metric = compute_metrics(predictions=predictions, references=references, metric=self.metric)
        if self.metric_1 is not None: 
            metric_1 = compute_metrics(predictions=predictions, references=references, metric=self.metric_1)
        return metric, metric_1

class CustomBERTModel(BERTClassifierModel):
    def __init__(self, model_checkpoint, num_labels, task=None):
        BERT_model = BertModel.from_pretrained(model_checkpoint)
        super(CustomBERTModel, self).__init__(BERT_model, num_labels, task)
class stsb_model(torch.nn.Module):
    def __init__(self,num_labels, task=None):
        super(stsb_model, self).__init__()
        self.task = task
        self.bert =BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=num_labels)

        self.loss = torch.nn.MSELoss()
        self.metric, self.metric_1 = get_metrics('stsb')

    def forward(self, input_ids, attention_mask,token_type_ids=None):
        if token_type_ids is None:
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        # linear_output = self.linear(bert_output['last_hidden_state'][:,0,:].view(-1,768))
        # linear_output=self.linear(bert_output)
        # if self.task == "stsb":
        #     linear_output = torch.clip((self.sigmoid(linear_output) * 5.5), min=0.0, max=5.0)
        # output = {"hidden_layer": bert_output, "logits":linear_output}
        return bert_output

    def compute_metrics(self, predictions, references):
        metric, metric_1 = None, None
        if self.metric is not None:
            metric = compute_metrics(predictions=predictions, references=references, metric=self.metric)
        if self.metric_1 is not None:
            metric_1 = compute_metrics(predictions=predictions, references=references, metric=self.metric_1)
        return metric, metric_1

class squad_model(torch.nn.Module):
    def __init__(self,task=None):
        super(squad_model, self).__init__()
        self.task = task
        self.bert =BertForQuestionAnswering.from_pretrained("bert-base-uncased")

        self.loss = torch.nn.MSELoss()
        self.metric, self.metric_1 = get_metrics('stsb')

    def forward(self,input):

        bert_output = self.bert(input)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        # linear_output = self.linear(bert_output['last_hidden_state'][:,0,:].view(-1,768))
        # linear_output=self.linear(bert_output)
        # if self.task == "stsb":
        #     linear_output = torch.clip((self.sigmoid(linear_output) * 5.5), min=0.0, max=5.0)
        # output = {"hidden_layer": bert_output, "logits":linear_output}
        return bert_output

    def compute_metrics(self, predictions, references):
        metric, metric_1 = None, None
        if self.metric is not None:
            metric = compute_metrics(predictions=predictions, references=references, metric=self.metric)
        if self.metric_1 is not None:
            metric_1 = compute_metrics(predictions=predictions, references=references, metric=self.metric_1)
        return metric, metric_1

