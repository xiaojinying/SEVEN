from transformers.optimization import Adafactor, get_scheduler
from torch.optim import AdamW
from typing import Any, Dict, Union
import math
import torch

import argparse
from easydict import EasyDict as edict
from model import get_metrics,compute_metrics
import numpy as np
from torch import nn
import random
import os
import torch.nn.functional as F
from transformers import glue_compute_metrics
from model import CustomBERTModel, stsb_model
import sys



def compute_loss(model, inputs):
    """

    """
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)

    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        loss = model.loss(logits, labels)
    else:
        logits = outputs['logits']
        loss = model.loss(logits, labels)
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss, torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())





def eval_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    model.eval()
    model.zero_grad()
    loss, _, metric, metric_1, _ = compute_loss(model, inputs)

    return loss.detach(), metric, metric_1


def matthews_correlation(y_true, y_pred):

    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    tp = np.sum((y_true * y_pred) > 0)
    tn = np.sum((y_true + y_pred) == 0)
    fp = np.sum((y_true < y_pred))
    fn = np.sum((y_true > y_pred))

    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if denominator == 0:
        mcc = 0
    else:
        mcc = numerator / denominator

    return mcc

def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def create_optimizer(model, adafactor=None, weight_decay=0.0, learning_rate=2e-5,
                     adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,config=None,batch_num=None):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = Adafactor if adafactor else AdamW
    if adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,

        }
    optimizer_kwargs["lr"] = learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


def create_scheduler(optimizer, lr_scheduler_type: str="linear", num_training_steps: int=10, 
                     warmup_steps: int=0, warmup_ratio: float=0.0):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    lr_scheduler=torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,total_iters=num_training_steps)
    return lr_scheduler

def load_model(model_checkpoint,task,device):
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    if task == 'stsb':
        model = stsb_model(num_labels=num_labels, task=task).to(device)
        for name, param in model.named_modules():
            if name == 'bert.classifier':
                setattr(param, "is_classifier", True)
    else:
        model = CustomBERTModel(model_checkpoint, num_labels=num_labels, task=task).to(device)
    return model

def init_log():
    import logging

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个文件处理器，将日志写入到文件中
    file_handler = logging.FileHandler('example.log')
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个日志记录器，并将处理器添加到记录器中
    logger = logging.getLogger('my_logger')
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def train_eval_loop(config,model,train_epoch_iterator,eval_epoch_iterator,optimizer,pruning,device,log):
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    l = length // 3
    metric_epoch = {}
    steps = config.epoch
    iter_num = 0
    metric_epoch['loss'] = []
    if model.metric != None:
        metric_name = f"{model.metric.__class__.__name__}"
        metric_epoch[f"{model.metric.__class__.__name__}"] = []
    if model.metric_1 != None:
        metric_1_name = f"{model.metric_1.__class__.__name__}"
        metric_epoch[f"{model.metric_1.__class__.__name__}"] = []

    pruning.init_mask()
    for epoch in range(steps):
        metric_batch = {}
        metric_batch_test = {}
        metric_batch['loss'] = []
        metric_batch_test['loss'] = []

        if model.metric != None:
            metric_batch[f"{model.metric.__class__.__name__}"] = []
            metric_batch_test[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch[f"{model.metric_1.__class__.__name__}"] = []
            metric_batch_test[f"{model.metric_1.__class__.__name__}"] = []
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        iterator_eval = iter(eval_epoch_iterator)
        trange_eavl = range(len(eval_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.train()
            optimizer.zero_grad()
            step_loss, logit, step_metric, step_metric_1, _ = compute_loss(model, inputs)
            if iter_num >= config.t_i and iter_num < (config.t_i + config.iter_num) - 1:
                if config.pruning_algo == 'soft_movement':
                    r= (pruning.keep_ratio) +(1-pruning.keep_ratio)*((1-(iter_num-config.t_i + 1) / pruning.config.iter_num)**(3))
                    reg = 0
                    num=0
                    g = torch.autograd.grad(step_loss, [i.weight for i in pruning._prunable_modules()], create_graph=True)
                    pruning._param_gradients=dict()
                    for module,grad in zip(pruning._prunable_modules(),g):
                        pruning._param_gradients[module]=grad
                        reg+=torch.norm(torch.sigmoid(grad*module.weight),1)
                        num+=grad.numel()
                    step_loss+=(r*reg/torch.tensor(num))

            step_loss.backward()
            optimizer.step()
            if pruning.lr_scheduler is not None:
                pruning.lr_scheduler.step()

            metric_batch['loss'].append(step_loss.item())
            if model.metric != None:
                metric_batch[f"{model.metric.__class__.__name__}"].append(list(step_metric.values())[0])
            if model.metric_1 != None:
                metric_batch[f"{model.metric_1.__class__.__name__}"].append(list(step_metric_1.values())[0])

            if step % l == 0:
                s=f'train:epoch({epoch})[{step}]/[{length}] lr {optimizer.state_dict()["param_groups"][0]["lr"]} loss {sum(metric_batch["loss"]) / len(metric_batch["loss"])}'
                if model.metric != None:
                    s+=','
                    s+=(f"{model.metric.__class__.__name__}: {sum(metric_batch[model.metric.__class__.__name__]) / len(metric_batch[model.metric.__class__.__name__])}")
                if model.metric_1 != None:
                    s += ','
                    s+=(f"{model.metric_1.__class__.__name__}: {sum(metric_batch[model.metric_1.__class__.__name__]) / len(metric_batch[model.metric_1.__class__.__name__])}")
                log.info(s)
            # if iter_num>=config.t_i and iter_num<(config.t_i+config.iter_num)-1 and 'movement' not in config.pruning_algo:
            if iter_num >= config.t_i and iter_num < (config.t_i + config.iter_num) - 1:
                pruning.model_masks(iter_num=iter_num-config.t_i)
                if iter_num==config.t_i:
                    pruning.log.info('------------pruning------------')
                if iter_num==(config.t_i+config.iter_num)-2:
                    remain=0
                    total=0
                    for _, mask in pruning.masks.items():
                        remain += mask.sum().item()
                        total += mask.numel()
                    pruning.log.info('------------pruning end,true remain ratio:'+str(remain/total)+'------------')
            iter_num += 1

        # Eval Loop
        if config.dataset == 'stsb' or config.dataset == 'cola':
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():

                model.eval()
                model.zero_grad()
                if config.dataset == 'stsb':
                    ref = np.array([])
                    pre = np.array([])
                else:
                    ref = np.array([], dtype=np.float64)
                    pre = np.array([], dtype=np.float64)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    if "labels" in inputs:
                        labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    if config.dataset == 'stsb':
                        predictions = outputs.logits.squeeze()
                    else:
                        predictions = outputs['logits']
                    if config.dataset == 'stsb':
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)
                        pre = np.concatenate((pre, torch.clone(predictions).detach().cpu().numpy()), axis=0)
                    else:
                        for i in range(predictions.shape[0]):
                            pre = np.append(pre, 0 if predictions[i][0] > predictions[i][1] else 1)
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)

                if config.dataset == 'stsb':
                    log.info(str(glue_compute_metrics('sts-b', pre, ref)))
                else:
                    log.info('matthews_correlation:'+str(matthews_correlation(ref, pre)))
        else:
            with torch.no_grad():
                for step in trange_eavl:
                    inputs = prepare_inputs(next(iterator_eval), device)
                    step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
                    metric_batch_test['loss'].append(step_loss.item())
                    if model.metric != None:
                        metric_batch_test[f"{model.metric.__class__.__name__}"].append(list(step_metric.values())[0])
                    if model.metric_1 != None:
                        metric_batch_test[f"{model.metric_1.__class__.__name__}"].append(
                            list(step_metric_1.values())[0])
                    if step == len(eval_epoch_iterator) - 1:
                        log.info('test---')
                        s=f'loss: {sum(metric_batch_test["loss"]) / len(metric_batch_test["loss"])}'
                        if model.metric != None:
                            s += ','
                            s+=(f"{model.metric.__class__.__name__}:{sum(metric_batch_test[model.metric.__class__.__name__]) / len(metric_batch_test[model.metric.__class__.__name__])}")
                        if model.metric_1 != None:
                            s += ','
                            s+=(f"{model.metric_1.__class__.__name__}: {sum(metric_batch_test[model.metric_1.__class__.__name__]) / len(metric_batch_test[model.metric_1.__class__.__name__])}")
                        log.info(s)




def init_config():
    gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--batchsize', type=int, default=None)
    parser.add_argument('--prune_batchsize', type=int, default=None)
    parser.add_argument('--target_ratio', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--iter_num', type=int, default=None)
    parser.add_argument('--alpha_1', type=float, default=None)
    parser.add_argument('--alpha_2', type=float, default=None)
    parser.add_argument('--pruning_algo', type=str, default=None,help='choose one:[SEVEN,PLATON,movement,soft_movement]')
    parser.add_argument('--t_i', type=int, default=None)

    args = parser.parse_args()
    base_config = {'dataset': "sst2",
                   'batchsize': 32, 'epoch': 10, 'learning_rate': 2e-5, 'target_ratio': 0.90,
                   'seed':3404,'iter_num':100,'grad':1,'alpha_1':0.8,'alpha_2':0.9,
                   'prune_batchsize':32,'t_i':200,'pruning_algo':'SEVEN'}
    config = edict(base_config)
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    if config.pruning_algo not in ['SEVEN','PLATON','movement','soft_movement']:
        raise ValueError("Unsupported pruning_algo")

    dataset_params = {
        'mnli': {'t_i': 5400, 'iter_num': 16600,'target_ratio':0.7},
        'qqp': {'t_i': 5400, 'iter_num': 16600,'target_ratio':0.9},
        'rte': {'t_i': 200, 'iter_num': 1000,'target_ratio':0.6},
        'qnli': {'t_i': 2000, 'iter_num': 12000,'target_ratio':0.7},
        'mrpc': {'t_i': 300, 'iter_num': 600,'target_ratio':0.5},
        'sst2': {'t_i': 1000, 'iter_num': 4000,'target_ratio':0.6},
        'cola': {'t_i': 500, 'iter_num': 1000,'target_ratio':0.5},
        'stsb': {'t_i': 500, 'iter_num': 2000,'target_ratio':0.5},
    }

    if config.dataset in dataset_params:
        config.t_i = dataset_params[config.dataset]['t_i']
        config.iter_num = dataset_params[config.dataset]['iter_num']
        config.target_ratio = dataset_params[config.dataset]['target_ratio']

    return config


def seed_torch(seed=3404):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
