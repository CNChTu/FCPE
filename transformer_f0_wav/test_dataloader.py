import numpy as np
import torch
import time
from transformer_f0_wav.saver.saver import Saver
from transformer_f0_wav.saver import utils
from torch import autocast
from torch.cuda.amp import GradScaler

from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy
from mir_eval.melody import voicing_recall, voicing_false_alarm
import gc

def train(args, initial_global_step, model, optimizer, scheduler, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # run
    num_batches = len(loader_train)
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(args.train.epochs):
        train_one_epoch(loader_train,saver,optimizer,model,dtype,scaler,epoch,args,scheduler,num_batches,loader_test)


def train_one_epoch(loader_train,saver,optimizer,model,dtype,scaler,epoch,args,scheduler,num_batches,loader_test):
    for batch_idx, data in enumerate(loader_train):
        train_one_step(batch_idx, data, saver,optimizer,model,dtype,scaler,epoch,args,scheduler,num_batches,loader_test)


def train_one_step(batch_idx, data, saver,optimizer,model,dtype,scaler,epoch,args,scheduler,num_batches,loader_test):
    # unpack data
    for k in data.keys():
        if not k.startswith('name'):
            data[k] = data[k].to(args.device)
    print("读取数据"+str(batch_idx))