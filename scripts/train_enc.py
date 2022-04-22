import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

from src.nri_encoder import MLPEncoder
from src.utils import mask, load_data


def parse_args():
    parser = argparse.ArgumentParser(description="code for training encoder")
    parser.add_argument('-n','--nodes', dest='nodes', default=5) 
    parser.add_argument('-d','--dims', dest='dims', default=4)
    parser.add_argument('-e','--epoch', dest='epoch_num', default=40)
    parser.add_argument('-hid','--hidden', dest='hidden_dims', default=256)
    parser.add_argument('-ts','--time_steps',dest='time_steps',default=49) # 49->4
    parser.add_argument('-et','--edge_types', dest='edge_types',default=2)
    parser.add_argument('-cuda', '--cuda',dest ='cuda', default = True)
    parser.add_argument('-dr','--dropout_rate', dest='dropout', default=0.01)

    return parser.parse_args()



def train(args):
    epoch_nums = args.epoch_num
    optimizer = optim.Adam(model.parameters())
    #.torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs. torch.optim.lr_scheduler.ReduceLROnPlateau allows dynamic learning rate reducing based on some validation measurements.

    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    criterion = nn.CrossEntropyLoss()
    timestr = time.strftime("%Y%m%d_%H%M")
    
    parent_folder = f'../saved_model/encoder/{timestr}'
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)
                
    best_acc = -1
    best_model_path = ''
    for i in range(epoch_nums):
        loss_train = []
        acc_train = []
        loss_val = []
        acc_val = []
        #  model.train tells your model that you are training the model
        model.train()

        model_path = f'../saved_model/encoder/{timestr}/{i}/'
        
        for batch_index, (input_batch, target) in enumerate(train_loader):
            # forward pass
            # print(input_batch.shape)
    
            if args.cuda and torch.cuda.is_available():
                input_batch.cuda()
                target.cuda()
                
            output = model(input_batch, send_mask, rec_mask)
            # Flatten batch dim
            output = output.view(-1, args.edge_types)
            target = target.view(-1)
            
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
        # All optimizers implement a step() method, that updates the parameters. It can be used in two ways:
        # Learning rate scheduling should be applied after optimizer’s update.
        
            optimizer.step()
            scheduler.step()

            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc = correct / pred.size(0)
            loss_train.append(loss.item())
            acc_train.append(acc)

            
            
        # Now tell the model that I want to test it
        model.eval()
        
        for batch_index, (input_batch, target) in enumerate(valid_loader):
            
            if args.cuda and torch.cuda.is_available():
                input_batch.cuda()
                target.cuda()
                
            output = model(input_batch, send_mask, rec_mask)
            # Flatten batch dim
            output = output.view(-1, args.edge_types)
            target = target.view(-1)
            
            loss = criterion(output, target)
            
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc = correct / pred.size(0)

            loss_val.append(loss.item())
            acc_val.append(acc)
            
        if np.mean(acc_val) > best_acc +1e-4 :
            best_acc = np.mean(acc_val)
            best_model_path = model_path
            if not os.path.exists(best_model_path):
                os.mkdir(best_model_path)
            torch.save(model.state_dict(), best_model_path + 'model.ckpt')
            pickle.dump({'args': args}, open(best_model_path + 'model_args.pkl',"wb"))
            print('-----------------------------------------------')
            print(f'epoch {i} encoder training finish. Model performance improved.')
            print(f'validation acc {np.mean(acc_val)}')
            print(f'save best model to {best_model_path}')
    
    return best_model_path

#----------------------------------------------
def test(args, best_model_path):
    loss_test = []
    acc_test = []
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.load_state_dict(torch.load(best_model_path+'model.ckpt'))
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda and torch.cuda.is_available():
            input_batch.cuda()
            target.cuda()

        data = data[:, :, :args.time_steps, :] # .contiguous()
        # print(data.shape)
        output = model(data, send_mask, rec_mask)
        # Flatten batch dim
        output = output.view(-1, args.edge_types)
        target = target.view(-1)
        # print(target)
        loss = nn.functional.cross_entropy(output, target)

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = correct / pred.size(0)

        loss_test.append(loss.item())
        acc_test.append(acc)
       
    print('-------------testing finish-----------------')
    print(f'load model from: {best_model_path}')
    print(f'test loss: {np.mean(loss_test)}')
    print(f'test acc: {np.mean(acc_test)}')

if __name__=="__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    # args, model, loaders are global variable
    np.random.seed(42)
    

    args = parse_args()
    send_mask, rec_mask = mask(args.nodes)

    model = MLPEncoder(args.time_steps * args.dims, args.hidden_dims, args.edge_types, args.dropout)
    
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        model.cuda()
        send_mask.cuda()
        rec_mask.cuda()
    if args.cuda and not torch.cuda.is_available():
        print('No GPU provided.')
    else:
        print('Run in GPU')
        
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(batch_size=10, suffix='_springsLight5')
    
    best_model_path = train(args)  
    test(args, best_model_path)
    