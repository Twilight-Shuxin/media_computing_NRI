import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

from src.nri_decoder import MLPDecoder
from src.utils import mask, load_data


parser = argparse.ArgumentParser()

def parse_args():
    parser = argparse.ArgumentParser(description="code for training decoder")
    parser.add_argument('-n','--node_dims', dest='node_dims', default=4) 
    parser.add_argument('-sd','--sep_hidden_dims', dest='sep_hidden_dims', default=256)
    parser.add_argument('-so','--sep_out_dims', dest='sep_out_dims', default=256)
    parser.add_argument('-hid','--hidden_dims', dest='hidden_dims', default=256)
    parser.add_argument('-e','--epoch', dest='epoch_num', default=30)
    parser.add_argument('-ps','--time_step', dest='time_steps_test',default=49)
    parser.add_argument('-pred_s','--pred_step', dest='pred_steps',default=1) # 49->4
    parser.add_argument('-et','--edge_types', dest='edge_types',default=2)
    parser.add_argument('-dr','--dropout_rate', dest='dropout', default=0.05)
    parser.add_argument('-nn','--num_nodes', dest='num_nodes', default=5)
    parser.add_argument('-cuda', '--cuda',dest ='cuda', default = True)

    return parser.parse_args()

def reconstruction_error(pred, target, variance=1):
    # assume variance to be 1
    loss = ((pred - target)**2) / (2 * variance)
    return loss.sum() / (loss.size(0) * loss.size(1))

def train(args):
    epoch_nums = args.epoch_num
    optimizer = optim.Adam(model.parameters())

    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    timestr = time.strftime("%Y%m%d_%H%M")
    
    parent_folder = f'../saved_model/decoder/{timestr}'
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)
        
    minimum_loss = 1000
    for i in range(epoch_nums):
        loss_train = []
        loss_val = []
        #  model.train tells your model that you are training the model
        model.train()
        # All optimizers implement a step() method, that updates the parameters. It can be used in two ways:
        # Learning rate scheduling should be applied after optimizer’s update.
        
        model_path = f'../saved_model/decoder/{timestr}/{i}/'
        best_model_path = model_path

        for batch_index, (input_batch, relations) in enumerate(train_loader):
            rel_type_onehot = torch.FloatTensor(input_batch.size(0), rec_mask.size(0),args.edge_types)
            rel_type_onehot.zero_()
            rel_type_onehot.scatter_(2, relations.view(input_batch.size(0), -1, 1), 1)
            
            if args.cuda and torch.cuda.is_available():
                input_batch.cuda()
                rel_type_onehot.cuda()
                
            output = model(input_batch, rel_type_onehot, send_mask, rec_mask, args.pred_steps)
            target = input_batch[:, :, 1:, :]
            
            loss = reconstruction_error(output, target, 1)
            loss_baseline = reconstruction_error(input_batch[:, :, :-1, :], input_batch[:, :, 1:, :])
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_train.append(loss.item())

        
        model.eval()
        for batch_index, (input_batch, relations) in enumerate(valid_loader):
            rel_type_onehot = torch.FloatTensor(input_batch.size(0), rec_mask.size(0),args.edge_types)
            rel_type_onehot.zero_()
            rel_type_onehot.scatter_(2, relations.view(input_batch.size(0), -1, 1), 1)
            
            if args.cuda and torch.cuda.is_available():
                input_batch.cuda()
                rel_type_onehot.cuda()

            output = model(input_batch, rel_type_onehot, send_mask, rec_mask, args.pred_steps)
            # print(target.shape)
            target = input_batch[:, :, 1:, :]
            
            loss = reconstruction_error(output, target, 1)
            
            loss_val.append(loss.item())
        
        if np.mean(loss_val) < minimum_loss - 1e-4:
            minimum_loss = np.mean(loss_val)
            best_model_path = model_path
            if not os.path.exists(best_model_path):
                os.mkdir(best_model_path)
            torch.save(model.state_dict(), best_model_path + 'model.ckpt')
            pickle.dump({'args': args}, open(best_model_path + 'model_args.pkl',"wb"))
            print('-----------------------------------------------')
            print(f'epoch {i} decoder training finish. Model performance improved.')
            print(f'validation loss {np.mean(loss_val)}')
            print(f'save best model to {best_model_path}')
            
    return best_model_path

    #----------------------------------------------
def test(args, best_model_path):
    loss_test = []
    loss_baseline_test = []
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.load_state_dict(torch.load(best_model_path+'model.ckpt'))
    for batch_idx, (input_batch, relations) in enumerate(test_loader):
        rel_type_onehot = torch.FloatTensor(input_batch.size(0), rec_mask.size(0),args.edge_types)
        rel_type_onehot.zero_()
        rel_type_onehot.scatter_(2, relations.view(input_batch.size(0), -1, 1), 1)
        input_batch = input_batch[:, :, -args.time_steps_test:, :] # .contiguous()
        
                    
        if args.cuda and torch.cuda.is_available():
            input_batch.cuda()
            rel_type_onehot.cuda()
        target = input_batch[:,:,1:,:]
        output = model(input_batch, rel_type_onehot, send_mask, rec_mask, args.pred_steps)       


        loss = reconstruction_error(output, target)
        loss_baseline = reconstruction_error(input_batch[:, :, :-1, :], input_batch[:, :, 1:, :])
        
        loss_test.append(loss)
        loss_baseline_test.append(loss_baseline)
        
    print('-------------testing finish-----------------')
    print(f'load model from: {best_model_path}')
    print(f'test reconstruction error: {sum(loss_test) / len(loss_test)}')
    print(f'test baseline loss: {np.mean(loss_baseline_test)}')

    
if __name__=="__main__":
    # args, model, loaders are global variable
    args = parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)

    send_mask, rec_mask = mask(args.num_nodes)    
    model = MLPDecoder(args.node_dims, args.sep_hidden_dims, args.sep_out_dims, args.edge_types, args.hidden_dims, args.dropout)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        model.cuda()
        send_mask.cuda()
        rec_mask.cuda()
    if args.cuda and not torch.cuda.is_available():
        print('No GPU provided.')
    else:
        print('Run in GPU')
        
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(batch_size=5, suffix='_springsLight5')

    best_model_path = train(args)  

    test(args, best_model_path)