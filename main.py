import argparse
import sys
import random
import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.models import load_model, MyModel
from utils.dataset import SOPDataset
from utils.sampler import TripletLoss, MarginLoss
from utils.evaluation import evaluate_recall
from utils.utils import plot_tsne

from tqdm import tqdm

def seed_everything(seed):
    """
    Fixing all seeds for reproducibillity
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--out-dir', default='./output', type=str,
                        help='output directory to save log file and model weight')
    parser.add_argument('--data-dir', default='./dataset/stanford_products', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--test', action='store_true', 
                       help='only testing the model')
    parser.add_argument('--eval-it', default=10, type=int,
                        help='validation interval in training')
    parser.add_argument('--topk', default=[1, 5, 10], nargs='+', type=int,
                       help='list of k values for Recall@k')
    
    # Training
    parser.add_argument('--scratch', action='store_true',
                        help='training from scratch')
    parser.add_argument('--resume', default=None, type=str, 
                        help='path of the pretrained moodel')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='set to a non-zero value when retraining')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weight-decay', default=4e-5, type=float)
    
    # Model
    parser.add_argument('--embedding-dim', default=512, type=int,
                        help='dimension of the output embedding vectors')
    
    # Augmentation
    parser.add_argument('--rnd-resize', action='store_true')
    parser.add_argument('--norm-m', default=[0.485, 0.456, 0.406],
                        help='mean values of image normalization')
    parser.add_argument('--norm-std', default=[0.228, 0.224, 0.225], type=list,
                        help='standard deviation value of image normalization')
    parser.add_argument('--h-prob', default=0, type=float,
                        help='probability of horizontal flip augmentation')
    parser.add_argument('--color-jitter', default=False, type=bool,
                        help='whether applying color jitter augmentation')
    
    # Loss
    parser.add_argument('--sampling', default='random', type=str)
    parser.add_argument('--loss', default='margin', type=str,
                        help='choosing loss function and sampling method')
    parser.add_argument('--tri-margin', default=0.2, type=float,
                        help='margin value for TripletLoss or MarginLoss')
    parser.add_argument('--beta', default=1.2, type=float,
                        help='trainable parameter in MarginLoss')
    parser.add_argument('--beta-lr', default=5e-5, type=float,
                        help='learning rate for class margin parameters in MarginLoss')
    
    # LR scheduler
    parser.add_argument('--lr-scheduler', default='step', type=str,
                        help='learninng rate scheduler')
    parser.add_argument('--tau', default=[100, 150], nargs='+', type=int,
                        help='stepsize before reducing learning rate')
    parser.add_argument('--gamma', default=0.3, type=float,
                        help='learning rate reduction after tau epochs')
    
    # Visualization
    parser.add_argument('--vis', action='store_true', 
                        help='plot t-SNE visualization')
    return parser

def main(args):
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    seed_everything(args.seed)
    
    log_file = open(f'{args.out_dir}/log_everything.log', 'a')
    sys.stdout = log_file
    
    device = torch.device(args.device)
    
    tf_list = [transforms.RandomResizedCrop((224, 224)) if args.rnd_resize 
               else transforms.Resize((224,224))]
        
    if args.h_prob != 0:
        tf_list.append(transforms.RandomHorizontalFlip(p=args.h_prob))
    if args.color_jitter:
        tf_list.append(transforms.ColorJitter(brightness=1, contrast=1, saturation=1))
    
    tf_list.append(transforms.ToTensor())
    tf_list.append(transforms.Normalize(args.norm_m, args.norm_std))
    
    train_transform = transforms.Compose(tf_list)
    
    
    # Train 
    train_dataset = SOPDataset(
        data_dir=args.data_dir,
        mode='train_split',
        transform=train_transform,
        samples_per_class=4
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation
    val_dataset = SOPDataset(
        data_dir=args.data_dir,
        mode='val_split',
        transform=train_transform,
        samples_per_class=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=True
    )
    
    # Test
    query_dataset = SOPDataset(
        data_dir=args.data_dir,
        mode='query_split',
        transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.norm_m, std=args.norm_std)
        ])
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=int(2*args.batch_size),
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )
    
    db_dataset = SOPDataset(
        data_dir=args.data_dir,
        mode='db_split',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.norm_m, std=args.norm_std)
        ])
    )
    
    db_loader = DataLoader(
        db_dataset,
        batch_size=int(2*args.batch_size),
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )
    
    # model = load_model(embedding_dim=args.embedding_dim, pretrained=not args.scratch)
    model = MyModel(embedding_dim=args.embedding_dim, pretrained=not args.scratch)
    model = model.to(device)
    
    train_epoch = args.epochs
    
    if args.resume is not None:
        model = torch.load(f'{args.resume}', map_location=device)
        train_epoch -= args.start_epoch
    
    # opt_param_groups = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    args.lr_proj = args.lr * 2
    args.weight_decay_proj = args.weight_decay / 2
    
    opt_param_groups = [{'params': model.backbone.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    opt_param_groups.append({'params': model.proj_head.parameters(), 'lr': args.lr_proj, 'weight_decay': args.weight_decay_proj})
    
    if args.loss == 'margin':
        criterion = MarginLoss(
            margin=args.tri_margin, 
            beta=args.beta, 
            n_classes=len(train_loader.dataset.avail_classes), 
            sampling_method=args.sampling).to(device)
        opt_param_groups.append({'params':criterion.parameters(), 'lr':args.beta_lr, 'weight_decay':0})
    elif args.loss == 'triplet':
        criterion = TripletLoss(margin=args.tri_margin, sampling_method=args.sampling).to(device)
    else:
        raise NotImplemented
    
    optimizer = optim.Adam(opt_param_groups)
    
    if args.lr_scheduler == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.tau, gamma=args.gamma)
    else:
        raise Exception(f'No scheduling option! | {args.lr_scheduler}')
    
    print(args)
    
    if not args.test:
        model = model.train()
        patience_cnt = 0
        
        print('Training start')
        
        for epoch in tqdm(range(train_epoch), total=args.epochs, initial=args.start_epoch):
            # try:
            train_loss = train_one_epoch(model, 
                            train_loader, 
                            optimizer,
                            criterion,
                            epoch=epoch,
                            resume=args.resume,
                            device=device
                            )
            
            lr_scheduler.step()
                
            # except Exception as e:
            #     print(f"Error occured! | {e}")
            #     print(f"Epoch [{epoch}] Training is stopped")
            #     torch.save(model.state_dict(), f'{args.out_dir}/model_{epoch}.pt')
            #     return     
                
            if epoch == 0 or (epoch+1) % args.eval_it == 0:
                val_loss = eval_engine(model, 
                                        val_loader, 
                                        criterion,
                                        epoch=epoch,
                                        device=device
                                        )
                if epoch == 0:
                    best_loss = val_loss       
                    
                if best_loss < val_loss:
                    patience_cnt += 1
                    if patience_cnt >= 5:
                        print('early stopped')
                        break
                else:
                    best_loss = val_loss
                    patience_cnt = 0
                    print(f"New best model saved(Epoch [{epoch+1}])")
                    torch.save(model.state_dict(), f"{args.out_dir}/best_model.pt")
            
                print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss {train_loss:.5f} | Learning Rate {optimizer.param_groups[0]['lr']} | Validation Loss: {val_loss:.5f}")
            else:
                print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss {train_loss:.5f} | Learning Rate {optimizer.param_groups[0]['lr']}")
                
                    
        print("Training is done!")
        torch.save(model.state_dict(), f'{args.out_dir}/model_{epoch+1}.pt')   
    
    model.load_state_dict(torch.load(f'{args.out_dir}/best_model.pt'))
    model.to(device)
    
    scores = evaluate_recall(model, query_loader, db_loader, args.topk, device)
    
    output = ' '.join(f"{k} {v:.4f}" for k, v in scores.items())
    print(f"Best Model {output}")
    
    if args.vis:
        test_dataset = SOPDataset(
            data_dir=args.data_dir,
            mode='test',
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.norm_m, std=args.norm_std)
            ])
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(2*args.batch_size),
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )
        plot_tsne(model, test_loader, device)
        

def train_one_epoch(model, loader, optimizer, criterion, epoch, **kwargs):
    if kwargs['resume'] is not None:
        epoch += kwargs['resume']
                
    total_loss = []
    
    for _, data in enumerate(tqdm(loader, desc='Training')):
        images = data['image']
        labels = data['label']
        
        images = images.to(kwargs['device'])
        labels = labels.to(kwargs['device'])
        
        embeddings = model(images)
        
        loss = criterion(embeddings, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss.item())

    epoch_loss = sum(total_loss) / len(total_loss)
    
    return epoch_loss

def eval_engine(model, loader, criterion, **kwargs):
    
    model.eval()
    
    total_loss = []
    
    for _, data in enumerate(tqdm(loader, desc='Validating')):
        images = data['image']
        labels = data['label']
        
        images = images.to(kwargs['device'])
        labels = labels.to(kwargs['device'])
        
        embeddings = model(images)
        
        loss = criterion(embeddings, labels)
        
        model.zero_grad()
        loss.backward()
        
        total_loss.append(loss.item())

    epoch_loss = sum(total_loss) / len(total_loss)
        
    return epoch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'simple image retrieval project', parents=[get_args()], add_help=False
        )
    args = parser.parse_args()
    main(args)