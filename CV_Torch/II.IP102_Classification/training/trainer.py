import sys
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import read_yaml
from models.resnet import ResNet34
from data.dataloader import get_dataloader
from losses import get_criterion
from optimizers import get_optimizer
from schedulers import get_scheduler
from scripts.evaluate import evaluate

class trainer():
    """The class of model train"""
    def __init__(self, config):
        
        self.config = config
        self.device = config['train']['device']
        self.model_name = config['model']['name']
        
        self.epoch = config['train']['num_epochs']
        self.lr = config['train']['learning_rate']
        self.weight_decay = config['train']['weight_decay'] 
        self.early_stop = config['train']['early_stopping_patience']
        
        self.num_class = config['model']['num_classes']
        self.pretrained = config['model']['pretrained']
        
        self.exp_name = config['exp']['name']
        self.exp_dir_name = config['exp']['save_dir']
        self.log_interval = config['exp']['log_interval']
        
        log_dir = os.path.join(self.exp_dir_name, 'log', datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(log_dir)
        
    def start_train(self):
        if self.model_name == 'resnet34':
            model = ResNet34(self.num_class).to(self.device)
        # TODO: ADD MORE MODELS
        elif self.model_name == 'googlenet':
            pass
        
        train_loader, val_loader = get_dataloader(self.config)
        optimizer = get_optimizer(model, self.config)
        criterion = get_criterion(self.config)
        schedulers = get_scheduler(optimizer, self.config, train_loader)
            
        for epoch in range(self.epoch):
            model.train()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 梯度置零
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                evaluate(epoch, model, val_loader, criterion, self.device, 'val', 5, self.writer)

        print("Training Finish!")

if __name__ == "__main__":
    config = read_yaml(r"CV_Torch\II.IP102_Classification\configs\train_config.yaml")
    trainera = trainer(config)
    trainera.start_train()