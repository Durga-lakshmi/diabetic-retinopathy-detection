from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import wandb
import torch.amp as amp

import random
import numpy as np

from hydra.utils import instantiate

from module_training.losses import DistancePenalty,feature_correlation_loss,feature_orthogonality_loss,tv_loss
from module_training.optim import build_optimizer

class Trainer_2c:
    def __init__(self, cfg, train_loader, val_loader, model, evaluator, device, save_path):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.evaluator = evaluator
        self.device = device
        self.save_path = cfg.save.path


        self.log_interval = cfg.log_interval
        self.eval_interval = cfg.eval_interval
        self.epochs = cfg.epochs
        self.patience = cfg.patience


        self.counter = 0        
        self.best_acc = 0.0
        self.best_AUC = 0.0
        self.best_F1 = 0.0
        self.best_Precision = 0.0
        self.best_Recall = 0.0
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_bal_acc = 0.0

        #self.lambda_tv = cfg.train.lambda_tv

        
        # TODO:
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr) # L2 regularization,weight_decay=1e-4 not good(?
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs) # adjust lr #, eta_min=1e-6
        

        # optimizer
        #self.optimizer = instantiate(cfg.optim, params=self.model.parameters())

        # scheduler
        #self.scheduler = None
        #if cfg.sched is not None:
        #    self.scheduler = instantiate(cfg.sched, optimizer=self.optimizer)   


        #pos_weight = torch.tensor([2.0], device=self.device)
        #self.criterion = nn.BCEWithLogitsLoss()  # since output is sigmoid(0–1), pos_weight
        #self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.criterion = instantiate(cfg.train_module.loss)
        self.scaler = amp.GradScaler(device=self.device) # AMP

        


        #self.aug_plugin = MixupCutMix(mixup_alpha=0.4,cutmix_alpha=1.0,prob=1.0) #None #cfg.aug_plugin
        #cfg.aug_plugin = MixupCutMix(mixup_alpha=0.4,cutmix_alpha=1.0,prob=1.0) #batchsize
        self.aug_plugin = instantiate(cfg.train_module.aug_other) if cfg.train_module.aug_other else None

        # Initialize WandB
        #wandb.init(entity="zizi1217-uni-stuttgart",project="diabetic_retinopathy_2c", config={
        #    "epochs": self.epochs,
        #    "optimizer": type(self.optimizer).__name__,
        #    "lr": self.optimizer.param_groups[0]['lr'],
        #})
        #wandb.watch(self.model, self.criterion, log="all", log_freq=100) #（? log_freq=100


    

    # ------------------ Checkpoint ------------------
    def save_checkpoint(self, epoch):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "epochs": self.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
            "patience_counter": self.counter
        }, self.save_path)
        wandb.save(self.save_path)
        print(f"Checkpoint saved at {self.save_path}")

    def load_checkpoint(self, path=None):
        path = path or self.check_path
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.epochs = checkpoint.get("epochs", self.epochs)
            self.best_acc = checkpoint.get("best_acc", 0.0)
            self.counter = checkpoint.get("patience_counter", 0)
            print(f"Loaded checkpoint from {path}, starting at epoch {self.start_epoch}")
        else:
            print(f"No checkpoint found at {path}, starting from scratch.")
        return self.start_epoch


    def train(self, resume=False, checkpoint_path=None):
        # TODO: Initialization
        start_epoch = 0     

        if resume and checkpoint_path is not None:
            start_epoch = self.load_checkpoint(checkpoint_path)

        total_steps = len(self.train_loader) * self.epochs
        #***
        warmup_steps = int(total_steps * 0.1)  # 10%
        def warmup_lr(step, warmup_steps=warmup_steps, lr_max=1e-4):
            if step < warmup_steps:
                return lr_max * (step + 1) / warmup_steps
            return None

        for epoch in range(start_epoch, self.epochs):
            self.model.train() #(!)
            torch.cuda.empty_cache()  # clear cache
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            print(f"=============================================")
            print(f"Epoch [{epoch+1}/{self.epochs}]")

            for step, (x, y_true) in enumerate(self.train_loader):
                x, y_true = x.to(self.device), y_true.to(self.device).float().unsqueeze(1)
                if self.aug_plugin is not None:
                    x, y_true = self.aug_plugin((x, y_true))
                #!y_pred = self.model(x) # Forward pass

                #if self.aug_plugin is not None:
                #    x, y_true = self.aug_plugin((x, y_true))

                # TODO:
                #!loss = self.criterion(y_pred, y_true)

                # TODO: Optimizer step
                # Backward pass
                self.optimizer.zero_grad()
                
                # using autocast for mixed precision
                with amp.autocast('cuda'):
                    #if self.cfg.train.use_tv:

                    #    # ======== 前向：返回 logits + heatmap ========
                    #    y_pred, heatmap = self.model(x, return_heatmap=True)

                    #    # 保持和之前一样的 label 形状处理
                    #    y_true = y_true.view_as(y_pred).float()

                    #    # ======== 分类 loss ========
                    #    loss_cls = self.criterion(y_pred, y_true)

                    #    # ======== TV 正则 ========
                    #    loss_tv = tv_loss(heatmap)

                        # ======== 总 loss ========
                    #    loss = loss_cls + self.lambda_tv * loss_tv

                    #else:   
                    y_pred = self.model(x)
                    y_true = y_true.view_as(y_pred).float()
                    loss = self.criterion(y_pred, y_true)

                    





                #***
                # using scaler to scale the loss and call backward()
                self.scaler.scale(loss).backward()
                # using scaler to step the optimizer
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Warmup
                lr_new = warmup_lr(step + epoch * len(self.train_loader), warmup_steps=warmup_steps, lr_max=1e-4)
                if lr_new is not None:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_new
                else:
                    self.scheduler.step()  # CosineAnnealing step (batch-level)

                # Normal backward pass and optimizer step for sanity check
                #loss.backward()
                #self.optimizer.step()

                total_loss += loss.item()

                # Compute accuracy
                predicted = (y_pred > 0.5).float()
                total_correct += (predicted == y_true).sum().item()
                total_samples += y_true.size(0)

                
                # Batch-level logging
                if step % self.log_interval == 0:
                    # TODO: Logging
                    batch_acc = (predicted == y_true).sum().item() / y_true.size(0)
                    
                    print(f"Epoch [{epoch+1}/{self.epochs}], Step [{step}/{len(self.train_loader)}], Train Loss: {loss.item():.4f}, Batch Accuracy: {batch_acc*100:.2f}%")
                    wandb.log({"batch_loss": loss.item(), "batch_accuracy": batch_acc})

            # each epoch adjust lr
            #self.scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            epoch_accuracy = total_correct / total_samples
            print(f"\n------------ Epoch [{epoch+1}/{self.epochs}] ------------") 
            print(f"Average Train Loss: {avg_loss:.4f}")
            print(f"Average Train Accuracy: {epoch_accuracy*100:.2f}%")
            print(f"learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

            



            # Epoch-level logging
            if epoch % self.eval_interval == 0:
                # TODO: Evaluate on validation set
                val_loss,val_acc,metrics = self.evaluator.eval()
                
                wandb.log({
                "train/epoch": epoch + 1,
                #"lr": self.scheduler.get_last_lr()[0],
                "train/train_loss": avg_loss,
                "val/val_loss": val_loss,
                "val/val_acc": val_acc,
                "val/AUC": metrics["auc"],
                "val/F1": metrics["f1"],
                "val/Precision": metrics["precision"],
                "val/Recall": metrics["recall"]
                })

                # Save checkpoint if accuracy improves

                if self.cfg.train.metric == "f1":

                    if metrics["f1"] >= self.best_F1:
                        self.best_F1 = metrics["f1"]
                        self.best_acc = val_acc
                        self.best_val_loss = val_loss
                        self.counter = 0
                        self.save_checkpoint(epoch) # normal save
                        

                        wandb.log({
                            "best_val_acc": self.best_acc,
                            "best_val_loss": self.best_val_loss,
                            "best_val_AUC": metrics["auc"],
                            "best_val_F1": metrics["f1"],
                            "best_val_Precision": metrics["precision"],
                            "best_val_Recall": metrics["recall"]
                            })
                        

                    else:
                        self.counter += 1
                        print(f"Validation acc did not improve. Counter: {self.counter}/{self.patience}")
                        if self.counter >= self.patience:
                            print("Early stopping triggered.")
                            return


                else:    
                    if val_acc >= self.best_acc:
                        self.best_acc = val_acc
                        self.best_val_loss = val_loss
                        self.counter = 0
                        self.save_checkpoint(epoch) # normal save
                        

                        wandb.log({
                            "best_val_acc": self.best_acc,
                            "best_val_loss": self.best_val_loss,
                            "best_val_AUC": metrics["auc"],
                            "best_val_F1": metrics["f1"],
                            "best_val_Precision": metrics["precision"],
                            "best_val_Recall": metrics["recall"]
                            })
                        

                    else:
                        self.counter += 1
                        print(f"Validation acc did not improve. Counter: {self.counter}/{self.patience}")
                        if self.counter >= self.patience:
                            print("Early stopping triggered.")
                            return


        wandb.log({
            "val/best_val_acc": self.best_acc,
            "val/best_val_loss": self.best_val_loss,
            "val/best_val_AUC": metrics["auc"],
            "val/best_val_F1": metrics["f1"],
            "val/best_val_Precision": metrics["precision"],
            "val/best_val_Recall": metrics["recall"]
        })
                    

        
        wandb.finish()

        


class Trainer_2c_fold:
    def __init__(self, cfg, train_loader, val_loader, model, evaluator, device, save_path,fold,fold_best_accs,fold_best_AUCs,fold_best_losses,fold_best_F1s,fold_best_Precisions,fold_best_Recalls):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.evaluator = evaluator
        self.device = device
        self.save_path = cfg.save.path
        self.check_path = cfg.check_path
        self.fold = fold
        

        self.log_interval = cfg.log_interval
        self.eval_interval = cfg.eval_interval
        self.epochs = cfg.epochs
        self.patience = cfg.patience


        self.counter = 0        
        self.best_acc = 0.0
        self.best_AUC = 0.0
        self.best_F1 = 0.0
        self.best_Precision = 0.0
        self.best_Recall = 0.0
        self.start_epoch = 0
        self.best_val_loss = float('inf')

        self.fold_best_accs = fold_best_accs
        self.fold_best_AUCs = fold_best_AUCs
        self.fold_best_losses = fold_best_losses
        self.fold_best_F1s = fold_best_F1s
        self.fold_best_Precisions = fold_best_Precisions
        self.fold_best_Recalls = fold_best_Recalls

        
        # TODO:
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr) # L2 regularization,weight_decay=1e-4 not good(?
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs) # adjust lr #, eta_min=1e-6
        
        # optimizer
        #self.optimizer = instantiate(cfg.optim, params=self.model.parameters())

        # scheduler
        #self.scheduler = None
        #if cfg.sched is not None:
        #    self.scheduler = instantiate(cfg.sched, optimizer=self.optimizer)

        #pos_weight = torch.tensor([2.0], device=self.device)
        #self.criterion = nn.BCEWithLogitsLoss()  # since output is sigmoid(0–1), pos_weight
        #self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        

        
        self.criterion = instantiate(cfg.train_module.loss)
        self.scaler = amp.GradScaler(device=self.device) # AMP

        
        # Initialize WandB
        wandb.init(entity="zizi1217-uni-stuttgart",project="diabetic_retinopathy_2c_kfold", name=f"fold_{fold+1}", reinit=True, config={
            "epochs": self.epochs,
            "optimizer": type(self.optimizer).__name__,
            "lr": self.optimizer.param_groups[0]['lr'],
            "fold": self.fold + 1
        })
        wandb.watch(self.model, self.criterion, log="all", log_freq=100) #（? log_freq=100



    def save_checkpoint_fold(self, epoch, fold):
        #fold_save_dir = f"./checkpoints/convnext/fold{fold+1}"
        fold_save_dir = os.path.join(self.cfg.save.base_dir, self.cfg.save.model_name, f"fold{fold+1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        model_version = os.path.basename(self.save_path).replace('.pth','')

        save_path = os.path.join(fold_save_dir, f"{model_version}_fold{fold+1}.pth") #best_model_fold{fold+1}.pth"

        torch.save({
            "epoch": epoch,
            "epochs": self.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
            "best_AUC": self.best_AUC,
            "best_F1": self.best_F1,
            "fold": fold,
            "patience_counter": self.counter
        }, save_path)

        wandb.save(save_path)
        print(f"Checkpoint saved at {save_path}")


    def load_checkpoint_fold(self, path=None):
        path = path or self.check_path
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.epochs = checkpoint.get("epochs", self.epochs)
            self.best_acc = checkpoint.get("best_acc", 0.0)
            self.fold = checkpoint.get("fold", 0)
            self.counter = checkpoint.get("patience_counter", 0)
            print(f"Loaded checkpoint from {path}, starting at epoch {self.start_epoch}")
        else:
            print(f"No checkpoint found at {path}, starting from scratch.")
        return self.start_epoch, self.fold


    def train(self, resume=False, checkpoint_path=None):
        # TODO: Initialization
        start_epoch = 0     

        if resume and checkpoint_path is not None:
            start_epoch = self.load_checkpoint(checkpoint_path)

        total_steps = len(self.train_loader) * self.epochs
        warmup_steps = int(total_steps * 0.1)  # 10%
        def warmup_lr(step, warmup_steps=warmup_steps, lr_max=1e-4):
            if step < warmup_steps:
                return lr_max * (step + 1) / warmup_steps
            return None

        for epoch in range(start_epoch, self.epochs):
            self.model.train() #(!)
            torch.cuda.empty_cache()  # clear cache
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            print(f"=============================================")
            print(f"Epoch [{epoch+1}/{self.epochs}]")

            for step, (x, y_true) in enumerate(self.train_loader):
                x, y_true = x.to(self.device), y_true.to(self.device).float().unsqueeze(1)
                #!y_pred = self.model(x) # Forward pass

                # TODO:
                #!loss = self.criterion(y_pred, y_true)

                # TODO: Optimizer step
                # Backward pass
                self.optimizer.zero_grad()
                
                # using autocast for mixed precision
                with amp.autocast('cuda'):
                    y_pred = self.model(x)
                    y_true = y_true.view_as(y_pred).float()
                    loss = self.criterion(y_pred, y_true)

                # using scaler to scale the loss and call backward()
                self.scaler.scale(loss).backward()
    
                # using scaler to step the optimizer
                self.scaler.step(self.optimizer)
                self.scaler.update()

                 # Warmup
                lr_new = warmup_lr(step + epoch * len(self.train_loader), warmup_steps=warmup_steps, lr_max=1e-4)
                if lr_new is not None:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_new
                else:
                    self.scheduler.step()  # CosineAnnealing step (batch-level)


                #!loss.backward()
                #!self.optimizer.step()

                total_loss += loss.item()

                # Compute accuracy
                predicted = (y_pred > 0.5).float()
                total_correct += (predicted == y_true).sum().item()
                total_samples += y_true.size(0)

                
                # Batch-level logging
                if step % self.log_interval == 0:
                    # TODO: Logging
                    batch_acc = (predicted == y_true).sum().item() / y_true.size(0)
                    
                    print(f"Epoch [{epoch+1}/{self.epochs}], Step [{step}/{len(self.train_loader)}], Train Loss: {loss.item():.4f}, Batch Accuracy: {batch_acc*100:.2f}%")
                    wandb.log({"batch_loss": loss.item(), "batch_accuracy": batch_acc})

            # each epoch adjust lr
            #self.scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            epoch_accuracy = total_correct / total_samples
            print(f"\n------------ Epoch [{epoch+1}/{self.epochs}] ------------") 
            print(f"Average Train Loss: {avg_loss:.4f}")
            print(f"Average Train Accuracy: {epoch_accuracy*100:.2f}%")
            print(f"learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

            



            # Epoch-level logging
            if epoch % self.eval_interval == 0:
                # TODO: Evaluate on validation set
                val_loss,val_acc,metrics = self.evaluator.eval()
                
                wandb.log({
                "epoch": epoch + 1,
                "lr": self.scheduler.get_last_lr()[0],
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_AUC": metrics["auc"],
                "val_F1": metrics["f1"],
                "val_Precision": metrics["precision"],
                "val_Recall": metrics["recall"]
                })

                val_auc = metrics["auc"]
                # Save checkpoint if accuracy improves
                if val_auc > self.best_AUC:
                    self.best_AUC = val_auc
                    self.best_acc = val_acc
                    self.best_val_loss = val_loss
                    self.best_F1 = metrics["f1"]
                    self.best_Precision = metrics["precision"]
                    self.best_Recall = metrics["recall"]

                    self.counter = 0
                    self.save_checkpoint_fold(epoch, self.fold)  # for k-fold, fold=0 as placeholder

                
                    
                    #fold_best_accs.append(self.best_acc)
                    #fold_best_AUCs.append(metrics["AUC"])
                    #fold_best_losses.append(self.best_val_loss)
                    #fold_best_F1s.append(metrics["F1"])
                    #fold_best_Precisions.append(metrics["Precision"])
                    #fold_best_Recalls.append(metrics["Recall"])

                else:
                    self.counter += 1
                    print(f"Validation AUC did not improve. Counter: {self.counter}/{self.patience}")
                    if self.counter >= self.patience:
                        print("Early stopping triggered.")
                        return


        wandb.log({
            "best_val_AUC": self.best_AUC,
            "best_val_acc": self.best_acc,
            "best_val_loss": self.best_val_loss,
            "best_val_F1": self.best_F1,
            "best_val_Precision": self.best_Precision,
            "best_val_Recall": self.best_Recall
        })
                    
        self.fold_best_accs.append(self.best_acc)
        self.fold_best_AUCs.append(self.best_AUC)
        self.fold_best_losses.append(self.best_val_loss)
        self.fold_best_F1s.append(self.best_F1)
        self.fold_best_Precisions.append(self.best_Precision)
        self.fold_best_Recalls.append(self.best_Recall)    
        
        





class Trainer_5c:
    #*** train head only - feature level - SMOTE **********
    #def __init__(self, cfg, model_head,train_loader, val_loader, model, evaluator, device, save_path,train_head_only =False):
    def __init__(self, cfg,train_loader, val_loader, model, evaluator, device, save_path,train_head_only =False):
        self.cfg = cfg
        #self.model_head = model_head
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.evaluator = evaluator
        self.device = device
        self.save_path = cfg.save.path
        self.train_head_only = train_head_only
        
        self.distance_loss = DistancePenalty(weight=0.3, p=1)
        

        self.log_interval = cfg.log_interval
        self.eval_interval = cfg.eval_interval
        self.epochs = cfg.epochs
        self.patience = cfg.patience


        self.counter = 0        
        self.best_acc = 0.0
        self.best_AUC = 0.0
        self.best_F1 = 0.0
        self.best_Precision = 0.0
        self.best_Recall = 0.0
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_qwk = 0.0
        self.best_bal_acc = 0.0
        self.best_val_f1 = 0.0

        # === NEW: flags for feature-level fusion regularization ===
        # 是否使用“特征融合 + 多样性正则”；从 cfg.train 读取，默认 False/0.0 不生效
        self.use_feature_fusion = getattr(cfg.train, "use_feature_fusion", False)
        self.lambda_decor = getattr(cfg.train, "lambda_decor", 0.0)


        
        # TODO:
        #params = list(self.model.parameters())
        #self.optimizer = build_optimizer(cfg, params)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr) # L2 regularization,weight_decay=1e-4 not good(?
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs) # adjust lr #, eta_min=1e-6
        
        # optimizer
        #self.optimizer = instantiate(cfg.optim, params=self.model.parameters())

        # scheduler
        #self.scheduler = None
        #if cfg.sched is not None:
        #    self.scheduler = instantiate(cfg.sched, optimizer=self.optimizer)

        #pos_weight = torch.tensor([2.0], device=self.device)
        #self.criterion = nn.BCEWithLogitsLoss()  # since output is sigmoid(0–1), pos_weight

        #alpha = torch.tensor([1.0, 2.0, 1.0, 1.0, 3.0], device=self.device)  # according to class distribution
        #self.criterion = FocalCELoss(alpha=alpha, gamma=2.0)  # CrossEntropyLoss + Focal Loss
        self.criterion =  instantiate(cfg.train_module.loss)
        self.scaler = amp.GradScaler(device=self.device) # AMP

        self.aug_plugin = None #cfg.aug_plugin
        #cfg.aug_plugin = MixupCutMix(mixup_alpha=0.4,cutmix_alpha=1.0,prob=1.0)

        self.early_stopping = EarlyStopping(
            patience=5,
            min_delta=1e-4,
            verbose=True,
        )

        
        # Initialize WandB
        #wandb.init(entity="zizi1217-uni-stuttgart",project="diabetic_retinopathy_5class", config={
        #    "epochs": self.epochs,
        #    "optimizer": type(self.optimizer).__name__,
        #    "lr": self.optimizer.param_groups[0]['lr'],
        #})


        #wandb.watch(self.model, self.criterion, log="all", log_freq=100) #（? log_freq=100


    

     # ------------------ Checkpoint ------------------
    def save_checkpoint(self, epoch):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "epochs": self.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
            "best_qwk": self.best_qwk,
            "best_bal_acc": self.best_bal_acc,
            "best_macro_f1": self.best_val_f1,
            "patience_counter": self.counter
        }, self.save_path)
        wandb.save(self.save_path)
        print(f"Checkpoint saved at {self.save_path}")

    def load_checkpoint(self, path=None):
        path = path or self.check_path
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.epochs = checkpoint.get("epochs", self.epochs)
            self.best_acc = checkpoint.get("best_acc", 0.0)
            self.best_qwk = checkpoint.get("best_qwk", 0.0)
            self.best_bal_acc = checkpoint.get("best_bal_acc", 0.0)
            self.best_macro_f1 = checkpoint.get("best_macro_f1", 0.0)
            self.counter = checkpoint.get("patience_counter", 0)
            print(f"Loaded checkpoint from {path}, starting at epoch {self.start_epoch}")
        else:
            print(f"No checkpoint found at {path}, starting from scratch.")
        return self.start_epoch

    def save_checkpoint_fold(self, epoch, fold):
        fold_save_dir = f"./checkpoints/convnext/fold{fold+1}"
        os.makedirs(fold_save_dir, exist_ok=True)

        model_version = os.path.basename(self.save_path).replace('.pth','')

        save_path = os.path.join(fold_save_dir, f"{model_version}_fold{fold+1}.pth") #best_model_fold{fold+1}.pth"

        torch.save({
            "epoch": epoch,
            "epochs": self.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
            "fold": fold,
            "patience_counter": self.counter
        }, save_path)

        wandb.save(save_path)
        print(f"Checkpoint saved at {save_path}")


    def load_checkpoint_fold(self, path=None):
        path = path or self.check_path
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.epochs = checkpoint.get("epochs", self.epochs)
            self.best_acc = checkpoint.get("best_acc", 0.0)
            self.fold = checkpoint.get("fold", 0)
            self.counter = checkpoint.get("patience_counter", 0)
            print(f"Loaded checkpoint from {path}, starting at epoch {self.start_epoch}")
        else:
            print(f"No checkpoint found at {path}, starting from scratch.")
        return self.start_epoch, self.fold


    def train(self, resume=False, checkpoint_path=None):
        # TODO: Initialization
        start_epoch = 0     

        # Data Augmentation Plugin

        if resume and checkpoint_path is not None:
            start_epoch = self.load_checkpoint(checkpoint_path)

        total_steps = len(self.train_loader) * self.epochs
        warmup_steps = int(total_steps * 0.1)  # 10%
        def warmup_lr(step, warmup_steps=warmup_steps, lr_max=1e-4):
            if step < warmup_steps:
                return lr_max * (step + 1) / warmup_steps
            return None

        for epoch in range(start_epoch, self.epochs):
            self.model.train() #(!)
            torch.cuda.empty_cache()  # clear cache
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            print(f"=============================================")
            print(f"Epoch [{epoch+1}/{self.epochs}]")

            for step, (x, y_true) in enumerate(self.train_loader):


                #if self.train_head_only:
                #    x = x.to(self.device).float()
                #    y_true = y_true.to(self.device).long()
                #else:
                #***
                x, y_true = x.to(self.device), y_true.to(self.device).float().unsqueeze(1)
                #x = x.to(self.device)
                #y_true = y_true.to(self.device).long()
                #print("labels:", y_true[:20])
                #print("unique:", torch.unique(y_true))
                

                #!y_pred = self.model(x) # Forward pass

                # TODO:
                #!loss = self.criterion(y_pred, y_true)

                # TODO: Optimizer step
                # Backward pass
                self.optimizer.zero_grad()
                
                # using autocast for mixed precision
                with amp.autocast('cuda'):
                    if self.train_head_only:
                        # 头部单独训练的情况，保持原逻辑
                        y_pred = self.model_head(x)  # [B, 5]
                        feats = None  # 占位
                    else:
                        # === NEW: 区分普通模型 vs 融合模型 ===
                        if self.use_feature_fusion:
                            # 融合模型需要支持 forward(x, return_feats=True)
                            # 返回：logits, [f1, f2, f3]（建议是 projection 之后的特征）
                            y_pred, feats = self.model(x, return_feats=True)
                        else:
                            # 普通模型：和原来一样
                            y_pred = self.model(x)
                            feats = None
                        # === NEW END ===

                    # label 处理保持不变
                    y_true = y_true.squeeze(1).long()

                    # 基础分类 loss（分类 + 你的 distance penalty）
                    base_loss = self.criterion(y_pred, y_true) + self.distance_loss(y_pred, y_true)

                    # === NEW: 如果是特征融合并且设置了 lambda_decor，就加多样性正则 ===
                    if self.use_feature_fusion and (feats is not None) and (self.lambda_decor > 0.0):
                        decor_loss = feature_correlation_loss(feats)
                        loss = base_loss + self.lambda_decor * decor_loss
                    else:
                        loss = base_loss 

                #print("DEBUG requires_grad:", y_pred.requires_grad)
                #print("DEBUG loss requires_grad:", loss.requires_grad)
                #print("DEBUG loss.grad_fn:", loss.grad_fn)
                # using scaler to scale the loss and call backward()
                self.scaler.scale(loss).backward()
    
                # using scaler to step the optimizer
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Warmup
                lr_new = warmup_lr(step + epoch * len(self.train_loader), warmup_steps=warmup_steps, lr_max=1e-4)
                if lr_new is not None:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_new
                else:
                    self.scheduler.step()  # CosineAnnealing step (batch-level)


                #!loss.backward()
                #!self.optimizer.step()

                total_loss += loss.item()

                # Compute accuracy
                #predicted = (y_pred > 0.5).float()
                #total_correct += (predicted == y_true).sum().item()
                #total_samples += y_true.size(0)

                
                # Batch-level logging
                if step % self.log_interval == 0:
                    # TODO: Logging
                    #batch_acc = (predicted == y_true).sum().item() / y_true.size(0)
                    
                    print(f"Epoch [{epoch+1}/{self.epochs}], Step [{step}/{len(self.train_loader)}], Train Loss: {loss.item():.4f}")
                    wandb.log({"batch_loss": loss.item()})

            # each epoch adjust lr
            #self.scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            print(f"\n------------ Epoch [{epoch+1}/{self.epochs}] ------------") 
            print(f"Average Train Loss: {avg_loss:.4f}")
            print(f"learning rate: {self.scheduler.get_last_lr()[0]:.2e}")

            



            # Epoch-level logging
            if epoch % self.eval_interval == 0:
                # TODO: Evaluate on validation set
                metrics = self.evaluator.eval()
                val_acc = metrics["acc"]
                val_qwk = metrics["qwk"]
                val_bal_acc = metrics["balanced_acc"]
                val_macro_f1 = metrics["macro_f1"]
                #val_loss = metrics["loss"]
                #val_cm = metrics['confusion_matrix']

                wandb.log({
                "train/epoch": epoch + 1,
                "train/lr": self.scheduler.get_last_lr()[0],
            #    "train_loss": avg_loss,
            #    "val_loss": val_loss,
                "val/val_acc": val_acc,
                "val/val_qwk": val_qwk,
                "val/val_balanced_acc": val_bal_acc,
                "val/val_macro_f1": val_macro_f1
                })
            #    "val_AUC": metrics["AUC"],
            #    "val_F1": metrics["F1"],
            #    "val_Precision": metrics["Precision"],
            #    "val_Recall": metrics["Recall"]
            #    })

                # Save checkpoint if accuracy improves
                # if val_acc >= self.best_acc:
                #     self.best_qwk = val_qwk
                #     self.best_acc = val_acc  
                #     self.best_bal_acc = val_bal_acc
                #     self.best_macro_f1 = val_macro_f1                  
                #     self.counter = 0
                #     self.save_checkpoint(epoch) # normal save
                #     #print(f"Validation QWK improved to {val_qwk:.4f}. Checkpoint saved.")
                #     print(f"Best Acc: {self.best_acc:.4f}, Best QWK: {self.best_qwk:.4f}, Best Bal Acc: {self.best_bal_acc:.4f}, Best Macro F1: {self.best_macro_f1:.4f}")
                #     # save best model for wandb

                    

                #     #wandb.log({
                #     #    "best_val_acc": self.best_acc,
                #     #    "best_val_loss": self.best_val_loss,
                #     #    "best_val_AUC": metrics["AUC"],
                #     #    "best_val_F1": metrics["F1"],
                #     #    "best_val_Precision": metrics["Precision"],
                #     #    "best_val_Recall": metrics["Recall"]
                #     #    })
                    


                # else:
                #     self.counter += 1
                #     print(f"Validation acc did not improve. Counter: {self.counter}/{self.patience}")
                #     if self.counter >= self.patience:
                #         print("Early stopping triggered.")
                #         return


                if self.cfg.train.metric == "f1":
                    score = metrics["macro_f1"]
                
                    if score >= self.best_val_f1:
                        self.best_val_f1 = score
                        self.best_val_acc = metrics["acc"]
                        self.best_val_bal_acc = metrics["balanced_acc"]
                        self.best_val_qwk = metrics["qwk"]
                        #val_acc = metrics["acc"]
                        #self.val_qwk = metrics["qwk"]
                        #val_bal_acc = metrics["balanced_acc"]
                        #val_macro_f1 = metrics["macro_f1"]
                        self.save_checkpoint(epoch)
                        print(f"✅ New best F1: {self.best_val_f1:.4f}")
                        print(f"Best Macro F1: {self.best_val_f1:.4f}, Best Bal Acc: {self.best_val_bal_acc:.4f}, Best Acc: {self.best_val_acc:.4f}, Best QWK: {self.best_val_qwk:.4f}")

                        wandb.log({
                            "best_Model/epoch": epoch,
                            "best_Model/val_acc": self.best_val_acc,
                            #"best_Model/val_loss": metrics["loss"],
                            "best_Model/best_val_F1": self.best_val_f1,
                            #"best_Model/val_Precision": metrics["precision"],
                            #"best_Model/val_Recall": metrics["recall"],
                            #"best_Model/ConfusionMatrix": metrics["confusion_matrix"],
                        })

                        #if self.cfg.train.use_bayes_prior and epoch == 1:
                        #    class_names = [str(i) for i in range(self.cfg.num_classes)]
                        #    plot_transition_prior(prior_module=self.transition_prior,class_names=class_names)

                else:
                    score = metrics["balanced_acc"]
                    if score >= self.best_val_bal_acc:
                        self.best_val_bal_acc = score
                        self.best_val_f1 = metrics["macro_f1"]
                        self.best_val_acc = metrics["acc"]
                        self.best_val_qwk = metrics["qwk"]
                        #val_acc = metrics["acc"]
                        #self.val_qwk = metrics["qwk"]
                        #val_bal_acc = metrics["balanced_acc"]
                        #val_macro_f1 = metrics["macro_f1"]
                        self.save_checkpoint(epoch)
                        print(f"✅ New best balanced accuracy: {self.best_val_bal_acc:.4f}")
                        print(f" Best Bal Acc: {self.best_val_bal_acc:.4f}, Best Macro F1: {self.best_val_f1:.4f}, Best Acc: {self.best_val_acc:.4f}, Best QWK: {self.best_val_qwk:.4f}")

                        wandb.log({
                            "best_Model/epoch": epoch,
                            "best_Model/val_acc": self.best_val_acc,
                            "best_Model/best_val_bal_acc": self.best_val_bal_acc,
                            #"best_Model/val_loss": metrics["loss"],
                            "best_Model/val_F1": self.best_val_f1,
                            #"best_Model/val_Precision": metrics["precision"],
                            #"best_Model/val_Recall": metrics["recall"],
                            #"best_Model/ConfusionMatrix": metrics["confusion_matrix"],
                        })

        #clean_name = self.cfg.check_path.replace("./checkpoints/", "")
        #wandb.init(project="diabetic_retinopathy_5class_eval")

        #wandb.log({
        #    "val/best_val_acc": self.best_acc,
        #    "val/best_val_qwk": self.best_qwk,
        #    "val/best_val_bal_acc": self.best_bal_acc,
        #    "val/best_val_macro_f1": self.best_macro_f1
        #})

                self.early_stopping.step(score, self.model)

                if self.early_stopping.should_stop:
                    print(f"Early stopped at epoch {epoch}")
                    self.early_stopping.restore_best(self.model, self.device)
                    break
                    
  
        
        wandb.finish()

        return self.best_val_acc, self.best_val_qwk, self.best_val_bal_acc, self.best_val_f1
          
 
 


class Trainer:
    def __init__(self, cfg, train_loader, val_loader, model, evaluator, device, max_checkpoints=5):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.evaluator = evaluator
        self.device = device

        # Training parameters
        self.epochs = cfg.epochs
        self.log_interval = cfg.log_interval

        if cfg.task == '5c':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.optimizer = optim.Adam(
                                self.model.parameters(),
                                lr=cfg.lr,
                                weight_decay=cfg.weight_decay
                            )
        self.best_val_loss = float('inf')
        self.best_val_f1 = -1.0
        self.patience = cfg.patience
        self.counter = 0

        # Checkpoints
        self.max_checkpoints = max_checkpoints
        self.checkpoint_queue = deque()
        self.checkpoint_folder = "artifacts/checkpoints"
        os.makedirs(self.checkpoint_folder, exist_ok=True)

    # ------------------------------
    # CHECKPOINT SAVE
    # ------------------------------
    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_folder, f"checkpoint_epoch{epoch}.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "besrt_val_f1": self.best_val_f1,
            "patience_counter": self.counter
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

        # Keep last N checkpoints only
        self.checkpoint_queue.append(path)
        if len(self.checkpoint_queue) > self.max_checkpoints:
            old_path = self.checkpoint_queue.popleft()
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"Deleted old checkpoint: {old_path}")

    # ------------------------------
    # CHECKPOINT LOAD
    # ------------------------------
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_f1 = checkpoint["besrt_val_f1"]
        self.counter = checkpoint["patience_counter"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch

    # ------------------------------
    # TRAIN FUNCTION
    # ------------------------------
    def train(self, resume=False, checkpoint_path=None):
        start_epoch = 1
        if resume and checkpoint_path is not None:
            start_epoch = self.load_checkpoint(checkpoint_path)

        for epoch in range(start_epoch, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            train_preds = []
            train_labels = []

            for step, (x, y_true) in enumerate(self.train_loader):
                x, y_true = x.to(self.device), y_true.to(self.device).float().unsqueeze(1)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_true)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Collect training predictions
                probs = torch.sigmoid(y_pred)
                preds = (probs > 0.5).long()

                train_preds.extend(preds.cpu().numpy().flatten())
                train_labels.extend(y_true.cpu().numpy().flatten())

                total_loss += loss.item()

                if step % self.log_interval == 0:
                    print(f"Epoch [{epoch}/{self.epochs}], Step [{step}/{len(self.train_loader)}], Loss: {loss.item():.4f}")

            avg_train_loss = total_loss / len(self.train_loader)
            train_preds_np = np.array(train_preds)
            train_labels_np = np.array(train_labels)

            train_acc = np.mean(train_preds_np == train_labels_np)
            train_f1 = f1_score(train_labels_np, train_preds_np)

            print(f"Epoch [{epoch}/{self.epochs}] "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_acc*100:.2f}%, "
                f"Train F1: {train_f1:.4f}")
            
            # Log training metrics to wandb
            wandb.log({
                "train/loss": avg_train_loss,
                "train/accuracy": train_acc,
                "train/f1": train_f1
            }, step=epoch)


            val_metrics = self.evaluator.eval_binary()

            #wanb logging 
            wandb.log({
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/f1": val_metrics["f1"]
                }, step=epoch)

            val_f1 = val_metrics["f1"]
            val_loss = val_metrics["loss"]

            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1= val_f1
                self.counter = 0
                if self.cfg.dataset.name == 'IDRID':
                    best_model_path = f'artifacts/idrid_{self.cfg.model.name}_best_model.pth'
                else:
                    best_model_path = f'artifacts/eyepacs_{self.cfg.model.name}_best_model.pth'
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved: {best_model_path}")
            # else:
            #     self.counter += 1
            #     print(f"Validation F1 did not improve. Counter: {self.counter}/{self.patience}")
            #     if self.counter >= self.patience:
            #         print("Early stopping triggered.")
            #         return

            # Save checkpoint every epoch
            self.save_checkpoint(epoch)
    
    # ------------------------------
    # TRAIN LOOP
    # ------------------------------
    def train_multiclass(self, resume=False, checkpoint_path=None):
        start_epoch = 1
        if resume and checkpoint_path is not None:
            start_epoch = self.load_checkpoint(checkpoint_path)
        for epoch in range(start_epoch, self.epochs + 1):
            self.model.train()

            total_loss = 0.0
            train_preds = []
            train_labels = []

            for step, (x, y_true) in enumerate(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device).long()   # ✅ class indices


                logits = self.model(x)                  # [B, C]
                loss = self.criterion(logits, y_true)

                if (y_true < 0).any() or (y_true >= logits.size(1)).any():
                    print(f"Invalid labels in batch: min={y_true.min().item()}, max={y_true.max().item()}")
                    raise ValueError("Labels are out of range for CrossEntropyLoss.")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds = torch.argmax(logits, dim=1)     # ✅ multiclass prediction

                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(y_true.cpu().numpy())

                total_loss += loss.item()

                if step % self.log_interval == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}] "
                        f"Step [{step}/{len(self.train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )

            # --------------------
            # Training metrics
            # --------------------
            train_preds = np.array(train_preds)
            train_labels = np.array(train_labels)

            train_acc = np.mean(train_preds == train_labels)
            train_f1 = f1_score(train_labels, train_preds, average="macro")

            print(
                f"Epoch [{epoch}/{self.epochs}] "
                f"Train Loss: {total_loss / len(self.train_loader):.4f}, "
                f"Train Acc: {train_acc*100:.2f}%, "
                f"Train F1: {train_f1:.4f}"
            )

            wandb.log({
                "train/loss": total_loss / len(self.train_loader),
                "train/accuracy": train_acc,
                "train/f1": train_f1
            }, step=epoch)

            # --------------------
            # Validation
            # --------------------
            val_metrics = self.evaluator.eval_multiclass()  # must also be multiclass

            wandb.log({
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/f1": val_metrics["f1"]
            }, step=epoch)

            # --------------------
            # Save best model
            # --------------------
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]

                best_path = f"artifacts/{self.cfg.dataset.name.lower()}_{self.cfg.model.name}_multiclass_best_model.pth"
                torch.save(self.model.state_dict(), best_path)
                print(f"New best model saved: {best_path}")

            self.save_checkpoint(epoch)



class EarlyStopping:
    """
    Early stopping based on a monitored metric (the higher the better).
    """
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_state = None

    def step(self, score: float, model: nn.Module):
        """
        score: monitored metric (e.g. val macro F1)
        """
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return

        if score >= self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"[EarlyStopping] New best score: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("[EarlyStopping] Patience exceeded. Stop training.")

    def restore_best(self, model: nn.Module, device):
        if self.best_state is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in self.best_state.items()}
            )