import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
import torch.amp as amp
from models import threshold_sweep


import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix,balanced_accuracy_score,f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import wandb



from hydra.utils import instantiate


class Evaluator:
    def __init__(self, cfg, eval_loader, model, device, return_samples=False, num_samples=None):
        self.cfg = cfg
        self.eval_loader = eval_loader
        self.model = model
        self.device = device
        
        self.return_samples = return_samples
        self.num_samples = num_samples

        if cfg.task == '5c':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.num_classes = cfg.model.num_classes

    @torch.no_grad()
    def eval_binary(self):
        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_probs = []
        all_targets = []

        for x, y_true in self.eval_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device).float().unsqueeze(1)

            logits = self.model(x)
            loss = self.criterion(logits, y_true)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_targets.extend(y_true.cpu().numpy().flatten().tolist())

        # ---- Metrics ----
        avg_loss = total_loss / len(self.eval_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)

        try:
            roc_auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            roc_auc = float('nan')  # In case only one class is present

        # ---- Print ----
        print("\nEvaluation Results:")
        print(f"  • Average Loss: {avg_loss:.4f}")
        print(f"  • Accuracy: {accuracy * 100:.2f}%")
        print(f"  • Precision: {precision:.4f}")
        print(f"  • Recall: {recall:.4f}")
        print(f"  • F1-Score: {f1:.4f}")
        print(f"  • ROC-AUC: {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }

    @torch.no_grad()
    def eval_multiclass(self):
        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_targets = []
        all_probs = []

        for x, y_true in self.eval_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device).long()   # ✅ class indices

            logits = self.model(x)                   # [B, C]
            loss = self.criterion(logits, y_true)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)     # ✅ multiclass probs
            preds = torch.argmax(probs, dim=1)       # ✅ class prediction

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())

        # --------------------
        # Stack arrays
        # --------------------
        all_probs = np.vstack(all_probs)       # [N, C]
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        avg_loss = total_loss / len(self.eval_loader)

        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro")
        precision = precision_score(all_targets, all_preds, average="macro")
        recall = recall_score(all_targets, all_preds, average="macro")
        cm = confusion_matrix(all_targets, all_preds)

        # --------------------
        # ROC-AUC (One-vs-Rest)
        # --------------------
        try:
            roc_auc = roc_auc_score(
                all_targets,
                all_probs,
                multi_class="ovr",
                average="macro"
            )
        except ValueError:
            roc_auc = float("nan")

        # --------------------
        # Print
        # --------------------
        print("\nEvaluation Results:")
        print(f"  • Average Loss: {avg_loss:.4f}")
        print(f"  • Accuracy: {accuracy * 100:.2f}%")
        print(f"  • Precision (macro): {precision:.4f}")
        print(f"  • Recall (macro): {recall:.4f}")
        print(f"  • F1-Score (macro): {f1:.4f}")
        print(f"  • ROC-AUC (OvR): {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_probs = []
        all_targets = []

        for x, y_true in self.eval_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device).float().unsqueeze(1)

            logits = self.model(x)
            loss = self.criterion(logits, y_true)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_targets.extend(y_true.cpu().numpy().flatten().tolist())

        # ---- Metrics ----
        avg_loss = total_loss / len(self.eval_loader)
        accuracy = correct / total if total > 0 else 0

        # concatenate all labels and scores
        y_true_np = torch.cat(all_labels).numpy()
        y_score_np = torch.cat(all_scores).numpy()

        # four metrics
        auc = roc_auc_score(y_true_np, y_score_np)
        f1 = f1_score(y_true_np, y_score_np >= t, zero_division=0)
        precision = precision_score(y_true_np, y_score_np >= t, zero_division=0)
        recall = recall_score(y_true_np, y_score_np >= t, zero_division=0)

        #precision, recall, thresholds = precision_recall_curve(y_true_np, y_pred_probs_np) #(?)
        #f1_scores = 2 * (precision * recall) / (precision + recall)
        #best_idx = f1_scores.argmax()
        #best_threshold = thresholds[best_idx]

        #print("Best threshold:", best_threshold ,"| Precision:", precision[best_idx], "| Recall:", recall[best_idx], "| F1:", f1_scores[best_idx])
        threshold_sweep(y_true_np, y_score_np)

        metrics = {
            "AUC": auc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall
        }

        # Log metrics
        print(f"\nEvaluation Results [t_eval_class {t}]")
        print(f"  • Average Loss: {avg_loss:.4f}")
        print(f"  • Accuracy: {accuracy * 100:.2f}%")
        print(f"  • AUC: {auc:.4f}")
        print(f"  • F1: {f1:.4f}")
        print(f"  • Precision: {precision:.4f}")
        print(f"  • Recall: {recall:.4f}\n")
        print(f"           Confusion Matrix            ")
        print(f"--------------------------------------")
        print(f"|        |    pred = 0 |    pred = 1 |")
        print(f"--------------------------------------")
        print(f"| true=0 |    {((y_true_np == 0) & (y_score_np < t)).sum():>6}   |    {((y_true_np == 0) & (y_score_np >= t)).sum():>6}   |")
        print(f"--------------------------------------")
        print(f"| true=1 |    {((y_true_np == 1) & (y_score_np < t)).sum():>6}   |    {((y_true_np == 1) & (y_score_np >= t)).sum():>6}   |")
        print(f"--------------------------------------\n")
        
        

        if return_samples:
            return avg_loss, accuracy, metrics, sample_list
        else:
            return avg_loss, accuracy, metrics



class Evaluator_2c:
    def __init__(self, cfg, eval_loader, model, device, return_samples=False, num_samples=None):
        self.cfg = cfg
        self.eval_loader = eval_loader
        self.model = model
        self.device = device
        #self.criterion = nn.BCEWithLogitsLoss()#FocalLoss(alpha=0.25, gamma=2.0) ##nn.BCELoss()
        #self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.criterion = instantiate(cfg.train_module.loss)
        self.metric_fn = instantiate(cfg.train_module.metrics)
        self.return_samples = return_samples 
        self.num_samples = num_samples

    @torch.no_grad()
    def eval(self, return_samples=False, num_samples=None):
        # TODO: Initialization
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # for AUC, F1-score
        all_labels = []
        all_scores = []

        # sample for visualization
        sample_list = []

        #t = 0.4  # Threshold for classification
        for step, (x, y_true) in enumerate(self.eval_loader):
            x, y_true = x.to(self.device), y_true.to(self.device).float().unsqueeze(1)
            #!y_pred = self.model(x)

            # TODO: Compute metrics
            # Compute loss
            #!loss = self.criterion(y_pred, y_true)
            
            with torch.no_grad(), amp.autocast('cuda'):
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_true)
            
            total_loss += loss.item()

            # Compute accuracy
            #predicted = (y_pred > t).float()
            #correct += (predicted == y_true).sum().item()
            #total += y_true.size(0)

            # For AUC, F1-score
            prob = torch.sigmoid(y_pred)
            all_scores.append(prob.cpu())
            all_labels.append(y_true.cpu())

            # For visualization samples
            if return_samples and len(sample_list) < num_samples:
                for i in range(x.size(0)):
                    if len(sample_list) >= num_samples:
                        break
                    sample_list.append( (x[i].cpu(), y_true[i].cpu().item(), prob[i].cpu().item()) )

        # TODO: Log metrics
        # Compute averages
        avg_loss = total_loss / len(self.eval_loader)
        #accuracy = correct / total if total > 0 else 0

        # concatenate all labels and scores
        y_true_np = torch.cat(all_labels).numpy()
        y_score_np = torch.cat(all_scores).numpy()
        threshold_sweep(y_true_np, y_score_np)

        metrics = self.metric_fn(y_true_np, y_score_np)

        accuracy = metrics["accuracy"]
        cm = metrics["confusion_matrix"]


        # Log metrics
        print(f"\nEvaluation Results ")  #[t_eval_class {t}]
        print(f"  • Average Loss: {avg_loss:.4f}")
        print(f"  • Accuracy: {accuracy * 100:.2f}%")
        print(f"  • AUC: {metrics['auc']:.4f}")
        print(f"  • F1: {metrics['f1']:.4f}")
        print(f"  • Precision: {metrics['precision']:.4f}")
        print(f"  • Recall: {metrics['recall']:.4f}\n")
        print(f"           Confusion Matrix            ")
        print(f"--------------------------------------")
        print(f"|        |    pred = 0 |    pred = 1 |")
        print(f"--------------------------------------")
        print(f"| true=0 |    {cm[0, 0]:>6}   |    {cm[0, 1]:>6}   |")
        print(f"--------------------------------------")
        print(f"| true=1 |    {cm[1, 0]:>6}   |    {cm[1, 1]:>6}   |")
        print(f"--------------------------------------\n")
        
        

        if return_samples:
            return avg_loss, accuracy, metrics, sample_list
        else:
            return avg_loss, accuracy, metrics




class Evaluator_5c:
    #train head only
    #def __init__(self, cfg, model_head=None,eval_loader, model, device,train_head_only=False):
    def __init__(self, cfg,eval_loader, model, device,train_head_only=False,return_samples=False, num_samples=None, weights=[1.0, 1.0, 1.0]):#):
        self.cfg = cfg
        #self.model_head = model_head
        self.eval_loader = eval_loader
        self.model = model
        self.device = device
        self.train_head_only = train_head_only

        self.return_samples = return_samples
        self.num_samples = num_samples
        self.weights = weights

        #alpha = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=self.device)  # 根据类别不平衡设置权重 (我把数据都over-sample了，这里可以不设置)
        #self.criterion = FocalCELoss(alpha=alpha, gamma=2.0)  # CrossEntropyLoss + Focal Loss

        self.criterion = instantiate(cfg.train_module.loss)
        self.metric_fn = instantiate(cfg.train_module.metrics)

    @torch.no_grad()
    def eval(self, return_samples=False, num_samples=None):
        if self.train_head_only:
            model = self.model_head
        else:
            model = self.model

        model.eval()

        all_logits = []
        all_targets = []

        # sample for visualization
        sample_list = []
        for x, y in self.eval_loader:
            x = x.to(self.device)
            y = y.to(self.device).long()

            if y.dim() == 2:
                y = y.squeeze(1)

            with torch.no_grad():
                logits = self.model(x)

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

             # sample collection
            if return_samples and len(sample_list) < num_samples:
                for i in range(x.size(0)):
                    if len(sample_list) >= num_samples:
                        break

                # append as tuple (image, label)
                sample_list.append((x[i].cpu(), y[i].item()))

        all_logits = torch.cat(all_logits, dim=0)   # [N,5]
        all_targets = torch.cat(all_targets, dim=0) # [N]

        #print(all_logits.shape, all_targets.shape)

        metrics = self.metric_fn(all_logits, all_targets)
        #metrics = evaluate_fiveclass(all_logits, all_targets)

        # --- Print results ---
        print(f"\nEvaluation Results [5-class]")
        print(f"  • Accuracy: {metrics['acc']:.4f}")
        print(f"  • Balanced Acc: {metrics['balanced_acc']:.4f}")
        print(f"  • QWK: {metrics['qwk']:.4f}") 
        print(f"  • Macro F1: {metrics['macro_f1']:.4f}")
        #print("Unique preds:", np.unique(y_pred))
        #print("Unique targets:", np.unique(y_true))
        print("Confusion Matrix:\n", metrics['confusion_matrix'])

        if return_samples:
            return metrics, sample_list
        else:
            return metrics




class Evaluator_output_ensemble:
    def __init__(self, cfg, loader, models, device, strategy="prob_avg",
                 return_samples=False, num_samples=None, weights=[1.0, 1.0, 1.0]):

        self.cfg = cfg
        self.loader = loader
        self.models = models if isinstance(models, list) else [models]
        self.device = device
        self.strategy = strategy
        self.return_samples = return_samples
        self.num_samples = num_samples
        self.weights = weights  # for weighted ensemble
        self.criterion = instantiate(cfg.train_module.loss)
        self.metric_fn = instantiate(cfg.train_module.metrics)

        wandb.init(entity="zizi1217-uni-stuttgart",project="diabetic_retinopathy_ensemble")

    @torch.no_grad()
    def _ensemble_predict(self):
        all_logits = []
        all_probs = []
        all_preds = []

        detected_C = None  # number of output classes/channels from model forward

        for model in self.models:
            model.eval()
            logits_list = []
            probs_list = []
            preds_list = []

            for x, _ in self.loader:
                x = x.to(self.device)
                logits = model(x)                     # [B,1]

                # normalize to (B,C)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)

                B, C = logits.shape
                if detected_C is None:
                    detected_C = C
                else:
                    # sanity check: all models should output same C
                    if C != detected_C:
                        raise ValueError(f"Ensemble models output mismatch: got C={C}, expected C={detected_C}")

                # decode to probabilities + predictions
                if C == 1:
                    # binary: single logit
                    probs = torch.sigmoid(logits).squeeze(1)          # (B,)
                    preds = (probs > 0.5).long()                      # (B,)
                    logits_list.append(logits.squeeze(1).detach().cpu())  # (B,)
                    probs_list.append(probs.detach().cpu())               # (B,)
                    preds_list.append(preds.detach().cpu())               # (B,)
                else:
                    # multiclass (including C=2 and C=5)
                    probs = torch.softmax(logits, dim=1)               # (B,C)
                    preds = probs.argmax(dim=1).long()                 # (B,)
                    logits_list.append(logits.detach().cpu())          # (B,C)
                    probs_list.append(probs.detach().cpu())            # (B,C)
                    preds_list.append(preds.detach().cpu())            # (B,)



            all_logits.append(torch.cat(logits_list, dim=0))
            all_probs.append(torch.cat(probs_list, dim=0))
            all_preds.append(torch.cat(preds_list, dim=0))

        # ----- ensemble aggregation -----
        # binary case: each probs tensor is (N,)
        # multiclass: each probs tensor is (N,C)
        if self.strategy == "prob_avg":
            prob_stack = torch.stack(all_probs, dim=0)                 # binary: (M,N)  multi: (M,N,C)
            ensemble_probs = prob_stack.mean(dim=0)                    # binary: (N)    multi: (N,C)

        elif self.strategy == "logit_avg":
            logit_stack = torch.stack(all_logits, dim=0)               # binary: (M,N)  multi: (M,N,C)
            ensemble_logits = logit_stack.mean(dim=0)                  # binary: (N)    multi: (N,C)
            if detected_C == 1:
                ensemble_probs = torch.sigmoid(ensemble_logits)        # (N,)
            else:
                ensemble_probs = torch.softmax(ensemble_logits, dim=1) # (N,C)

        elif self.strategy == "hard_vote":
            pred_stack = torch.stack(all_preds, dim=0)                 # (M,N)
            ensemble_preds = torch.mode(pred_stack, dim=0).values      # (N,)

            # also return probs as average probs for convenience
            prob_stack = torch.stack(all_probs, dim=0)
            ensemble_probs = prob_stack.mean(dim=0)

            return ensemble_preds.long(), ensemble_probs

        elif self.strategy == "weighted_prob":
            w = torch.tensor(self.weights, dtype=torch.float32).view(-1, *([1] * all_probs[0].ndim))
            # w shape: (M,1) for binary; (M,1,1) for multiclass
            prob_stack = torch.stack(all_probs, dim=0)                 # (M,N) or (M,N,C)
            ensemble_probs = (prob_stack * w).sum(dim=0)               # (N)   or (N,C)

            # optional: normalize if weights don't sum to 1
            wsum = w.sum(dim=0)
            ensemble_probs = ensemble_probs / (wsum + 1e-12)

        else:
            raise ValueError("Unknown ensemble strategy")

        # derive preds from ensemble_probs
        if ensemble_probs.ndim == 1:
            # binary (N,)
            ensemble_preds = (ensemble_probs > 0.5).long()
        else:
            # multiclass (N,C)
            ensemble_preds = ensemble_probs.argmax(dim=1).long()

        return ensemble_preds, ensemble_probs




    # ===================================
    #   Main eval function 
    # ===================================
    def eval(self, return_samples=False, num_samples=None):
        # run ensemble predictions
        ensemble_preds, ensemble_probs = self._ensemble_predict()
  

        # true labels
        labels = torch.cat([y for _, y in self.loader]).cpu().numpy()
        unique_labels = set(labels)
        num_classes = len(unique_labels)

        #print("Detected classes:", unique_labels)
        #print("num_classes:", num_classes)

        preds_np = ensemble_preds.cpu().numpy()
        
        # For binary, if ensemble_probs is (N,2) we may want prob of class1
        if isinstance(ensemble_probs, torch.Tensor):
            probs_np = ensemble_probs.detach().cpu().numpy()
        else:
            probs_np = np.asarray(ensemble_probs)

        if num_classes == 2:
        # ··········· binary class ···········
            # metrics
            acc = (preds_np == labels).mean()
            f1 = f1_score(labels, preds_np, average='macro')
            precision = precision_score(labels, preds_np, average='macro')
            recall = recall_score(labels, preds_np, average='macro')
            #print("[DEBUG] y_true dtype/unique:", labels.dtype, np.unique(labels)[:10])
            #print("[DEBUG] y_score shape/min/max:", None if probs_np is None else (probs_np.shape, float(np.min(probs_np)), float(np.max(probs_np))))
            auc = roc_auc_score(labels, probs_np)#, multi_class='ovo', average='macro')

            # confusion matrix
            cm = confusion_matrix(labels, preds_np)

            # print
            print("\n========== Ensemble Evaluation [2-class] ==========")
            print(f"  • Strategy:  {self.strategy}")
            print(f"  • Accuracy:  {acc * 100:.2f}%")
            print(f"  • AUC:       {auc:.4f}")
            print(f"  • F1:        {f1:.4f}")
            print(f"  • Precision: {precision:.4f}")
            print(f"  • Recall:    {recall:.4f}")
            print("\nConfusion Matrix:")
            print(cm)
            print("===================================================")

            metrics = {
                "Accuracy": acc,
                "AUC": auc,
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "ConfusionMatrix": cm
            }

            
            wandb.log({
                "Strategy": self.strategy, 
                "ensemble_Accuracy": acc,
                "ensemble_F1": f1,
                "ensemble_AUC": auc,
                "ensemble_Precision": precision,
                "ensemble_Recall": recall
            })
        
        #·············· 5 classes ··············
        else:

            #metrics = evaluate_fiveclass(ensemble_preds, labels)
            metrics = self.metric_fn(ensemble_probs, labels)
            # --- Print results ---
            print("\n========== Ensemble Evaluation [5-class] ==========")
            print(f"  • Strategy: {self.strategy}")
            print(f"  • Accuracy: {metrics['acc'] * 100:.2f}%")
            print(f"  • Balanced Acc: {metrics['balanced_acc']:.4f}")
            print(f"  • QWK: {metrics['qwk']:.4f}") 
            print(f"  • Macro F1: {metrics['macro_f1']:.4f}")
            #print("Unique preds:", np.unique(y_pred))
            #print("Unique targets:", np.unique(y_true))
            print("\nConfusion Matrix:\n", metrics['confusion_matrix'])
            print("==================================================\n")



        return metrics, ensemble_preds, ensemble_probs
