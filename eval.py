import hydra
import torch
from omegaconf import OmegaConf

from datasets import get_dataset, get_graham_idrid_dataset
from models import get_model
from evaluator import Evaluator

import hydra
import torch
import wandb
import numpy as np

from datasets import get_dataset_0
from models import get_model

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score, 
    precision_score, 
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from hydra.utils import instantiate
from evaluator import Evaluator_2c, Evaluator_5c
from evaluator import Evaluator_output_ensemble as EnsembleEvaluator



# ======================================================
# main 
# ======================================================
print("PROGRAM START", flush=True)
@hydra.main(version_base=None, config_path='config', config_name='default.yaml')
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("ENTER MAIN", flush=True)
    print(cfg, flush=True)
    print("Running Test Evaluation...", flush=True)
    # Get Model
    model = get_model(cfg)
    model.to(device)

    if cfg.name =="default":
        if cfg.task == '2c':
            run_eval_binary(cfg, model, device)
        elif cfg.task == '5c':
            run_eval_multiclass(cfg, model, device)

    #run_eval_2c(cfg, model, device)
    #run_eval_5c(cfg, model, device)
    #run_eval_output_ensemble(cfg, models, device)

# ======================================================
# run_eval
# ======================================================

def run_eval_binary(cfg, model, device):
    # 1. Load the best model checkpoint
    # best_model_path = f"artifacts/models/idird_SmallDRNet_best_model_79.pth" 
    best_model_path = f'artifacts/idrid_{cfg.model.name}_best_model.pth' 
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Get test loader
    _, test_loader = get_dataset(cfg, split='test')

    all_preds = []
    all_labels = []

    # 3. Run predictions
    with torch.no_grad():
        for x, y_true in test_loader:
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = model(x)
            preds = (y_pred > 0.5).long()

            # ---- FIX 2: flatten numpy arrays ----
            all_preds.extend(preds.cpu().numpy().reshape(-1).tolist())
            all_labels.extend(y_true.cpu().numpy().reshape(-1).tolist())


    # 4. Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, digits=4))
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    # plt.savefig(f"artifacts/images/{cfg.dataset.name}_{cfg.model.name}_confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to artifacts/images/confusion_matrix.png")

    if wandb.run is not None:
        wandb.log({
        "test/accuracy": acc,
        "test/f1": f1,
        "test/precision": precision,
        "test/recall": recall,
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=["No_DR", "DR"]
        )
    })


def run_eval_multiclass(cfg, model, device):

    # wandb.init(
    #    project="diabetic_retinopathy",
    #    entity="st192588-university-of-stuttgart",  
    #    id="d3183bhq",              # <-- your run’s ID
    #    resume="allow"
#    )
    # -----------------------------
    # 1. Load trained checkpoint
    # -----------------------------
    # best_model_path = "artifacts/models/eyepacs_ResNet18Multiclass_multiclass_56.pth"
    best_model_path = f"artifacts/{cfg.dataset.name.lower()}_{cfg.model.name}_multiclass_best_model.pth"

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # -----------------------------
    # 2. Test dataloader
    # -----------------------------

    if cfg.dataset.name == 'IDRID':
         _, test_loader = get_graham_idrid_dataset(cfg, split='test')  ##graham
    else:
        _, test_loader = get_dataset(cfg, split="test")

    all_preds = []
    all_labels = []

    # -----------------------------
    # 3. Inference
    # -----------------------------
    with torch.no_grad():
        for x, y_true in test_loader:
            x = x.to(device)
            y_true = y_true.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_true.cpu().numpy())

    # -----------------------------
    # 4. Metrics (MULTICLASS!)
    # -----------------------------
    acc = accuracy_score(all_labels, all_preds)

    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")
    print(f"Weighted F1:     {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            digits=4,
            target_names=[
                "0",
                "1",
                "2",
                "3",
                "4",
            ],
        )
    )

    # -----------------------------
    # 5. Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
    # 5. Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    # plt.savefig(f"artifacts/images/{cfg.dataset.name}_{cfg.model.name}_confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to artifacts/confusion_matrix.png")

    if wandb.run is not None:
        wandb.log({
        "test/accuracy": acc,
        "test/f1_macro": f1_macro,
        "test/f1_weighted": f1_weighted,
        "test/precision_macro": precision_macro,
        "test/recall_macro": recall_macro,
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=[
                "0",
                "1",
                "2",
                "3",
                "4",
            ],
        ),
    })


def run_eval_2c(cfg, model, device):

    # Load Best Model
    
    #
    #_, test_dataloader = get_dataset(cfg, split='test')
    test_dataset, test_dataloader = get_dataset_0(cfg, split='test')
    test_evaluator = Evaluator_2c(cfg, test_dataloader, model, device)

    if cfg.test.check_mode: 
        # check modus turn on, use provided checkpoint path
        checkpoint = torch.load(cfg.test.check_path, map_location=device, weights_only=False)
        print(f"Checkpath:{cfg.test.check_path}")
        clean_name = cfg.test.check_path.replace("./checkpoints/", "")
    else:
        # default modus, use training checkpoint path    
        checkpoint = torch.load(cfg.save.path, map_location=device, weights_only=False)
        clean_name = f"{cfg.save.date_prefix}_{cfg.save.model_name}"



    # Lingzhi: convnext, efficientnet, dense121
    model.load_state_dict(checkpoint['model_state_dict'])

    # Durga: drnet, resnet_transfer, basic_cnn
    #model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    #print(f"Checkpath:{cfg.check_path}")

    if cfg.deep_viz.enable:
        print("\n!! Deep Visualization Enabled")
        test_loss,test_acc,metrics,samples_test = test_evaluator.eval(return_samples=True, num_samples=cfg.deep_viz.num_samples)
        return samples_test
    else:
        test_loss,test_acc,metrics = test_evaluator.eval()


    # Initialize WandB
    maybe_init_wandb(cfg)
    #wandb.init(entity="zizi1217-uni-stuttgart",project="diabetic_retinopathy_2class_eval")
    #wandb.watch(self.model, self.criterion, log="all", log_freq=100) #（? log_freq=100

    wandb.log({
        "test/test_loss": test_loss,
        "test/test_acc": test_acc,
        "test/test_auc": metrics["auc"],
        "test/test_f1": metrics["f1"],
        "test/test_precision": metrics["precision"],
        "test/test_recall": metrics["recall"]
        })
    
    

def run_eval_5c(cfg, model, device):

    # 1. Load Dataset
    test_dataset, test_loader = get_dataset_0(cfg, split='test')
    evaluator = Evaluator_5c(cfg, test_loader, model, device)


    # need to be thinked more

    # 2. Load checkpoint
    if cfg.test.check_mode: 
        # check modus turn on, use provided checkpoint path 
        checkpoint = torch.load(cfg.test.check_path, map_location=device, weights_only=False)
        print(f"Checkpath:{cfg.test.check_path}")
        clean_name = cfg.test.check_path.replace("./checkpoints/", "")
    else:  
        # default modus, use training checkpoint path
        checkpoint = torch.load(cfg.save.path, map_location=device, weights_only=False)
        clean_name = f"{cfg.save.date_prefix}_{cfg.save.model_name}"



    #checkpoint = torch.load(cfg.check_path, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    #print(f"Checkpath:{cfg.check_path}")

    # 3. Run evaluator
    metrics = evaluator.eval()

    # 4. Log to WandB
    maybe_init_wandb(cfg)

    wandb.log({
        "test/test_accuracy": metrics["acc"],
        "test/test_balanced_acc": metrics["balanced_acc"],
        "test/test_macro_f1": metrics["macro_f1"],
        "test/test_qwk": metrics["qwk"],
        "test/confusion_matrix": metrics["confusion_matrix"]
    })

    return metrics


def run_eval_output_ensemble(cfg, models, device,strategy="prob_avg"):
    #maybe_init_wandb(cfg)
    print("\n==> OUTPUT ENSEMBLE START ")
    print("CHECK 1: start run_eval")
    # === get test dataset ===
    #print(f"[DEBUG] cfg.problem={cfg.problem} ")
    
    test_dataset, test_dataloader = get_dataset_0(cfg, split='test')
    
    print("CHECK 2: dataset loaded")

    evaluator = EnsembleEvaluator(cfg, test_dataloader, models, device,strategy)
    print("CHECK 3: evaluator created")
    metrics, preds, probs = evaluator.eval()
   

    print("CHECK 4: model.eval done")
    


def maybe_init_wandb(cfg):
    if (
        hasattr(cfg, "wandb")
        and hasattr(cfg.wandb.init, "enable")
        and cfg.wandb.init.enable
    ):
        wandb.init(
            #entity=cfg.wandb.init.entity,
            project=cfg.wandb.init.project,
            name=cfg.wandb.init.name,
            group=cfg.wandb.init.group,
            tags=list(cfg.wandb.init.tags) if "tags" in cfg.wandb.init else None,
            id=cfg.wandb.init.id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )


if __name__ == '__main__':
    main()