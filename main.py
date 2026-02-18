import hydra
import torch
import numpy as np
from collections import Counter
import os
from hydra.utils import call,instantiate
from omegaconf import OmegaConf,DictConfig,open_dict
from sklearn.model_selection import KFold
import wandb

from datasets import get_dataset, get_balanced_augmented_loader_offline, get_loader_and_dataset_online_augmentations
from datasets.idrid.dataset import IDRID
from artifacts.notebooks.plots import plot_label_distribution, print_split_distribution

from visualization.viz import show_tensor_image,plot_label_distribution 
from module_training.augmentations import get_balanced_dataset,build_weighted_dataloader,downsampling_majority_class

from torch.utils.data import DataLoader

from datasets import get_dataset
from models import get_model
from evaluator import Evaluator
from trainer import Trainer
from eval import run_eval_binary, run_eval_multiclass

from datasets import get_dataset_fold, get_dataset_0
from evaluator import Evaluator_2c, Evaluator_5c
from trainer import Trainer_2c, Trainer_5c, Trainer_2c_fold
from ensemble import FeatureExtractor, AttentionFusionModel
from eval import run_eval_2c, run_eval_5c, run_eval_output_ensemble

from module_training import get_balanced_dataset, get_balanced_dataset_fixed_classes,downsampling_majority_class,build_weighted_dataloader
from visualization.viz import show_tensor_image,plot_label_distribution 
from visualization import run_deep_viz






from sklearn.model_selection import StratifiedShuffleSplit

@hydra.main(version_base=None, config_path='config', config_name='default.yaml')
#print(OmegaConf.to_yaml(cfg))
def main(cfg: DictConfig):


    # Initialize Weights & Biases if enabled in config
    maybe_init_wandb(cfg)

    if cfg.name not in ["default", "default_2c_kfold"]:

        device = _device()

        # === 1. dataset + augmentation ===
        train_dataset, train_loader, val_dataset, val_loader = get_dataset_0(cfg, split="train")
        train_aug_dataset, train_aug_loader = apply_aug_and_build_loader(cfg, train_dataset)

        # === 2. model（normal model or feature ensemble） ===
        model = build_model(cfg, device)

        # === 3. according to class number to choose 2c / 5c pipeline ===
        n_classes = cfg.problem.num_classes

        if n_classes == 1:
            # 2-class
            val_evaluator = Evaluator_2c(cfg, val_loader, model, device)
            trainer = Trainer_2c(cfg, train_aug_loader, val_loader, model,val_evaluator, device, cfg.save.path) 

            #resume training from checkpoint
            ##trainer.train(resume=True, checkpoint_path = cfg.check_path)

            trainer.train()

            # deep visualization only for 2c case
            if getattr(cfg, "deep_viz", None) and cfg.deep_viz.enable:
                samples_test = run_eval_2c(cfg, model, device)
                run_deep_viz(cfg, model, device, samples_test)
            else:
                run_eval_2c(cfg, model, device)

        elif n_classes == 5:
            # 5-class
            val_evaluator = Evaluator_5c(cfg, val_loader, model, device)
            trainer = Trainer_5c(cfg, train_aug_loader, val_loader, model,val_evaluator, device, cfg.save.path)

            trainer.train()
            run_eval_5c(cfg, model, device)

        else:
            raise ValueError(f"Unsupported num_classes={n_classes}, only 2 or 5 are handled.")
    
    if cfg.name == 'default_2c_kfold':

        device = _device()
        dataset, _ = get_dataset_fold(cfg, split="train")
        kfold = KFold(n_splits=cfg.kfold.n_splits, shuffle=True, random_state=cfg.kfold.seed)

        fold_best_accs, fold_best_AUCs, fold_best_losses = [], [], []
        fold_best_F1s, fold_best_Precisions, fold_best_Recalls = [], [], []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset   = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

            train_aug_dataset, train_aug_loader = apply_aug_and_build_loader(cfg, train_dataset)

            model = get_model(cfg).to(device)
            val_evaluator = Evaluator_2c(cfg, val_loader, model, device)

            trainer = Trainer_2c_fold(cfg, train_aug_loader, val_loader, model, val_evaluator,device, cfg.save.path, fold,fold_best_accs, fold_best_AUCs, fold_best_losses,fold_best_F1s, fold_best_Precisions, fold_best_Recalls)
            trainer.train()

        # After all folds
        mean_acc = np.mean(trainer.fold_best_accs)
        std_acc = np.std(trainer.fold_best_accs)
        mean_AUC = np.mean(trainer.fold_best_AUCs)
        std_AUC = np.std(trainer.fold_best_AUCs)
        mean_loss = np.mean(trainer.fold_best_losses)
        std_loss = np.std(trainer.fold_best_losses)
        mean_F1 = np.mean(trainer.fold_best_F1s)
        std_F1 = np.std(trainer.fold_best_F1s)
        mean_Precision = np.mean(trainer.fold_best_Precisions)
        std_Precision = np.std(trainer.fold_best_Precisions)
        mean_Recall = np.mean(trainer.fold_best_Recalls)
        std_Recall = np.std(trainer.fold_best_Recalls)
        print(f"\n===== K-Fold Cross-Validation Results =====")
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Mean AUC: {mean_AUC:.4f} ± {std_AUC:.4f}")
        print(f"Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"Mean F1 Score: {mean_F1:.4f} ± {std_F1:.4f}")
        print(f"Mean Precision: {mean_Precision:.4f} ± {std_Precision:.4f}")
        print(f"Mean Recall: {mean_Recall:.4f} ± {std_Recall:.4f}")
        #wandb.init(project="diabetic_retinopathy_K-Fold", name=f"K-Fold_Results")

        wandb.log({
        "val_acc_mean": mean_acc,
        "val_acc_std": std_acc,
        "val_AUC_mean": mean_AUC,
        "val_AUC_std": std_AUC,
        "val_loss_mean": mean_loss,
        "val_loss_std": std_loss,
        "val_F1_mean": mean_F1,
        "val_F1_std": std_F1,
        "val_Precision_mean": mean_Precision,
        "val_Precision_std": std_Precision,
        "val_Recall_mean": mean_Recall,
        "val_Recall_std": std_Recall
        })
        wandb.finish()

    
    

    elif cfg.name == "default":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ----------------------------------------------------------------------
        # Get Datasets
        if cfg.dataset.name == "IDRID":
            # ----- Create split once -----
            full_dataset = IDRID(cfg, split='train')
            labels = torch.tensor([full_dataset[i][1] for i in range(len(full_dataset))]).numpy()

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
            train_idx, val_idx = next(splitter.split(torch.zeros(len(labels)), labels))

            train_indices = train_idx.tolist()
            val_indices = val_idx.tolist()

            print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

            # ----- Build loaders from fixed indices -----
            train_dataset_before_aug, train_dataloader_before_aug = get_dataset(cfg, train_indices, "train")
            eval_dataset, eval_dataloader = get_dataset(cfg, val_indices, split = "eval")
            test_dataset, test_dataloader = get_dataset(cfg, indices = None,split='test')
            
            # Build NEW augmented + balanced train loader
            train_dataset, train_dataloader = get_balanced_augmented_loader_offline(cfg,train_subset = train_dataset_before_aug) 
        
        else:
            print ('yes')
            # Get Datasets
            # train_dataset, train_dataloader = get_dataset(cfg, indices = None, split='train')
            train_dataset, train_dataloader = get_loader_and_dataset_online_augmentations(cfg, split='train')
            eval_dataset, eval_dataloader = get_dataset(cfg, indices = None, split='eval')
            test_dataset, test_dataloader = get_dataset(cfg, indices = None, split='test')


        # Get Model
        model = get_model(cfg)
        model.to(device)

        # # Get Evaluator
        val_evaluator = Evaluator(cfg, eval_dataloader, model, device)

        # Initialize Weights & Biases
        wandb.init(
        project="diabetic_retinopathy",
        # entity="st192588-university-of-stuttgart",  
        # id="d3183bhq",              # If you want to resume add your run’s ID
        # resume="allow", 
        config={
            "epochs": cfg.epochs,
            "learning_rate": cfg.lr,
            "batch_size": cfg.dataset.batch_size,
            "optimizer": "Adam"
        }
    )

        trainer = Trainer(cfg,
                    train_dataloader,
                    eval_dataloader,
                    model,
                    val_evaluator,
                    device
                    )

        if cfg.task == '2c':
            trainer.train()
            #trainer.train(resume=True, checkpoint_path="artifacts/checkpoints/checkpoint_epoch8.pth") #resume training from checkpoint
            run_eval_binary(cfg, model, device)
        else:
            trainer.train_multiclass()
            #trainer.train_multiclass(resume=True, checkpoint_path="artifacts/checkpoints/checkpoint_epoch8.pth") #resume training from checkpoint
            run_eval_multiclass(cfg, model, device)

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

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(cfg, device):
    """  
    - 2c / 5c: directly get_model
    - feature_ensemble：build submodel + FeatureExtractor + AttentionFusionModel
    """
    n_classes = cfg.problem.num_classes

    # === situation A：feature-level ensemble ===
    if "feature_ensemble" in cfg.name:
        models = []
        for item in cfg.ensemble_models:
            # use hydra.compose to load submodel config
            model_cfg = hydra.compose(config_name=f"model/{item.cfg}")
            with open_dict(model_cfg):
                if "model" in model_cfg:
                    model_cfg.model.num_classes = n_classes
            m = get_model(model_cfg)
            ckpt = torch.load(item.ckpt, map_location=device, weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(device).eval()
            models.append(m)

        # build extractor
        extractors = []
        for item, model in zip(cfg.ensemble_models, models):
            cfg_name = item.cfg.lower()
            if "dense" in cfg_name:
                model_type = "dense121"
            elif "convnext" in cfg_name:
                model_type = "convnext"
            elif "efficient" in cfg_name:
                model_type = "efficientnet"
            else:
                raise ValueError(f"Unknown model type: {item.cfg}")
            extractors.append(FeatureExtractor(model, model_type))

        fusion_model = AttentionFusionModel(extractors, num_classes=n_classes).to(device)
        print("FEATURE FUSION DONE")

        # freeze backbone parameter，only train fusion head
        for e in extractors:
            for p in e.parameters():
                p.requires_grad = False

        return fusion_model

    # === situation B：normal Model ===
    else:
        model = get_model(cfg).to(device)
        return model

def apply_aug_and_build_loader(cfg, train_dataset):
    # default loader
    def default_loader(ds):
        return DataLoader(ds, batch_size=cfg.dataset.batch_size, shuffle=True, pin_memory=True)

    if (not hasattr(cfg, "train_module")) or (not cfg.train_module.aug.enable) or (cfg.train_module.aug.method == "none"):
        return train_dataset, default_loader(train_dataset)

    m = cfg.train_module.aug.method

    print(f"Applying augmentation method: {m}")

    if m == "upsample_all":
        ds = get_balanced_dataset(train_dataset, target_ratio_up=cfg.train_module.aug.target_ratio_up)  
        return ds, default_loader(ds)

    if m == "upsample_fixed":
        ds = get_balanced_dataset_fixed_classes(
            train_dataset,
            target_classes=cfg.train_module.aug.target_classes,
            target_ratio_up=cfg.train_module.aug.target_ratio_up,
        )  
        return ds, default_loader(ds)

    if m == "downsample":
        ds = downsampling_majority_class(train_dataset, target_ratio_down=cfg.train_module.aug.target_ratio_down)  # :contentReference[oaicite:4]{index=4}
        return ds, default_loader(ds)

    if m == "weighted_sampler":
        loader = build_weighted_dataloader(train_dataset, batch_size=cfg.dataset.batch_size)  # :contentReference[oaicite:5]{index=5}
        return train_dataset, loader

    raise ValueError(f"Unknown aug.method: {m}")



if __name__ == '__main__':
    main()