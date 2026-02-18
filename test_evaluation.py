import hydra
import torch
from collections import Counter
import os
import wandb
import numpy as np
from omegaconf import OmegaConf,open_dict

from datasets import get_dataset_0
from models import get_model

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score
)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score



from hydra.utils import instantiate
from evaluator import Evaluator_2c, Evaluator_5c
from evaluator import Evaluator_output_ensemble as EnsembleEvaluator



from ensemble import FeatureExtractor,FusionHead,FusionModel,AttentionFusionHead,AttentionFusionModel

from eval import run_eval_2c, run_eval_5c, run_eval_output_ensemble

from visualization import run_deep_viz





@hydra.main(version_base=None, config_path='config', config_name='default.yaml')
#print(OmegaConf.to_yaml(cfg))
def main(cfg):

    name = cfg.test.name
    print(f"Running test: {name}")

    if name not in TESTS:
        raise ValueError(f"Unknown task: {name}. Available: {list(TESTS.keys())}")
    TESTS[name](cfg)


def test_eval_2c(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg).to(device)
    if cfg.deep_viz.enable:
        samples_test = run_eval_2c(cfg, model, device)
        run_deep_viz(cfg, model, device, samples_test)
    else:
        run_eval_2c(cfg, model, device)



def test_eval_5c(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg).to(device)
    run_eval_5c(cfg, model, device)

def test_eval_ensemble(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # =========== test ensemble model ==========
    models = []
    n_classes = cfg.problem.num_classes
    # =============================
    # Load all ensemble models
    # =============================
    for item in cfg.ensemble_models:

        model_cfg_name = item.cfg      # e.g. "dense121.yaml"
        ckpt_path = item.ckpt          # e.g. "checkpoints/dense/...pth"

        # load model config
        model_cfg = hydra.compose(config_name=f"model/{model_cfg_name}")
        with open_dict(model_cfg):
           # model_cfg.num_classes = n_classes
            if "model" in model_cfg:
                model_cfg.model.num_classes = n_classes

        # build model
        model = get_model(model_cfg)

        # load weights
        ckpt = torch.load(ckpt_path, map_location=device,weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        model.to(device)
        model.eval()

        models.append(model)

    print(f"Loaded {len(models)} models.")




    if cfg.name == "default_5c_output_ensemble" or cfg.name == "default_2c_output_ensemble" :

        run_eval_output_ensemble(cfg, models, device, strategy="prob_avg")
        run_eval_output_ensemble(cfg, models, device, strategy="hard_vote")
        run_eval_output_ensemble(cfg, models, device, strategy="logit_avg")
        run_eval_output_ensemble(cfg, models, device, strategy="weighted_prob")

    else:

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
        print(f"FEATURE FUSION DONE")
        

        if cfg.name == "default_2c_feature_ensemble": 
                
            if cfg.deep_viz.enable:
                samples_test = run_eval_2c(cfg, fusion_model, device)
                run_deep_viz(cfg, fusion_model, device, samples_test)
            else:
                run_eval_2c(cfg, fusion_model, device)
            
        elif cfg.name == "default_5c_feature_ensemble": 

            run_eval_5c(cfg, fusion_model, device)
        


TESTS = {
    "test_2c": test_eval_2c,
    "test_5c": test_eval_5c,
    "test_ensemble": test_eval_ensemble,
}










if __name__ == '__main__':
    main()