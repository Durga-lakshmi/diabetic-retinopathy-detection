# HOW TO RUN - Diabetic Retinopathy 
This project uses **Hydra configurations** to switch between different DR experiment variants.

Different functionalities are activated by specifying `--config-name` in the command line.

All training-related experiments are launched via `main.py`.

### 1. Custom Architecture - CNN
1.1**Default Configurations** (Please use these configurations in default.yaml)

-------------------------------------------
```yaml
  defaults:
    - dataset: idrid
    - model: small_drnet
  task : 2c
```
-------------------------------------------

1.2 Run the command
```python
python3 main.py --config-name=default
```
Note : To view model artifacts and plots such as class distributions, confusion matrix, please navigate to : dl-lab-25w-team03/diabetic_retinopathy/artifacts/images.
       Please find the path files of the best models we trained inside : dl-lab-25w-team03/diabetic_retinopathy/artifacts/models
       
-------------------------
### 2. Transfer Learning for Multiclass Classification
2.1 **Default Configurations** (Please use these configurations in default.yaml)  

-------------------------------------------
```yaml
defaults:
  - dataset: eyepacs
  - model: resnet_multiclass
task : 5c
```
-------------------------------------------
2.2 Run the command
```python
python3 main.py --config-name=default
```
To evaluate the multiclass model on IDRID dataset use this configuration

  -------------------------------------------
  ```yaml
  defaults:
  - dataset: eyepacs
  - model: resnet_multiclass
  task : 5c
  ```
  -------------------------------------------
  
  Run this command
  ```python
  python3 eval.py --config-name=default
  ```

Note : To view model artifacts and plots such as class distributions, confusion matrix, please navigate to : dl-lab-25w-team03/diabetic_retinopathy/artifacts/images.
       Please find the path files of the best models we trained inside : dl-lab-25w-team03/diabetic_retinopathy/artifacts/models  
       
---
### 3. Ensemble learning
**Dataset: IDRiD**  
**Models: DenseNet-121, ConvNeXt, EfficientNet**
#### 3.1 Individual Model

**Two-class classification**

```bash
python3 main.py --config-name=default_2c
```

**Two-class with k-fold cross-validation**

```bash
python3 main.py --config-name=default_2c_kfold
```

**Five-class classification**

```bash
python3 main.py --config-name=default_5c
```

#### 3.2 Feature-level Ensemble 
**IMPORTANT：If want to run feature-level ensemble learning, muss change this part to use the trained models**

```yaml
ensemble_models:
- { cfg: "dense121.yaml", ckpt: <dense121_check_path> }
- { cfg: "convnext.yaml", ckpt: <convnext_check_path> }
- { cfg: "efficientnet.yaml", ckpt: <efficientnet_check_path> }
```

**REMENBER:** In Binary/Multiclasses classification use the corresponding paths   
**Two-class with feature-level ensemble**

```bash
python3 main.py --config-name=default_2c_feature_ensemble
```

**Five-class with feature-level ensemble**

```bash
python3 main.py --config-name=default_5c_feature_ensemble
```

Each configuration controls the training setup (number of classes, cross-validation, and feature-level ensembling) through predefined Hydra config files.

---
#### 3.3 Evaluation: Individual Model + Feature-level Ensemble Model

```bash
python3 test_evaluation.py --config-name=default_2c / default_5c / default_2c_feature_ensemble / default_5c_feature_ensemble
```
**IMPORTANT:change some parts in defaults**

```yaml
- test:  
    check_mode: false -> true  
    check_path: <your_check_path_after_training>
```

  
**If want to do deep visualization (ONLY for 2-class):**


```yaml
- deep_viz:  
   enable: false -> true
```


#### 3.4 Output-level Ensemble (Evaluation Only)

Output-level ensemble is performed **after training**, using a dedicated evaluation script.

**IMPORTANT：If want to run output-level ensemble learning, muss change this part to use the trained models**

```yaml
ensemble_models:
- { cfg: "dense121.yaml", ckpt: <dense121_check_path> }
- { cfg: "convnext.yaml", ckpt: <convnext_check_path> }
- { cfg: "efficientnet.yaml", ckpt: <efficientnet_check_path> }
```

**REMENBER:** In Binary/Multiclasses classification use the corresponding paths     
**Two-class output-level ensemble**

```bash
python3 test_evaluation.py --config-name=default_2c_output_ensemble
```

**Five-class output-level ensemble**

```bash
python3 test_evaluation.py --config-name=default_5c_output_ensemble
```

This step aggregates predictions from multiple trained models at the output level without additional training.

---

### 4. Notes

- All experiment variants are controlled purely by configuration files; no code modification is required.
- Feature-level ensemble is handled during training (`main.py`).
- Output-level ensemble is implemented as a post-hoc evaluation procedure (`test_evaluation.py`).
  
---
# RESULTS
## Diabetic Retinopathy
### Custom Architecture and Transfer Learning for Multiclass Classification
| Model           | Training Dataset |   Test Dataset    |Test Accuracy    | Classification Type | Training Strategy |
|-----------------|------------------|-------------------|-----------------|---------------------|--------------------
| small_drnet     | IDRID            |      IDRID        |     79.8%       |       Binary        |Train from Scratch |
| ConvexNet       | IDRID            |      IDRID        |    87. 38%      |       Binary        |Transfer Learning  |            
| Resnet18        | EyePACS          |      IDRID        |    56.21%       |      Multiclass     |Transfer Learning  |           
| Resnet18        | EyePACS          |      EyePACS      |    78.25%       |      Multiclass     |Transfer Learning  | 

---
### Ensemble Learning
Binary | **IDRiD**
| Model / Strategy           | Accuracy (%) | AUC (%) | F1-score (%) | Precision (%) | Recall (%) | 
| -------------------------- | ------------ | ------- | ------------ | ------------- | ---------- |
| DenseNet-121               | 82.52        | 93.15   | 86.96        | 81.08         | 93.75      |
| ConvNeXt                   | **88.35**    | 92.33   | **90.16**    | **94.83**     | 85.94      | 
| EfficientNet               | 83.50        | 88.94   | 86.18        | 89.83         | 82.81      | 
| Feature-level Ensemble     | 84.47        | 91.09   | 86.89        | 91.38         | 82.81      | 
| Output Ensemble (Prob Avg) | 85.44        | 92.91   | 84.74        | 84.37         | 85.28      | 
| Output Ensemble (Others*)  | 84.47        | 92.67   | 83.79        | 83.37         | 84.50      |

Multiclass | **IDRiD**
| Model / Strategy           | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | QWK (%)   |
| -------------------------- | ------------ | ---------------- | ------------ | --------- |
| DenseNet-121               | **58.25**    | **45.42**        | **45.12**    | 69.68     |
| ConvNeXt                   | 54.37        | 40.88            | 40.97        | **70.74** |
| EfficientNet               | 54.37        | 41.97            | 41.47        | 65.68     |
| Feature-level Ensemble     | 57.28        | 42.10            | 41.39        | 67.57     |
| Output Ensemble (Prob Avg) | 52.43        | 39.44            | 38.96        | 66.63     |
| Output Ensemble (Others*)  | 54.37        | 40.62            | 39.98        | 68.54     | 

---

