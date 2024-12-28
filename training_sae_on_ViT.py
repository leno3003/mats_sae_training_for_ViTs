import os
import sys
import torch
import wandb
import json
import pickle
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from tqdm import tqdm
from functools import partial


sys.path.append("..")

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from vit_sae_analysis.dashboard_fns import get_feature_data

if torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

cfg = ViTSAERunnerConfig(
    
    # Data Generating Function (Model + Training Distibuion)
    class_token = True,
    image_width = 224,
    image_height = 224,
    model_name = "openai/clip-vit-base-patch32", # "openai/clip-vit-large-patch14",
    module_name = "resid",
    block_layer = -2,
    dataset_path = "evanarlian/imagenet_1k_resized_256",
    use_cached_activations = False,
    cached_activations_path = None,
    d_in = 768, # 1024,
    
    # SAE Parameters
    expansion_factor = 8, # 64,
    b_dec_init_method = "mean",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    lr_scheduler_name="constantwithwarmup",
    batch_size = 1024,
    lr_warm_up_steps=500,
    total_training_tokens = 2_621_440,
    n_batches_in_store = 15,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_method = None,
    feature_sampling_window = 64,
    dead_feature_window=64,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = False, # True,
    wandb_project= "SAE_on_ViT-B32_clip", # "SAE_on_ViT-L_clip",
    wandb_entity = "test_1",
    wandb_log_frequency=20,
    
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 0,
    checkpoint_path = "/scratch/sae_on_vit/checkpoints",
    dtype = torch.float32,
    )

torch.cuda.empty_cache()
sparse_autoencoder, model = vision_transformer_sae_runner(cfg)
sparse_autoencoder.eval()
feature_idx = list(range(10))

# feature_data = 

get_feature_data(
    sparse_autoencoder,
    model,
    number_of_images = 524_288,
    number_of_max_activating_images = 20,
)

# for test_idx in feature_idx:
#     html_str = feature_data[test_idx].get_all_html()
#     with open(f"/scratch/sae_on_vit/data_{test_idx:04}.html", "w") as f:
#         f.write(html_str)

print("*****Done*****")