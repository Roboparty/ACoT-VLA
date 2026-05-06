import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import json
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download, nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi import transforms as _transforms
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image

config = _config.get_config("acot_icra_sim_custom_dataset")
checkpoint_dir = "/mnt/h20_1data0/ygx_dataset/checkpoints/acot_icra_sim_custom_dataset/resume_from_baseline/99999"

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
ckpt_root = os.path.dirname(checkpoint_dir)
steps = int(os.path.basename(checkpoint_dir))


data_config = config.data.create(config.assets_dirs, config.model)
base_dataset = _data_loader.create_torch_dataset(data_config, config.model)
dataset = _data_loader.TransformedDataset(
    base_dataset,
    [
        *data_config.repack_transforms.inputs,
    ],
)

# Build the same data_transforms used during training (input + output side, for GT extraction)
# GT actions need to go through both input transforms (→ delta) and output transforms (→ absolute)
# so they are in the same absolute space as the inferred actions.
gt_input_transforms = _transforms.compose(data_config.data_transforms.inputs)
gt_output_transforms = _transforms.compose(data_config.data_transforms.outputs)

max_inferences = 20 # set to determine max infer times
chunk_size = 30

all_gt_actions = []
all_inferred_actions = []

for i in range(max_inferences):
    data = dataset[i * chunk_size]
    if i >= max_inferences:
        break

    # Apply the same data transforms as training to get GT actions in the same absolute space as inferred
    gt_data = gt_input_transforms(copy.deepcopy(data))
    gt_data = gt_output_transforms(gt_data)
    gt_actions = gt_data["actions"]

    inferred_result = policy.infer(data)
    inferred_actions = inferred_result['actions']
    num_out_dims = inferred_actions.shape[1]

    gt_actions = gt_actions.squeeze()[:inferred_actions.shape[0], :num_out_dims]
    all_gt_actions.append(gt_actions)
    all_inferred_actions.append(inferred_actions)

if all_gt_actions:
    gt_actions_continuous = np.concatenate(all_gt_actions, axis=0)
    inferred_actions_continuous = np.concatenate(all_inferred_actions, axis=0)
    total_steps, num_dims = inferred_actions_continuous.shape
    time_steps_per_inference = gt_actions_continuous.shape[0] // max_inferences

    # Compute metrics
    abs_error = np.abs(gt_actions_continuous - inferred_actions_continuous)
    mae_per_dim = abs_error.mean(axis=0)
    mse_per_dim = ((gt_actions_continuous - inferred_actions_continuous) ** 2).mean(axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    overall_mae = mae_per_dim.mean()
    overall_rmse = rmse_per_dim.mean()
    max_ae_per_dim = abs_error.max(axis=0)

    print(f"\n{'='*60}")
    print(f"  Open-loop Evaluation @Step {steps}  ({max_inferences} samples)")
    print(f"{'='*60}")
    print(f"  Overall MAE:  {overall_mae:.6f} rad ({np.degrees(overall_mae):.4f} deg)")
    print(f"  Overall RMSE: {overall_rmse:.6f} rad ({np.degrees(overall_rmse):.4f} deg)")
    print(f"{'-'*60}")
    print(f"  {'Dim':<6} {'MAE(rad)':<14} {'MAE(deg)':<14} {'RMSE(rad)':<14} {'MaxAE(rad)':<14}")
    print(f"{'-'*60}")
    for d in range(num_dims):
        print(f"  {d:<6} {mae_per_dim[d]:<14.6f} {np.degrees(mae_per_dim[d]):<14.4f} {rmse_per_dim[d]:<14.6f} {max_ae_per_dim[d]:<14.6f}")
    print(f"{'='*60}\n")

    nrows = (num_dims + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(20, nrows * 4.5), sharex=True)
    axes = axes.flatten()

    x_axis = np.arange(total_steps)
    for dim_idx in range(num_dims):
        ax = axes[dim_idx]

        ax.plot(x_axis, gt_actions_continuous[:, dim_idx], label='Ground Truth', color='cornflowerblue', alpha=0.9)
        ax.plot(x_axis, inferred_actions_continuous[:, dim_idx], label='Inferred', color='tomato', linestyle='--', alpha=0.9)

        start_indices = np.arange(0, total_steps, time_steps_per_inference)
        ax.scatter(start_indices, gt_actions_continuous[start_indices, dim_idx], c='blue', marker='o', s=40, zorder=5, label='GT Start')
        ax.scatter(start_indices, inferred_actions_continuous[start_indices, dim_idx], c='darkred', marker='x', s=40, zorder=5, label='Inferred Start')

        ax.set_title(f'Dim {dim_idx} | MAE={mae_per_dim[dim_idx]:.4f} ({np.degrees(mae_per_dim[dim_idx]):.2f}°)')
        ax.set_ylabel('Value (rad)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

    for dim_idx in range(num_dims, len(axes)):
        axes[dim_idx].set_visible(False)

    fig.supxlabel(f'Continuous Timestep (across {max_inferences} inferences)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'GT vs Inferred @Step {steps} | Overall MAE={overall_mae:.4f} rad ({np.degrees(overall_mae):.2f}°)', fontsize=16)
    plt.savefig(f'{ckpt_root}/inferred_vs_gt_actions-{steps}-model.png', dpi=300, bbox_inches='tight')
    # plt.show()

else:
    print("No data was collected for plotting.")
