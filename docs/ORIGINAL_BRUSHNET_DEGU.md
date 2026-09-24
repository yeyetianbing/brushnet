# Original BrushNet + DeGu

This experiment uses the released original BrushNet architecture and checkpoint while applying multi-reward gradient
guidance during inference. Adaptive feature fusion and timestep-adaptive modulation are both disabled.

## Model modes

`evaluate_brushnet.py` supports the four paper ablations with the same model and pipeline code:

| Experiment | Adaptive fusion | DeGu |
|---|---:|---:|
| Original BrushNet | No | No |
| Original BrushNet + DeGu | No | Yes |
| BrushNet + A | Yes | No |
| BrushNet + A + DeGu | Yes | Yes |

Adaptive fusion is enabled only by passing `--use_adaptive_fusion`. DeGu is enabled by setting
`--reward_guidance_scale` to a value greater than zero.

## Run Original BrushNet + DeGu

Run from the repository root:

```bash
PYTHONPATH=src python examples/brushnet/evaluate_brushnet.py \
  --brushnet_ckpt_path data/ckpt/segmentation_mask_brushnet_ckpt \
  --base_model_path data/ckpt/realisticVisionV60B1_v51VAE \
  --image_save_path runs/evaluation_result/BrushBench/original_brushnet_degu/inside \
  --mapping_file data/BrushBench/mapping_file.json \
  --base_dir data/BrushBench \
  --reward_guidance_scale 5 \
  --guide_per_steps 5 \
  --overall_reward_scale 0.1 \
  --prompt_reward_scale 4.0 \
  --harmonic_reward_scale 1.0 \
  --imagereward_path data/ckpt \
  --harmonic_config_path examples/freeinpaint/metrics/configs.yaml \
  --harmonic_ckpt_path data/ckpt/prefpaintReward.pt \
  --num_inference_steps 50 \
  --seed 1234
```

Do not pass `--use_adaptive_fusion` for this experiment. The script stores all command-line arguments in `args.json`
inside the result directory and rejects accidental reuse of that directory with different arguments.

Use a new result directory for every hyperparameter setting. Pass `--overwrite` only when intentionally regenerating
an existing experiment.

## Recommended ablation commands

- Original BrushNet: use the command above with `--reward_guidance_scale 0`.
- Original BrushNet + DeGu: use the command above as written.
- BrushNet + A: load the A checkpoint, add `--use_adaptive_fusion`, and use `--reward_guidance_scale 0`.
- BrushNet + A + DeGu: load the same A checkpoint, add `--use_adaptive_fusion`, and use the same DeGu settings as the
  original + DeGu experiment.

For a fair comparison, keep the base model, scheduler, inference steps, seed, conditioning scale, dataset, and DeGu
reward weights identical across all four groups.
