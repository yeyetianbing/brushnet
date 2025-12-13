import torch
import json
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import open_clip
import hpsv2
import ImageReward as RM
import math

def rle2mask(mask_rle, shape):
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)


class MetricsCalculator:
    def __init__(self, device, ckpt_path="data/ckpt") -> None:
        self.device = device
        print("Initializing metrics calculators...")

        # clip
        print("Loading CLIP model...")
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)

        # lpips
        print("Loading LPIPS model...")
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)

        # aesthetic model
        print("Loading Aesthetic model...")
        self.aesthetic_model = torch.nn.Linear(768, 1)
        aesthetic_model_ckpt_path = os.path.join(ckpt_path, "sa_0_4_vit_l_14_linear.pth")

        if not os.path.exists(aesthetic_model_ckpt_path):
            raise FileNotFoundError(f"Aesthetic model not found at {aesthetic_model_ckpt_path}")

        self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path))
        self.aesthetic_model.eval()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')

        # image reward model
        print("Loading ImageReward model...")
        self.imagereward_model = RM.load("ImageReward-v1.0")

        print("All models loaded successfully!")

    def calculate_image_reward(self, image, prompt):
        reward = self.imagereward_model.score(prompt, [image])
        return reward

    def calculate_hpsv21_score(self, image, prompt):
        result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
        return result.item()

    def calculate_aesthetic_score(self, img):
        image = self.clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.aesthetic_model(image_features)
        return prediction.cpu().item()

    def calculate_clip_similarity(self, img, txt):
        img = np.array(img)
        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        return score

    def calculate_psnr(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.
        img_gt = np.array(img_gt).astype(np.float32) / 255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum / difference_size

        if mse < 1.0e-10:
            return 1000
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def calculate_lpips(self, img_gt, img_pred, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)

        score = self.lpips_metric_calculator(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1)
        score = score.cpu().item()

        return score

    def calculate_mse(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.
        img_gt = np.array(img_gt).astype(np.float32) / 255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum / difference_size

        return mse.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate generated images')
    parser.add_argument('--image_save_path',
                        type=str,
                        default="runs/evaluation_result/BrushBench/brushnet_segmask/inside",
                        help='Path where generated images are saved')
    parser.add_argument('--mapping_file',
                        type=str,
                        default="data/BrushBench/mapping_file.json",
                        help='Path to mapping file')
    parser.add_argument('--base_dir',
                        type=str,
                        default="data/BrushBench",
                        help='Base directory for source images')
    parser.add_argument('--mask_key',
                        type=str,
                        default="inpainting_mask",
                        help='Key for mask in mapping file')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default="data/ckpt",
                        help='Path to model checkpoints')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load mapping file
    print(f"Loading mapping file from {args.mapping_file}")
    with open(args.mapping_file, "r") as f:
        mapping_file = json.load(f)

    print(f"Found {len(mapping_file)} images to evaluate")

    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator(device, ckpt_path=args.ckpt_path)

    # Create evaluation dataframe
    evaluation_df = pd.DataFrame(columns=['Image ID', 'Image Reward', 'HPS V2.1', 'Aesthetic Score', 'PSNR', 'LPIPS', 'MSE', 'CLIP Similarity'])

    # Evaluate each image
    for idx, (key, item) in enumerate(mapping_file.items()):
        print(f"\n[{idx+1}/{len(mapping_file)}] Evaluating image {key} ...")

        image_path = item["image"]
        mask = item[args.mask_key]
        prompt = item["caption"]

        # Check if generated image exists
        tgt_image_path = os.path.join(args.image_save_path, image_path)
        if not os.path.exists(tgt_image_path):
            print(f"  WARNING: Generated image not found at {tgt_image_path}, skipping...")
            continue

        # Load images
        src_image_path = os.path.join(args.base_dir, image_path)
        src_image = Image.open(src_image_path).resize((512, 512))
        tgt_image = Image.open(tgt_image_path).resize((512, 512))

        evaluation_result = [key]

        # Prepare mask
        mask = rle2mask(mask, (512, 512))
        mask = 1 - mask[:, :, np.newaxis]

        # Calculate each metric
        for metric in evaluation_df.columns.values.tolist()[1:]:
            print(f"  Calculating {metric}...", end=" ")

            try:
                if metric == 'Image Reward':
                    metric_result = metrics_calculator.calculate_image_reward(tgt_image, prompt)

                elif metric == 'HPS V2.1':
                    metric_result = metrics_calculator.calculate_hpsv21_score(tgt_image, prompt)

                elif metric == 'Aesthetic Score':
                    metric_result = metrics_calculator.calculate_aesthetic_score(tgt_image)

                elif metric == 'PSNR':
                    metric_result = metrics_calculator.calculate_psnr(src_image, tgt_image, mask)

                elif metric == 'LPIPS':
                    metric_result = metrics_calculator.calculate_lpips(src_image, tgt_image, mask)

                elif metric == 'MSE':
                    metric_result = metrics_calculator.calculate_mse(src_image, tgt_image, mask)

                elif metric == 'CLIP Similarity':
                    metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)

                print(f"{metric_result:.4f}")
                evaluation_result.append(metric_result)

            except Exception as e:
                print(f"ERROR: {e}")
                evaluation_result.append(None)

        evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

    # Save results
    print("\n" + "="*80)
    print("Evaluation completed!")
    print("="*80)

    print("\nThe averaged evaluation results:")
    averaged_results = evaluation_df.mean(numeric_only=True)
    print(averaged_results)

    # Save to CSV
    result_sum_path = os.path.join(args.image_save_path, "evaluation_result_sum.csv")
    result_detail_path = os.path.join(args.image_save_path, "evaluation_result.csv")

    averaged_results.to_csv(result_sum_path)
    evaluation_df.to_csv(result_detail_path, index=False)

    print(f"\nResults saved:")
    print(f"  - Summary: {result_sum_path}")
    print(f"  - Details: {result_detail_path}")
