import argparse
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import json

def main(args):

    records = []
    for path in tqdm(args.paths, desc="Loading loss data"):
        print(path)
        data = torch.load(path, weights_only=False)
        for res in data.keys():
            for image_id in data[res].keys():
                if any([image_id.startswith(prefix) for prefix in args.exclude_prefix]):
                    continue
                for timestep, loss in data[res][image_id]:
                    records.append({
                        'res': res,
                        'image_id': image_id,
                        'timestep': timestep,
                        'loss': loss
                    })
    print(len(records), "data points loaded")
    df = pd.DataFrame(records)
    mean_loss_per_timestep = df.groupby('timestep')['loss'].mean()
    df = df.merge(mean_loss_per_timestep.rename('mean_loss'), on='timestep')


    def plot_means(df):
        # scatter plot of timestep vs mean loss
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.scatter(mean_loss_per_timestep.index, mean_loss_per_timestep.values, alpha=0.5)
        plt.xlabel('Timestep')
        plt.ylabel('Mean Loss')

    # find outlier images
    # outlier images are images where their loss is much higher than the mean loss for that timestep
    df['loss_diff'] = df['loss'] - df['mean_loss']
    df['log_loss_diff_proportional'] = np.log(df['loss'] / df['mean_loss'])

    # group by image_id and count how many outlier timesteps each image has; also sum the loss_diff for each image
    outlier_summary = df.groupby('image_id').agg({
        'timestep': 'count',
        'log_loss_diff_proportional': 'mean',
    }).rename(
        columns={'timestep': 'outlier_timestep_count', 'log_loss_diff_proportional': 'log_loss_diff_proportional_mean'})
    outlier_summary = outlier_summary[outlier_summary.outlier_timestep_count > args.sample_count_thresh]
    print(f"After thresholding by sample count {args.sample_count_thresh}, {len(outlier_summary)} images will get multipliers")

    # multiplier is just the exponential of the log_loss_diff_proportional_mean, but we clip it to be between min_multiplier and max_multiplier
    log_min_multiplier = np.log(args.min_multiplier)
    log_max_multiplier = np.log(args.max_multiplier)
    log_multiplier = (outlier_summary['log_loss_diff_proportional_mean'] * args.multiplier_expand_factor).clip(lower=log_min_multiplier, upper=log_max_multiplier)
    outlier_summary['multiplier'] = np.exp(log_multiplier)
    print('Overall multiplier:', outlier_summary['multiplier'].mean(), 'min:', outlier_summary['multiplier'].min(), 'max:', outlier_summary['multiplier'].max())

    # output
    per_path_multiplier = outlier_summary.to_dict()['multiplier']
    with open(args.output_json, 'w') as f:
        json.dump(per_path_multiplier, f)

    print("Top 20 images by multiplier:")
    print("\n".join([f"{row['multiplier']:2.4}\t{index}" for index, row in outlier_summary.sort_values('multiplier', ascending=False).head(20).iterrows()]))

    print(f'Saved multiplier info for {len(per_path_multiplier)} images to {args.output_json}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a JSON file containing multiplier info for each image and timestep based on loss values. This can be used to apply dynamic loss scaling during training, where samples with higher losses receive higher multipliers to focus training on harder examples.\n\nBy default, any sample with a loss above the mean will receive a multiplier above 1, and samples with losses below the mean will receive a multiplier of 1. You can adjust the std_dev_cutoff to control when to start increasing multiplier. To reduce the appearance of well-trained images, reduce min_multiplier to <1. Eg, to reduce appearance of *all* images that have low losses compared to means down to a minimum of 25%, set --min_multiplier to 0.25")
    parser.add_argument('paths', nargs='+', type=str, help='Paths to loss_per_image_and_timestep.pt files')
    parser.add_argument('--output_json', required=True, type=str, help='Path to output JSON multiplier info file')
    parser.add_argument('--sample_count_thresh', default=3, type=int, help='Minimum umber of samples required to produce a multiplier for a given image.')
    parser.add_argument('--max_multiplier', default=4.0, type=float, help='Maximum multiplier to apply')
    parser.add_argument('--min_multiplier', default=1, type=float, help='Minimum multiplier to apply')
    parser.add_argument('--multiplier_expand_factor', default=1.0, type=float, help='Factor to expand the multiplier range. Multiplier will be raised to the power of this factor, so values >1 will increase the gap between high and low multipliers, while values <1 will decrease the gap.')
    parser.add_argument("--exclude_prefix", nargs='+', default=[], help="Exclude images whose image_id starts with any of these prefixes from multiplier calculation.")

    args = parser.parse_args()
    main(args)
