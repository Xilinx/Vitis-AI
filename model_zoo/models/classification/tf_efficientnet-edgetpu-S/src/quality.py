import argparse
import os
import numpy as np
from sklearn.metrics import top_k_accuracy_score


def calculate_metric_on_single_image(predictions, groundtruth):
    top1_accuracy = top_k_accuracy_score([groundtruth], predictions, k=1)
    top5_accuracy = top_k_accuracy_score([groundtruth], predictions, k=5)
    return top1_accuracy, top5_accuracy


def calculate_metric_on_batch(images_folder, results_folder):
    image_files = sorted(os.listdir(images_folder))
    result_files = sorted(os.listdir(results_folder))
    if len(image_files) != len(result_files):
        raise ValueError("Number of images and results files do not match")

    top1_accuracies = []
    top5_accuracies = []
    for image_file, result_file in zip(image_files, result_files):
        result_path = os.path.join(results_folder, result_file)
        with open(result_path, 'r') as file:
            predictions = file.read().splitlines()
            predictions = [int(pred) for pred in predictions]
        groundtruth = int(image_file.split('.')[0])  # Assuming image filename corresponds to groundtruth label
        top1_accuracy, top5_accuracy = calculate_metric_on_single_image(predictions, groundtruth)
        top1_accuracies.append(top1_accuracy)
        top5_accuracies.append(top5_accuracy)

    mean_top1_accuracy = np.mean(top1_accuracies)
    mean_top5_accuracy = np.mean(top5_accuracies)
    return mean_top1_accuracy, mean_top5_accuracy


def calculate_metric_on_dataset(dataset_path):
    images_folder = os.path.join(dataset_path, 'images')
    results_folder = os.path.join(dataset_path, 'results')
    mean_top1_accuracy, mean_top5_accuracy = calculate_metric_on_batch(images_folder, results_folder)
    return mean_top1_accuracy, mean_top5_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate top-1 and top-5 accuracy metrics.')
    parser.add_argument('image', nargs='?', help='Path to the image inference result (single mode)')
    parser.add_argument('groundtruth', nargs='?', type=int, help='Groundtruth label for the image (single mode)')
    parser.add_argument('--batch', action='store_true', help='Calculate metrics on a batch of images')
    parser.add_argument('--dataset', help='Path to the ImageNet dataset')
    args = parser.parse_args()

    if args.batch:
        mean_top1_accuracy, mean_top5_accuracy = calculate_metric_on_batch(args.image, args.groundtruth)
        print(f"Mean top-1 accuracy on the batch: {mean_top1_accuracy}")
        print(f"Mean top-5 accuracy on the batch: {mean_top5_accuracy}")
    elif args.dataset:
        mean_top1_accuracy, mean_top5_accuracy = calculate_metric_on_dataset(args.dataset)
        print(f"Mean top-1 accuracy on the dataset: {mean_top1_accuracy}")
        print(f"Mean top-5 accuracy on the batch: {mean_top5_accuracy}")
    else:
        top1_accuracy, top5_accuracy = calculate_metric_on_single_image(args.image, args.groundtruth)
        print(f"Top-1 accuracy: {top1_accuracy}")
        print(f"Top-5 accuracy: {top5_accuracy}")

