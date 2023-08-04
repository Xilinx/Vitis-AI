import argparse
import os
import numpy as np
from PIL import Image


def compute_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between prediction and ground truth masks.
    """
    intersection = np.logical_and(prediction, ground_truth)
    union = np.logical_or(prediction, ground_truth)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def compute_pixel_accuracy(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Pixel-wise Accuracy between prediction and ground truth masks.
    """
    correct_pixels = np.sum(prediction == ground_truth)
    total_pixels = np.prod(prediction.shape)
    accuracy = correct_pixels / total_pixels
    return accuracy


def evaluate_images(prediction_path: str, ground_truth_path: str) -> tuple[float, float]:
    """
    Evaluate IoU and Pixel-wise Accuracy for two individual images.
    """
    prediction = np.array(Image.open(prediction_path))
    ground_truth = np.array(Image.open(ground_truth_path))

    iou = compute_iou(prediction, ground_truth)
    accuracy = compute_pixel_accuracy(prediction, ground_truth)

    return iou, accuracy


def evaluate_dataset(predictions_folder: str, ground_truth_folder: str) -> tuple[float, float]:
    """
    Evaluate mean IoU and mean Pixel-wise Accuracy for a dataset.
    """
    iou_list = []
    accuracy_list = []

    for prediction_file in os.listdir(predictions_folder):
        prediction_path = os.path.join(predictions_folder, prediction_file)
        ground_truth_path = os.path.join(ground_truth_folder, prediction_file)

        iou, accuracy = evaluate_images(prediction_path, ground_truth_path)
        iou_list.append(iou)
        accuracy_list.append(accuracy)

    mean_iou = np.mean(iou_list)
    mean_accuracy = np.mean(accuracy_list)

    return mean_iou, mean_accuracy


def evaluate_cityscapes(results_folder: str, cityscapes_folder: str) -> tuple[float, float]:
    """
    Evaluate mean IoU and mean Pixel-wise Accuracy for Cityscapes dataset.
    """
    predictions_folder = os.path.join(results_folder, 'predictions')
    ground_truth_folder = os.path.join(cityscapes_folder, 'gtFine')

    mean_iou, mean_accuracy = evaluate_dataset(predictions_folder, ground_truth_folder)

    return mean_iou, mean_accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model evaluation script')
    parser.add_argument('inference_result', help='Path to the inference result image')
    parser.add_argument('ground_truth', help='Path to the ground truth image')
    parser.add_argument('--dataset', action='store_true', help='Evaluate a dataset')
    parser.add_argument('--cityscapes', action='store_true', help='Evaluate Cityscapes dataset')
    args = parser.parse_args()

    if args.cityscapes:

        results_folder = args.inference_result
        cityscapes_folder = args.ground_truth

        mean_iou, mean_accuracy = evaluate_cityscapes(results_folder, cityscapes_folder)

        print(f"Mean IoU: {mean_iou}")
        print(f"Mean Pixel-wise Accuracy: {mean_accuracy}")
    else:
        if args.dataset:
            predictions_folder = args.inference_result
            ground_truth_folder = args.ground_truth

            mean_iou, mean_accuracy = evaluate_dataset(predictions_folder, ground_truth_folder)

            print(f"Mean IoU: {mean_iou}")
            print(f"Mean Pixel-wise Accuracy: {mean_accuracy}")
        else:
            inference_result_path = args.inference_result
            ground_truth_path = args.ground_truth

            iou, accuracy = evaluate_images(inference_result_path, ground_truth_path)

            print(f"IoU: {iou}")
            print(f"Pixel-wise Accuracy: {accuracy}")
