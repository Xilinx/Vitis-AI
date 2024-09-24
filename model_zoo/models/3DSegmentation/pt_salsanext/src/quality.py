import numpy as np
import nibabel as nib
import sys

def load_nifti_file(filepath):
    """
    Load 3D NIfTI file and return the data as a NumPy array.

    Parameters:
        filepath (str): Path to the NIfTI file.

    Returns:
        np.ndarray: 3D NumPy array containing the image data.
    """
    nifti_data = nib.load(filepath)
    return nifti_data.get_fdata()

def calculate_iou(segmented_data, ground_truth_data):
    """
    Calculate the Intersection over Union (IoU) score for 3D segmentation evaluation.

    Parameters:
        segmented_data (np.ndarray): 3D NumPy array representing the segmented data.
        ground_truth_data (np.ndarray): 3D NumPy array representing the ground truth data.

    Returns:
        float: Intersection over Union (IoU) score.
    """
    intersection = np.logical_and(segmented_data, ground_truth_data).sum()
    union = np.logical_or(segmented_data, ground_truth_data).sum()
    iou = intersection / union
    return iou

def main():
    """
    Main function to evaluate 3D segmentation using IoU.

    Usage: python evaluate_segmentation.py <segmented_filepath> <groundtruth_filepath>

    The script calculates the Intersection over Union (IoU) score for 3D segmentation evaluation
    using the provided segmented file and ground truth file as arguments.
    """
    if len(sys.argv) != 3:
        print("Usage: python evaluate_segmentation.py <segmented_filepath> <groundtruth_filepath>")
        sys.exit(1)

    segmented_filepath = sys.argv[1]
    groundtruth_filepath = sys.argv[2]

    try:
        segmented_data = load_nifti_file(segmented_filepath)
        ground_truth_data = load_nifti_file(groundtruth_filepath)

        if segmented_data.shape != ground_truth_data.shape:
            raise ValueError("Segmented and ground truth data have different shapes.")

        iou = calculate_iou(segmented_data, ground_truth_data)
        print("Intersection over Union (IoU) score:", iou)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
