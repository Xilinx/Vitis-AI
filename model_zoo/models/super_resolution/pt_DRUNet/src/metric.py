import os
import argparse

import numpy as np
import cv2
from tqdm import tqdm


def get_image(image_path):
    image = cv2.imread(image_path)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return RGB_img


def resize_denoised_image(denoised_image, original_image):
    return cv2.resize(
        denoised_image,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_AREA
    )


def get_images_to_compare(original_image_path, denoised_image_path):
    original_image = get_image(original_image_path)
    denoised_image = get_image(denoised_image_path)
    if not original_image.shape == denoised_image.shape:
        denoised_image = resize_denoised_image(denoised_image, original_image)
    return original_image, denoised_image


def get_psnr(original_image_path, denoised_image_path):
    original_image, denoised_image = get_images_to_compare(
        original_image_path, denoised_image_path
    )
    return cv2.PSNR(original_image, denoised_image)


def get_psnr_all(dataset_folder, inference_folder):
    psnr_all = {}
    inference_folder_names = os.listdir(inference_folder)
    original_files_folder = os.path.join(dataset_folder, 'original_png')
    for noisy_folder in tqdm(inference_folder_names):
        psnr_all[noisy_folder] = []
        noisy_folder_path = os.path.join(inference_folder, noisy_folder)
        inference_file_names = os.listdir(noisy_folder_path)
        for inference_filename in sorted(inference_file_names):
            denoised_image_path = os.path.join(noisy_folder_path, inference_filename)
            original_filename = inference_filename.split('_')[0] + '.png'
            original_image_path = os.path.join(original_files_folder, original_filename)
            noisy_image_path = os.path.join(dataset_folder, noisy_folder, original_filename)
            psnr_all[noisy_folder].append({
                'original_image_path': original_image_path,
                'noisy_image_path': noisy_image_path,
                'denoised_image_path': denoised_image_path,
                'psnr_denoised': get_psnr(original_image_path, denoised_image_path),
                'psnr_noisy': get_psnr(original_image_path, noisy_image_path),
            })
    return psnr_all


def get_psnr_mean(psnr_all):
    psnr_mean = {}
    for noisy_sigma in psnr_all:
        psnrs = [e['psnr_denoised'] for e in psnr_all[noisy_sigma]]
        psnr_mean[noisy_sigma] = np.mean(psnrs)
    return psnr_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model evaluation script')

    parser.add_argument('dataset_folder', help='The folder where original dataset is stored.')
    parser.add_argument('inference_folder', help='The folder where results of model inference is stored.')

    args = parser.parse_args()

    psnr_all = get_psnr_all(args.dataset_folder, args.inference_folder)
    psnr_mean = get_psnr_mean(psnr_all)
    print(f"PSNR all:\n{psnr_all}")
    print(f"PSNR mean:\n{psnr_mean}")

