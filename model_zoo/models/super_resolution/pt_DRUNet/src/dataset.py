import os
import argparse


def prepare_dataset(dataset_folder: str) -> None:
    """
    Function that creates list file for each sub-folder in dataset folder.
    :param dataset_folder: Path where the whole dataset is stored.
    """
    noisy_folders = [
        p for p in os.listdir(dataset_folder) if 'noisy' in p and
                                                 os.path.isdir(os.path.join(dataset_folder, p))
    ]

    for nfolder in noisy_folders:
        nfolder_path = os.path.join(dataset_folder, nfolder)

        list_filepath = os.path.join(dataset_folder, f'{nfolder}.list')
        if os.path.exists(list_filepath):
            os.remove(list_filepath)

        filenames = sorted(os.listdir(nfolder_path))

        for filename in filenames:
            with open(list_filepath, 'a') as f:
                f.write(f'{nfolder}/{filename}\n')

        print(f"List file {list_filepath} is created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preparation script.')

    parser.add_argument('dataset_folder', help='The folder where dataset is stored.')
    args = parser.parse_args()

    prepare_dataset(args.dataset_folder)
