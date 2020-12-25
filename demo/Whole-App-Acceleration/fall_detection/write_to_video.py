
import cv2
import os

from argparse import ArgumentParser

def main(input_dir, images_dir, output_dir, fps=30,
                    height=224, width=224, threshold=0.5):
    """
    Reads the probabilities by image filename or video frame id written by the
    accuracy kernel of graph_of_inference and puts the probability as text on
    the correspoding image/video frame and saves as video.

    Parameters:
    input_dir (str): Directory of the files where each file contains the
    probability of no-fall against each filename (if images) or
    frame id (if video).

    images_dir (str): Path to the directory which has the directory of image
    sequences or videos

    fps (int): Frames per seconds used in the output videos

    height (int): Height of the frames in output videos

    width (int): Width of the frames in output videso

    threshold (int): Threshold used to color code the probability.
    If probability less than threshold, the color of the text will be red,
    else green

    output_dir: Directory where to store the output videos

    """

    for file in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file)
        with open(file_path) as f:
            lines = sorted(f.readlines())
        lines = sorted(lines, key=lambda x: os.path.splitext(
            os.path.basename(x.split()[0]))[0])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        seq_name, ext = os.path.splitext(file)
        video = cv2.VideoWriter(
            os.path.join(output_dir, '{}.avi'.format(seq_name)),
            fourcc, fps, (width, height))
        seq_path = os.path.join(images_dir, seq_name)
        print("[INFO] Writing {}".format(seq_path))
        if os.path.isdir(seq_path):
            for line in lines:
                image, probability = line.split()
                image_path = os.path.join(seq_path, image)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (width, height))
                if float(probability) > threshold:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.putText(img, str(probability), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
                video.write(img)
        else:
            cap = cv2.VideoCapture(seq_path)
            f_count = -1
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break
                f_count += 1
                # First 10 frames are skipped as they are considered for
                # prediction
                if f_count < 10:
                    continue
                else:
                    frame_name, probability = lines[f_count-10].split()
                    frame = cv2.resize(frame, (width, height))
                    if float(probability) > threshold:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.putText(frame, str(probability), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    video.write(frame)
            cap.release()
        video.release()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-id", "--input-dir", required=True,
        help="Path to the directory which has the files containing the "
        "probabilities by the image file or video frame index")
    parser.add_argument(
        "-im", "--images-dir", required=True,
        help="Path to the directory which has the directory of image "
        "sequences or videos")
    parser.add_argument("-o", "--output-dir", default='./')
    parser.add_argument("-f", "--fps", default=30,
                        help="FPS of the output video")
    parser.add_argument("-w", "--width", default=224,
                        help="Width of the output video")
    parser.add_argument("-l", "--height", default=224,
                        help="Height of the output video")
    parser.add_argument("-t", "--threshold", default=0.5,
                        help="Threshold used to color code the probability")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.input_dir, args.images_dir, args.output_dir, args.fps,
         args.height, args.width, args.threshold)
