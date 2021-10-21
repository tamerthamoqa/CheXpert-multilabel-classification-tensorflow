import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', type=str, required=True,
                    help="Path to input dataset image folder"
                    )
parser.add_argument('--output_directory', type=str, required=True,
                    help="Path to output dataset image folder"
                    )
parser.add_argument('--image_size', type=int, required=True,
    help='Required squared image size (image_size, image_size) each image in the dataset image folder would be resized to with keeping the original aspect ratio intact'
                    )

args = parser.parse_args()
input_directory = args.input_directory
output_directory = args.output_directory
image_size = args.image_size


def resize_image_keep_aspect_ratio(image_path, desired_image_size):
    """This resizing method was taken from blog post by jdhao:
    https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/#resize-and-pad-with-image-module

    Creates a blank square image, then pastes the resized image onto the blank square image to form a new image with
    the aspect ratio intact.
    """

    image = Image.open(image_path)
    old_image_size = image.size

    ratio = float(desired_image_size) / max(old_image_size)
    new_image_size = tuple([int(x * ratio) for x in old_image_size])

    image = image.resize(new_image_size, Image.ANTIALIAS)

    new_image = Image.new("RGB", (desired_image_size, desired_image_size))
    new_image.paste(
        image,
        ((desired_image_size - new_image_size[0]) // 2, (desired_image_size - new_image_size[1]) // 2)
    )
    del image

    return new_image


def resize_image_folder_keep_aspect_ratio(input_directory, output_directory, image_size):
    """Loops through all images in Image folder, resizes them to specified image_size with keeping aspect ratio
     intact."""

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for (root, dirs, files) in os.walk(input_directory, topdown=True):
        if len(files) != 0:
            for image_filename in files:
                image_path = os.path.join(root, image_filename)
                resized_image = resize_image_keep_aspect_ratio(image_path=image_path, desired_image_size=image_size)

                # Save as .png format to have higher quality image
                resized_image_filename = image_filename.split(".")[0] + ".png"

                # Set output directory parent folders for resized image
                splitted_root_path = root.split("/")
                train_valid_folder_name = splitted_root_path[-3]
                patient_folder_name = splitted_root_path[-2]
                study_folder_name = splitted_root_path[-1]
                resized_image_directory_name = train_valid_folder_name + "/" + patient_folder_name + "/" + study_folder_name

                if not os.path.exists(os.path.join(output_directory, resized_image_directory_name)):
                    os.makedirs(os.path.join(output_directory, resized_image_directory_name))

                # Save image
                resized_image_path = os.path.join(output_directory, resized_image_directory_name, resized_image_filename)
                resized_image.save(resized_image_path)
                print(resized_image_path)

    print("\nProcess done.")


if __name__ == '__main__':
    resize_image_folder_keep_aspect_ratio(
        input_directory=input_directory,
        output_directory=output_directory,
        image_size=image_size
    )
