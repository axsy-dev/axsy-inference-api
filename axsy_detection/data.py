import json
import os
from datetime import datetime

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


class Data_Manager:
    def __init__(
        self,
        path: str,
        dataset_id: str,
        input: str,
    ) -> None:
        """
        Initializes the Data_Manager instance.

        Args:
            path (str): Path to the axsy_notation_data directory
            dataset_id (str): The dataset/project in question
            input (str): Subpath to the input image or directory
        """
        self.path = path
        self.dataset_id = dataset_id
        self.img_exts = ["jpg", "png", "jpeg", "heic", "avif"]

        self.input_path = os.path.join(path, dataset_id, "images", input)
        self.model_path = os.path.join(path, dataset_id, "label-assist")
        self.output_path = os.path.join(path, dataset_id, "annotations")
        os.makedirs(self.output_path, exist_ok=True)

        self.annotations = []

    def load_yolo(
        self,
        model_name: str,
    ) -> YOLO:
        try:
            yolo = YOLO(f"{self.model_path}/{model_name}")
            print(f"Loaded Yolo model {model_name}")
            return yolo
        except BaseException as err:
            raise FileNotFoundError(f"Could not load file {model_name}\n", err)

    def load_data(self):
        images = []
        names = []

        # Check if the input_path is a file or a directory
        if os.path.isdir(self.input_path):
            # If it's a directory, load all images in the directory
            for filename in tqdm(
                os.listdir(self.input_path), desc=f"Loading directory {self.input_path}"
            ):
                if any(filename.lower().endswith(ext) for ext in self.img_exts):
                    img_path = os.path.join(self.input_path, filename)
                    try:
                        with Image.open(img_path) as img:
                            images.append(img.copy())
                            names.append(filename)
                    except IOError as err:
                        print(f"Could not load image {filename}\n", err)
        elif os.path.isfile(self.input_path):
            # If it's a single image file, load just that image
            if any(self.input_path.lower().endswith(ext) for ext in self.img_exts):
                try:
                    with Image.open(self.input_path) as img:
                        images.append(img.copy())
                        names.append(os.path.basename(self.input_path))
                except IOError as err:
                    print(f"Could not load image {self.input_path}\n", err)
            else:
                print(f"File {self.input_path} is not a valid image.")

        return images, names

    def save_annotations(
        self,
        annotations: list[str],
        f_name: str,
    ) -> None:
        """
        Saves annotation detections to a file and logs the action.

        Args:
            annotations (list[str]): List of detection strings to save.
            f_name (str): Name of the file to save the annotations to.
        """
        file_path = os.path.join(self.output_path, f_name)
        with open(file_path, "w") as f:
            f.write("\n".join(annotations))
        print(f"Saved annotation {file_path}")
        self.annotations.append(f_name)

    def save_logs(
        self,
    ) -> None:

        # Log the action to log.txt
        log_dir = os.path.join(self.model_path, "logs")
        dt = datetime.now()
        log_file_path = os.path.join(log_dir, f"log_{dt}.txt")
        log_line = f"Saved @@ to {self.output_path} at {dt}\n"
        log_entry = "".join(
            [log_line.replace("@@", fname) for fname in self.annotations]
        )

        os.makedirs(log_dir, exist_ok=True)
        try:
            with open(log_file_path, "w") as log_file:
                log_file.write(log_entry)
            print(f"Saved log to {log_file_path}")
        except Exception as e:
            print(f"Error writing log file: {e}")

    def save_classes(
        self,
        json_data: dict,
    ) -> None:
        json_path = os.path.join(self.output_path, "classes.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)
        print(f"\nSaved classes.json")
