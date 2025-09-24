import numpy as np
import torch
import os
from .data import Data_Manager


class Inferencer:
    def __init__(
        self,
        project_args,
        detector,
        run_inference,
    ):
        self.data_manager = Data_Manager(**project_args)
        self.detector = self.data_manager.load_yolo(detector)
        self.run_inference = run_inference

    def __call__(
        self,
    ) -> None:

        if not self.run_inference:
            return

        if not (
            os.path.isdir(self.data_manager.input_path)
            or os.path.isfile(self.data_manager.input_path)
        ):
            print(
                f"{self.data_manager.input_path} is neither a valid directory nor a file."
            )
            return

        images, f_names = self.data_manager.load_data()
        detections = self.detect(images)

        cls_labels = self.detector.names

        for ix, (detection, name) in enumerate(zip(detections, f_names)):

            Y, X = detection.orig_shape
            annotations = []

            for cls_id, lab in cls_labels.items():

                def get_row(box):
                    row = [
                        cls_id,
                        np.around(box[0] / X, 4),
                        np.around(box[1] / Y, 4),
                        np.around(box[2] / X, 4),
                        np.around(box[3] / Y, 4),
                    ]
                    return " ".join([str(item) for item in row])

                instances = detection.boxes.xywh[
                    torch.where(detection.boxes.cls == cls_id)
                ]
                annotations += [get_row(box) for box in instances.tolist()]

            f_name = ".".join(name.split(".")[:-1]) + ".txt"
            self.data_manager.save_annotations(annotations, f_name)
        self.data_manager.save_classes(cls_labels)
        self.data_manager.save_logs()

    def detect(
        self,
        images,
    ):
        print(f"starting detection with {len(images)} images")
        detections = self.detector(images)
        return detections
