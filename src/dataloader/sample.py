from ..configuration.constants import CLASS_MAPPING, NUM_CLASSES

import os
from os import path
from PIL import Image
import numpy as np
import json
import shutil

class Sample:
    def __init__(self, input_path, output_path, positive=True):
        self.input_path = input_path
        file_number = int(path.splitext(path.basename(input_path))[0].rsplit("_", 2)[-1])
        assert str(file_number) in output_path
        self.output_path = output_path
        self.is_positive_sample = positive

    def load(self):
        input = np.array(Image.open(self.input_path), dtype=np.float32) / 255.0
        output = self.load_output()
        
        return input, output

    def move_to(self, replacer):
        new_image_path = replacer(self.input_path)
        new_mask_path  = replacer(self.output_path)
        new_json_path  = path.splitext(new_mask_path)[0] + ".json"

        os.makedirs(path.dirname(new_image_path), exist_ok=True)
        os.makedirs(path.dirname(new_mask_path), exist_ok=True)
        shutil.copy(self.input_path, new_image_path)
        shutil.copy(self.output_path, new_mask_path)
        shutil.copy(path.splitext(self.output_path)[0] + ".json", new_json_path)

    def load_output(self):
        if self.is_positive_sample:
            output = np.array(Image.open(self.output_path))
            output = self.apply_mapping(output, CLASS_MAPPING)
        else:
            input = np.array(Image.open(self.input_path))
            output = np.zeros_like(input, dtype=np.uint8)[:,:,0]

        return output
        
    def load_this_mapping(self):
        json_path = path.splitext(self.output_path)[0] + ".json"
        with open(json_path, "r") as file:
            return json.load(file)

    def apply_mapping(self, label_image: np.array, target_mapping):
        current_mapping = self.load_this_mapping()
        output_labels = np.zeros_like(label_image)

        if "field" in current_mapping:
            for key, assigned_values in current_mapping["field"].items():
                assert key in target_mapping["field"]
                target_value = target_mapping["field"][key]
                for current_value in assigned_values:
                    output_labels[label_image == current_value] = target_value
        
        if "robots" in current_mapping:
            for robot in current_mapping["robots"]:
                for key, current_value in robot.items():
                    if key in target_mapping["robots"]:
                        target_value = target_mapping["robots"][key]
                        output_labels[label_image == current_value] = target_value

        # In the .json, the circle field is not included... It has value 20
        output_labels[label_image == 20] = target_mapping["field"]["circle"]
        
        assert np.all(output_labels < NUM_CLASSES)
        return output_labels