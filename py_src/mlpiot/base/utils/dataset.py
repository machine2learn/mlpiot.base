from glob import glob
import mimetypes
from os.path import join
import pathlib
from typing import Dict
from xml.etree import ElementTree

from mlpiot.proto import VisionPipelineData, VisionPipelineDataset


class DatasetParams:
    def __init__(self,
                 keep_in_memory: bool = False,
                 embed_image: bool = True):
        self.keep_in_memory = keep_in_memory
        self.embed_image = embed_image


class DatasetFromPascalVoc(VisionPipelineDataset):
    def __init__(self,
                 directory_path: str,
                 dataset_params: DatasetParams):
        self.xml_files = glob(join(directory_path, "*.xml"))
        self.directory_path = directory_path
        self.dataset_params = dataset_params
        self._cache = {}  # type: Dict[int, VisionPipelineData]
        mimetypes.init()

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, params):
        if params in self._cache:
            return self._cache[params]

        xml_file = self.xml_files[params]
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        vision_pipeline_data = VisionPipelineData()

        img_filename = root.find('filename').text
        if img_filename:
            absolute_path_string = join(self.directory_path, img_filename)
        else:
            absolute_path_string = root.find('path').text
        img_size = root.find('size')
        vision_pipeline_data.input_image.height = \
            int(img_size.find("height").text)
        vision_pipeline_data.input_image.width = \
            int(img_size.find("width").text)
        vision_pipeline_data.input_image.channels = \
            int(img_size.find("depth").text)

        img_type, _ = mimetypes.guess_type(img_filename)
        if img_type:
            vision_pipeline_data.input_image.format = img_type

        if self.dataset_params.embed_image:
            with open(absolute_path_string, 'rb') as img_file:
                vision_pipeline_data.input_image.data = img_file.read()
        else:
            vision_pipeline_data.input_image.url = pathlib.Path(absolute_path_string).as_uri()

        for xml_obj in root.iter('object'):
            obj = vision_pipeline_data.scene_description.objects.add()
            obj.class_name = xml_obj.find("name").text
            for box in xml_obj.findall("bndbox"):
                v = [obj.bounding_box.vertices.add() for i in range(5)]
                v[0].x = v[1].x = v[4].x = int(box.find("xmin").text)
                v[0].y = v[3].y = v[4].y = int(box.find("ymin").text)
                v[3].x = v[2].x = int(box.find("xmax").text)
                v[1].y = v[2].y = int(box.find("ymax").text)
                # TODO: Merge the boxes in a better way

        if self.dataset_params.keep_in_memory:
            self._cache[params] = vision_pipeline_data

        return vision_pipeline_data


def auto_detect_dataset(
        directory_path: str,
        dataset_params: DatasetParams) -> VisionPipelineDataset:
    # TODO
    return DatasetFromPascalVoc(directory_path, dataset_params)
