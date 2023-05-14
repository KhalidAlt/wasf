# coding=utf-8
# Copyright 2022 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import io

import datasets
from datasets.tasks import ImageClassification
from PIL import Image

from classes import IMAGENET2012_CLASSES


_CITATION = """\
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
"""

_HOMEPAGE = "https://image-net.org/index.php"

_DESCRIPTION = """\
ILSVRC 2012, commonly known as 'ImageNet' is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet aims to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, ImageNet hopes to offer tens of millions of cleanly sorted images for most of the concepts in the WordNet hierarchy. ImageNet 2012 is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images
"""

_DATA_Path = {
    "val": [
        glob.glob("/media/khalid/data_disk/Dataset/Image/imagenet/val_images/*")
    ],
}


def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a fake file stored in memory
    imgByteArr = io.BytesIO()
    # image.save expects a file as a argument, passing a bytes io ins
    image.save(imgByteArr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


class Imagenet1k(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        assert len(IMAGENET2012_CLASSES) == 1000
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=list(IMAGENET2012_CLASSES.values())),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[ImageClassification(image_column="image", label_column="label")],
        )

    def _split_generators(self,dl_manager):
        """Returns SplitGenerators."""
        return [

            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "paths": i  for i in _DATA_Path['val'] } ,
            ) ,
        ]

    def _generate_examples(self, paths):
        """Yields examples."""
        idx = 0
        for path in paths:
            if path.endswith(".JPEG"):
                # image filepath format: <IMAGE_FILENAME>_<SYNSET_ID>.JPEG
                #print("###################################")
                #print(idx)                
                root, _ = os.path.splitext(path)
                _, synset_id = os.path.basename(root).rsplit("_", 1)
                label = IMAGENET2012_CLASSES[synset_id]
                image = Image.open(path)
                #ex = {"image": {"path": path, "bytes": image}, "label": label}
                #print(ex)
                yield  idx, { 'image':image, 'label':label}
                idx += 1
