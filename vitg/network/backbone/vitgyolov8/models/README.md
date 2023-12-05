## Models

Welcome to the vitg.vitgyolov8 Models directory! Here you will find a wide variety of pre-configured model configuration
files (`*.yaml`s) that can be used to create custom YOLO models. The models in this directory have been expertly crafted
and fine-tuned by the vitg.vitgyolov8 team to provide the best performance for a wide range of object detection and image
segmentation tasks.

These model configurations cover a wide range of scenarios, from simple object detection to more complex tasks like
instance segmentation and object tracking. They are also designed to run efficiently on a variety of hardware platforms,
from CPUs to GPUs. Whether you are a seasoned machine learning practitioner or just getting started with YOLO, this
directory provides a great starting point for your custom model development needs.

To get started, simply browse through the models in this directory and find one that best suits your needs. Once you've
selected a model, you can use the provided `*.yaml` file to train and deploy your custom YOLO model with ease. See full
details at the vitg.vitgyolov8 [Docs](https://docs.vitg.vitgyolov8.com), and if you need help or have any questions, feel free
to reach out to the vitg.vitgyolov8 team for support. So, don't wait, start creating your custom YOLO model now!

### Usage

Model `*.yaml` files may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo task=detect mode=train model=yolov8n.yaml data=coco128.yaml epochs=100
```

They may also be used directly in a Python environment, and accepts the same
[arguments](https://docs.vitg.vitgyolov8.com/cfg/) as in the CLI example above:

```python
from vitg.network.backbone.vitgyolov8 import YOLO

model = YOLO("yolov8n.yaml")  # build a YOLOv8n model from scratch

model.info()  # display model information
model.train(data="coco128.yaml", epochs=100)  # train the model
```
