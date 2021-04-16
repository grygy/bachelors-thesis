import time

import numpy as np
from PIL import Image
import json
import copy
from json import JSONEncoder
import os
import os
import zipfile
import tempfile

# todo uncomment this on device and comment line bellow
# import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite


class EvaluatedModel:
    y_pred = []
    y_true = []
    model_name = ''
    avg_time = 0.0
    gzip_size = 0

    def __init__(self, dict1=None, y_pred=[], y_true=[], model_name='', avg_time=0.0, gzip_size=0):
        if (dict1 == None):
            self.y_pred = copy.deepcopy(y_pred)
            self.y_true = copy.deepcopy(y_true)
            self.model_name = model_name
            self.avg_time = avg_time
            self.gzip_size = gzip_size
        else:
            self.__dict__.update(dict1)


# subclass JSONEncoder
class EvaluatedModelEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)


def evaluate_model(model_path, dataset_path):
    """ returns EvaluatedModel for given model and dataset """
    interpreter = tflite.Interpreter(
        model_path=model_path, num_threads=None)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    if not floating_model:
        print('model has int i/o')

    input_index = input_details[0]["index"]

    y_true = []
    y_pred = []
    avg_time = 0.0
    
    first_invoke = True

    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file == '.DS_Store':
                continue

            y_true.append(int(subdir.split('/')[-1]))

            img = Image.open(os.path.join(subdir, file))

            # add batch dimension
            test_image = np.expand_dims(img, axis=0)

            if floating_model:
                test_image = (np.float32(test_image) / 255.0)

            interpreter.set_tensor(input_details[0]['index'], test_image)

            # ignore the 1st invoke (model optimization is happening)
            if first_invoke:
                interpreter.invoke()
                first_invoke = False

            startTime = time.time()
            interpreter.invoke()
            delta = (time.time() - startTime) * 1000
            avg_time += delta

            output = interpreter.get_tensor(output_details[0]['index'])
            # remove batch dimension and return predicted label
            y_pred.append(np.argmax(output[0]).item())

    gzip_size = get_gzipped_model_size(model_path)
    
    evaluated_model = EvaluatedModel(dict1=None, 
                                     y_pred=y_pred, 
                                     y_true=y_true, 
                                     model_name=model_path.split('/')[-1],
                                     avg_time=avg_time / len(y_pred), 
                                     gzip_size=gzip_size)
    print('avg time: ' + str(evaluated_model.avg_time) + 'ms, gzip: ' + str(evaluated_model.gzip_size) + 'Bytes\n')

    return evaluated_model


# todo change 
dataset_path = 'datasets/flowers_test/'

evaluated_models = []

for subdir, dirs, files in os.walk('flowers_models_optimized/'):
    for file in files:

        if file.split('.')[-1] == 'tflite':
            # check if model is in tflite
            print('evaluating: ' + file)
            evaluated_models.append(evaluate_model(os.path.join(subdir, file), dataset_path))


encoded_json = EvaluatedModelEncoder().encode(evaluated_models)
with open('evaluated_flowers_models.json', 'w') as f:
    f.write(encoded_json)
