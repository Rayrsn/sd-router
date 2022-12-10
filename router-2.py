from pprint import pprint
from munch import DefaultMunch
import torch
from flatten_dict import unflatten
from flatten_dict.splitters import dot_splitter
import json
model_path = 'D:\stable-diffusion-webui\models\Stable-diffusion'
model_name = 'sd-v1-4.ckpt'
#model_name = '768-v-ema.ckpt'

# function to load model into variable


def load_model(model_name):
    model = torch.load(model_path + '\\' + model_name, map_location='cpu')
    return model


def replace_tensor_with_shape(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = (replace_tensor_with_shape(value))
            # replace_tensor_with_shape(value)
        elif isinstance(value, torch.Tensor):
            d[key] = str(value.shape)
        elif isinstance(value, list):
            d[key] = str(value)
        else:
            # print(str(type(d[key])) + ' ' + str(value))
            d[key] = str(value)
    return d


# convert to obj
def to_obj(model):
    obj = {}
    for key, value in model.items():
        if isinstance(key, type):
            key = str(key)
        if isinstance(value, torch.Tensor):
            obj[key] = value.shape
        elif isinstance(value, dict):
            obj[key] = to_obj(value)
        else:
            obj[key] = value
    return obj


model = replace_tensor_with_shape(load_model(model_name))


# write json
model_obj = to_obj(model)
model_json = json.dumps(model_obj, indent=4)
with open('model.json', 'w') as f:
    f.write(model_json)

# print(model_json)

# check if model can be converted to json, print detailed error
# try:
#   print(obj["state_dict"])
# except TypeError as e:
#   print(e)
#   print('Model cannot be converted to json, please check the model')
#   print(obj.keys())
#
# else:
#   print('Model can be converted to json')
#
#
#

