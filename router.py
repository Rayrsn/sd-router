from pprint import pprint

import torch
from flatten_dict import unflatten
from flatten_dict.splitters import dot_splitter
import json
model_path = 'D:\stable-diffusion-webui\models\Stable-diffusion'
model_name = 'sd-v1-4.ckpt'
#model_name = '768-v-ema.ckpt'

description = dict()

# function to load model into variable
def load_model(model_name):
  model = torch.load(model_path + '\\' + model_name, map_location='cpu')
  return model


def get_type(m_key):
  if type(m_key) == int:
    description[key] = m_key
  elif type(m_key) == dict:
    description[key] = get_sub(model[key])
  elif type(m_key) == str:
    description[key] = m_key
  elif type(m_key) == torch.Tensor:
    description[key] = m_key.shape
  elif type(m_key) == list:
    description[key] = []
  else:
    description[key] = 'Unknown'

def get_sub(model_dict):
  value = {}
  value_nondot = {}
  for key in model_dict:
    if type(model_dict[key]) == torch.Tensor:
      if '.' in key:
        value[key] = model_dict[key].shape
      elif '.' not in key:
        value_nondot[key] = model_dict[key].shape
      value = unflatten(value, splitter=dot_splitter)
      value.update(value_nondot)
    elif type(model_dict[key]) == dict:
      print(model_dict[key].keys())
      for keys in model_dict[key]:
        value.update(get_type(model_dict[keys]))
    else:
      print(f'{key} {type(model_dict[key])}')

  return value


# get information about model
model = load_model(model_name)

# for each model.keys print the keys
for key in model.keys():  # if type is int, print the value
  get_type(model[key])

# write pretty model to json file
with open('model.json', 'w') as f:
  json.dump(description, f, indent=2)




# json encode description and pretty print
print(json.dumps(description, indent=4))
