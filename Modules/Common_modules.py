import os
from glob import glob

class workingPath(list):
  # setup working paths:  (for Windows: \\)
  working_path = './'
  home_path = "/home/schen/Desktop/"
  # setup model paths:
  model_path = os.path.join(working_path, 'Models/')
  best_model_path = os.path.join(model_path, 'Best_Models/')

  # setup training paths:
  trainingSet_path = os.path.join(working_path, 'trainingSet/')
  originTrainingSet_path = os.path.join(trainingSet_path, 'originSet/')
  maskTrainingSet_path = os.path.join(trainingSet_path, 'maskSet/')
  aortaTrainingSet_path = os.path.join(maskTrainingSet_path, 'Aorta/')
  pulTrainingSet_path = os.path.join(maskTrainingSet_path, 'Pul/')
  training3DSet_path = os.path.join(working_path, 'trainingSet/3D/')
  trainingAugSet_path = os.path.join(trainingSet_path, 'originAugSet/')

  # setup validation paths:
  validationSet_path = os.path.join(working_path, 'validationSet/')
  originValidationSet_path = os.path.join(validationSet_path, 'originSet/')
  maskValidationSet_path = os.path.join(validationSet_path, 'maskSet/')
  aortaValidationSet_path = os.path.join(maskValidationSet_path, 'Aorta/')
  pulValidationSet_path = os.path.join(maskValidationSet_path, 'Pul/')

  # setup testing paths:
  testingSet_path = os.path.join(working_path, 'testingSet/')
  originTestingSet_path = os.path.join(testingSet_path, 'originSet/')
  maskTestingSet_path = os.path.join(testingSet_path, 'maskSet/')
  aortaTestingSet_path = os.path.join(maskTestingSet_path, 'Aorta/')
  pulTestingSet_path = os.path.join(maskTestingSet_path, 'Pul/')

def mkdir(path):
  import os

  path = path.strip()
  path = path.rstrip("\\")

  isExists = os.path.exists(path)

  if not isExists:
    os.makedirs(path)
    return True
  else:
    return False