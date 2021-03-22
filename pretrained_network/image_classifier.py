from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# Models to play around with:
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

def image_to_array(image_path='/images/aquarium.jpg', 
                   image_width=996, 
                   add_border=True):
  """
  Given an image_path resizes the image to image_width, 
  adds a border to the image if add_border=True,
  and processes the image to a Numpy array
  """
  basewidth = image_width
  img = Image.open(image_path)
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  if add_border == True:
    img = ImageOps.expand(img, border=50, fill='grey')
  img_array = image.img_to_array(img)
  return img_array

def image_tile_slicer(img_array=image_to_array(), steps_for_x_frames=14, steps_for_y_frames=14):
  """
  Given an img_array slices the img_array in 224 by 224 tiles,
  with the given steps_for_x_frames and the given steps_for_y_frames
  """
  tiles = []
  for x in range(0, 1920, steps_for_x_frames): 
      for y in range(0, 1080, steps_for_y_frames): 
          tile = img_array[x:x+224, y:y+224, :]
          if tile.shape == (224, 224, 3):  
              tiles.append(tile)
  return tiles

def make_predictions(chosen_model=MobileNetV2,
                     tiles=image_tile_slicer()):
  """
  Spots an Anemoe fish on the picture,
  given a pretrained chosen_model (e.g. MobileNetV2, ResNet50 or VGG16), 
  Numpy arrays from the 224 by 224 tiles created from the original picture,
  and an accuracy_threshold (default is 90%) for the prediction.
  """
  model = chosen_model(weights='imagenet', include_top=True)
  model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
  tiles = np.array(tiles)
  predictions = decode_predictions(model.predict(tiles))
  return predictions

def nemo_classifier(predictions=make_predictions(),
                    accuracy_threshold=0.9):
  """
  Spots an Anemoe fish on the Numpy arrays 
  from the 224 by 224 tiles created from the original picture,
  with the selected accuracy_threshold (default is 90%) for the prediction.
  """
  prediction_nr = 0
  nemo_count = 0
  confidence_count = 0
  confident_nemo_count = 0

  for i in predictions:
    if predictions[prediction_nr][0][1] == 'anemone_fish':
      nemo_count +=1
      if predictions[prediction_nr][0][2] > accuracy_threshold:
        confident_nemo_count +=1
        print('Anemoe fish found in the frame nr.: '+str(prediction_nr))
        print()
    if predictions[prediction_nr][0][2] > accuracy_threshold:
      confidence_count +=1
    prediction_nr += 1

  if confident_nemo_count == 0:
    print("No Anemoe fish was found with the given accuracy threshold of "+str(accuracy_threshold)+'\n')

  print('=========================================================\n')
  print('Total number of 224x224 frames: '+str(len(predictions)))
  print('Frames with Nemos: '+str(nemo_count))
  print('Frames with accuracy threshold over '+str(accuracy_threshold)+': '+str(confidence_count))
  print('Frames with accuracy of Nemo over '+str(accuracy_threshold)+': '+str(confident_nemo_count)+'\n')
  print('=========================================================\n')

def show_top_predictions(predictions=make_predictions(),
                         accuracy_threshold=0.9):
  """
  Shows top two predictions for every 224 by 224 tile
  created from the original picture,
  with the selected accuracy_threshold (default is 90%) for the prediction.
  """
  prediction_nr = 0

  for i in predictions:
    if predictions[prediction_nr][0][2] > accuracy_threshold:
      print('Prediction for frame nr.: ' + str(prediction_nr))
      print('Best guess is: '+str(predictions[prediction_nr][0][2])+str(' ')+str(predictions[prediction_nr][0][1]))
      print('Second best guess is: '+str(predictions[prediction_nr][1][2])+str(' ')+str(predictions[prediction_nr][1][1]))
      print('---\n')
    prediction_nr += 1