from tensorflow.keras.models import model_from_json


# SERIALIZE_MODEL ------------------------------------------
def serialize_model( model, model_name, path ):
  # serialize model to JSON
  model_json = model.to_json()
  with open(path + model_name+"_DUO.json", "w") as json_file:
    json_file.write(model_json)
    
  # serialize weights to HDF5
  model.save_weights( path + model_name+"_DUO.h5")
  print("Model saved to disk") 

# LOAD_MODEL ------------------------------------------
def load_model( model_name, path ):
  json_file = open( path + model_name +  "_DUO.json", 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  # load weights into new model
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights(path + model_name +  "_DUO.h5")
  print("Loaded model from disk")
  return loaded_model