import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('spice') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)


  #### Other features
## Apply optimizations. A common optimization used is post training quantization, which can further reduce your model latency and size with minimal loss in accuracy.

## Add metadata, which makes it easier to create platform specific wrapper code when deploying models on devices.