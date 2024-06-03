import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("GPU is", "available" if tf.config.experimental.list_physical_devices('GPU') else "NOT AVAILABLE")

# GPU check
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is not using the GPU.")