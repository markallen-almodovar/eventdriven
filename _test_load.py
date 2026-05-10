import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

print("Testing gender model...")
try:
    m = tf.keras.models.load_model('saved_model/gender_classification_model.keras', compile=False)
    print("Gender OK:", m.input_shape)
except Exception as e:
    print("Gender ERROR:", e)

print("Testing pets model...")
try:
    m2 = tf.keras.models.load_model('saved_model/pets_classification_model.keras', compile=False)
    print("Pets OK:", m2.input_shape)
except Exception as e:
    print("Pets ERROR:", e)

print("All done.")
