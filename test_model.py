import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Show input shape
print("Model input shape:", model.input_shape)

# Show summary
model.summary()
