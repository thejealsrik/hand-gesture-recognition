import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib  # For saving/loading the Random Forest model

# Define data path
train_data_dir = "C:\\Users\\anand\\OneDrive\\Desktop\\dataset\\model1"

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Define validation split (e.g., 80% for training, 20% for validation)
validation_split = 0.2

# Creating separate data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split  # Set the validation split here
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Specify this as the training set
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Specify this as the validation set
)

# Define the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Remove the final Dense layer with sigmoid, we will use this model for feature extraction
cnn_feature_extractor = Model(inputs=model.input, outputs=model.layers[-1].output)

# Extract features using the CNN model (training set)
X_train = cnn_feature_extractor.predict(train_generator, steps=train_generator.samples // batch_size)
y_train = train_generator.classes  # Labels from the data generator

# Extract features using the CNN model (validation set)
X_val = cnn_feature_extractor.predict(validation_generator, steps=validation_generator.samples // batch_size)
y_val = validation_generator.classes  # Labels from the validation generator

# Train the Random Forest Classifier on the CNN extracted features
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the Random Forest classifier on the validation set
y_pred = rf_classifier.predict(X_val)
rf_accuracy = accuracy_score(y_val, y_pred)
print(f"Random Forest validation accuracy: {rf_accuracy * 100:.2f}%")

# Save the Random Forest model to a file using joblib
model_path = "C:\\dpev\\model\\rf_model.pkl"
joblib.dump(rf_classifier, model_path)
print(f"Random Forest model saved at: {model_path}")


model.save("C:\\dpev\\model\\m1.h5")
