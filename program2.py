import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the mapping data from the CSV file
mapping_file = "c:\\Users\\anand\\OneDrive\\Desktop\\Desgin projrct\\_classes.csv" # Replace with your CSV file path
data = pd.read_csv(mapping_file)
np.random.seed(42)
# Directory containing the images
image_dir = "C:\\Users\\anand\\OneDrive\\Desktop\\Desgin projrct\\valid"  # Replace with the directory containing your images

# Data Preprocessing using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 80-20 split for training-validation
image_size = (100, 100)  # Set your image dimensions

# Train and Validation Data Generators
train_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory=image_dir,
    x_col='filename',
    y_col=data.columns[1:],  # Select all class columns
    target_size=image_size,
    batch_size=32,
    class_mode='raw',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory=image_dir,
    x_col='filename',
    y_col=data.columns[1:],
    target_size=image_size,
    batch_size=32,
    class_mode='raw',
    subset='validation'
)

# CNN Model Setup
num_classes = len(data.columns) - 1  # Number of classes excluding 'Image' column

cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
# Add more layers as needed
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(num_classes, activation='sigmoid'))  # Sigmoid for multi-label classification

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the CNN model
cnn_model.save("C:\\dpev\\model\\m2.h5")
