import keras
import os
import keras_metrics


class ModelClass:
    def __init__(self, model_name="vgg"):
        self.model_name = model_name
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.model = keras.models.Sequential()

    def create_model_inception(self, input_shape=(500, 500, 3), output_size=6):
        base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
        #
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(50, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        predictions = keras.layers.Dense(output_size, activation='linear')(x)
        self.model = keras.models.Model(input=base_model.input, output=predictions)
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy', keras_metrics.precision(),
                                                                  keras_metrics.recall()])

        for layer in base_model.layers:
            layer.trainable = True

        self.model.summary()

    def create_model(self, input_shape=(500, 500, 3), output_size=6):
        img_input = keras.layers.Input(shape=input_shape)

        x = keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same', name='block1_conv2')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        x = keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='block2_conv1')(x)
        x = keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same', name='block2_conv2')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(output_size, activation='linear')(x)

        self.model = keras.models.Model(input=img_input, output=x)

        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy', keras_metrics.precision(),
                                                                  keras_metrics.recall()])

        self.model.summary()

    def load_model(self, weights_path):
        self.create_model()
        self.model.load_weights(weights_path)
        return self.model

    def train_model(self, data_set, target_set):
        print(data_set, target_set)

        file_path = "model-{epoch:02d}-{loss:.4f}.hdf5"

        checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True,
                                                     mode='min', period=10)

        self.model.fit([data_set], y=target_set, batch_size= 5, verbose=1,
                       shuffle=True, epochs=500, validation_split=0.2, callbacks=[checkpoint])

        model_json = self.model.to_json()

        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights("model.h5")


if __name__ == "__main__":
    mode = ModelClass()
    mode.create_model()
