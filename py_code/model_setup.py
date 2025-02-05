# Ask Harris about the code when we set this up; how to access the model from another py file
def model():
    model = Sequential()
    model.add(Dense(32, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))