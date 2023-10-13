import tensorflow as tf
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import cv2

mnist = tf.keras.models.load_model('models/mnist.h5')
cifar10 = tf.keras.models.load_model('models/cifar10.h5')
cifar100 = tf.keras.models.load_model('models/cifar100.h5')
fashion_mnist = tf.keras.models.load_model('models/fashion_mnist.h5')

classes = {
    'fashion_mnist' : ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    'cifar10' : ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'cifar100' : ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
           'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra',
           'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
           'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
           'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
           'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
           'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
           'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
}

st.title("Keras Datasets")

image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
if image:
    image = np.array(Image.open(image))
    st.image(image)

    option = st.selectbox(
        'Which model do you like to use model?',
        ('mnist', 'cifar10', 'cifar100', 'fashion_mnist'))

    if option=='mnist':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA) / 255.
        img = np.expand_dims(img, axis=0)

        if st.button('Predict!'):
            result = mnist.predict(img, verbose=0)
            st.write('Result:',result[0].argmax())
            st.bar_chart(result[0])
            
    else:
        img = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA) / 255.
        img = np.expand_dims(img, axis=0)

        if st.button('Predict!'):
            result = eval(f'{option}.predict(img, verbose=0)')
            st.write('Result:',classes[option][result[0].argmax()])
            st.bar_chart(pd.DataFrame(result[0], index=classes[option]))
