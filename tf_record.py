

import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

### get the image
def load_image(img_addr, img_width, img_height):
    img = cv2.imread(img_addr)
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

### convert image into binary format
def get_image_binary(img):
    shape = np.array(img.shape, np.int32)
    img = np.asarray(img,np.uint8)
    return img.tobytes(), shape.tobytes()

### write data into tf record file format
def write_tfrecord(tfrecord_filename, img_directory, img_width, img_height):
    
    #get a list of file in a certain directory
    img_addrs = os.listdir(img_directory)
    
    ### step 1: create a writer
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    
    for img_addr in img_addrs:
        
        ### step 2: load data and process them
        # in this particular case, the image is loaded, and we assign the label 1 if the image is a cat
        img = load_image(os.path.join(img_directory,img_addr), img_width, img_height)
        img, shape = get_image_binary(img)
        if img_addr.find('cat') != -1: 
            label = 1
        else:
            label = 0
        
        ### step 3: create features
        feature = {'label': _int64_feature(label),
                   'image': _bytes_feature(img),
                   'shape':_bytes_feature(shape)}
        features = tf.train.Features(feature=feature)
        
        ### step 4: create example
        example = tf.train.Example(features=features)
        
        ### step 5: write example
        writer.write(example.SerializeToString())
        
    writer.close()

### parse serialized data back into the usable form
def _parse(serialized_data):
    features = {'label': tf.FixedLenFeature([], tf.int64),
               'image': tf.FixedLenFeature([], tf.string),
               'shape': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(serialized_data,
                                      features)
    img = tf.decode_raw(features['image'],tf.uint8)
    shape = tf.decode_raw(features['shape'],tf.int32)
    img = tf.reshape(img, shape)
    return img, features['label']

### read tf record
def read_tfrecord(tfrecord_filename, tfrecord_directory):
    
    ### create dataset
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    dataset = dataset.map(_parse)
    
    ### create iterator
    iterator = dataset.make_initializable_iterator()
    img, label = iterator.get_next()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        counter = 0
        try:
            while True:
                _image, _label = sess.run([img, label])
                print("image " + str(counter) + " label:" + str(_label))
                counter += 1
                plt.imshow(_image)
                plt.show()
        except tf.errors.OutOfRangeError:
            pass
        

if __name__ == '__main__':
    write_tfrecord('cat_classification_tfrecord','content' ,400 ,300)
    read_tfrecord('cat_classification_tfrecord','')

