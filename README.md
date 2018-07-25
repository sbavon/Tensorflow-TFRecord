# Tensorflow's TFRecord with tf.data - Short Note

In this implementation, I convert image and label to TFrecord file, and then read the file using tf.data

## What is TFRecord ?
- TFRecord is a Tensorflow's standard file format, which stores data in binary format
- It is recommended to use for machine learning projects, especially one that involves with big dataset

## Why it is recommended ?
- Since data are stored in binary format, it is faster and more flexible
- For a very large dataset whose size is too big to fit into your memory, it is easy to read partial data from the file

## How to implement ?
- There are two mainstep to implement TFRecord: convert data to TFRecord, and read TFRecord data

### Convert data to TFRecord format format
To convert data to TFRecord file format, there are five main steps as below:

#### 1. Create a writer 
``` python
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
```
#### 2. Prepare data and make it binary (In this solution, I use input as an image)
``` python
    img = load_image(os.path.join(img_directory,img_addr), img_width, img_height)
    img, shape = get_image_binary(img)
    if img_addr.find('cat') != -1: 
        label = 1
    else:
        label = 0
```
#### 3. Create features
``` python
    feature = {'label': _int64_feature(label),
               'image': _bytes_feature(img),
               'shape':_bytes_feature(shape)}
    features = tf.train.Features(feature=feature)
```
#### 4. create example
``` python
    example = tf.train.Example(features=features)
```
#### 5. write the example in the TFRecord file
``` python
    writer.write(example.SerializeToString())
```

### read TFRecord using tf.data
Reading data using tf.data is much easier than traditional method. You can follow the instructions below:

#### 1. Create parse function that map serialized data back to the usable form
``` python
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
```
#### 2. create a dataset and map it with parse function
``` python
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    dataset = dataset.map(_parse)
```

#### 3. create an iterator. That's it
``` python
    iterator = dataset.make_initializable_iterator()
    img, label = iterator.get_next()
```

## Execution
To test the code, just run *tf_record.py* and you will see the tfrecord file in the working directory

## Reference
- [CS20: "TensorFlow for Deep Learning Research", Stanford](http://web.stanford.edu/class/cs20si/)
- [Tensorflow.org](https://www.tensorflow.org/guide/datasets)
