from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.cvs", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')

#print(train.head())
#print(train.shape)

def input_fn(features, labels, training=True, batch_size=256):
    #converts inputs into datasets
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

my_feature_columns=[]
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
#print(my_feature_columns)
#you dont need to do all the unqiue operators like in Linear regression as the data is encoded already for us
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[30, 10], n_classes=3)

classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)
#lambda is used as an alternative to the function within a function,
#lambda is an anonymous function that can be defined in one line
#steps similar to ephoc expect it will use the same data 5000 times and repeat as necessary

eval_results = classifier.evaluate(input_fn=lambda : input_fn(test, test_y, training=False))



print('\n test set accuracy: {accuracy:0.3f}\n'.format(**eval_results))
