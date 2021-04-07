#Author: Matthew Viafora
#Reference: https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=bQRLq4M1k1jm
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import os

CSV_COLUMN_NAMES = ['Date','Open','Close','Pre_Result','Result']
PREDICTORS = ['Net Gain','Net Loss']


#Of course update this path so that it matches your system
data_folder = r"/Users/mattviafora/Desktop/Tensorflow"

train_path = os.path.join(data_folder, 'BitcoinData_Train.csv')
test_path = os.path.join(data_folder, 'BitcoinData_Test.csv')

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, skipinitialspace=True, skiprows=1, engine="python")
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, skipinitialspace=True, skiprows=1, engine="python")
train.pop('Close')
test.pop('Close')
train.pop('Pre_Result')
test.pop('Pre_Result')


train['Date'] = train['Date'].astype(float)
train['Open'] = train['Open'].astype(float)
train['Result'] = train['Result'].astype(float)
test['Date'] = test['Date'].astype(float)
test['Open'] = test['Open'].astype(float)
test['Result'] = test['Result'].astype(float)



train_y = train.pop('Result')
test_y = test.pop('Result')


def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))


    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


print(train.keys())
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# print("____________________________________")
# print(my_feature_columns)
# print("____________________________________")

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[300,300,300],
    n_classes=2)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000
)

eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))





