# Tensorflow learning algorithms
- Some examples of learning algorithms with tensorflow
- Tensorflow 2.0 library along with all other packages (SciPy, Pandas, etc.) are distrubtuted with the Anaconda distribution: https://www.anaconda.com/


Based on the TensorFlow documentation: 
- https://www.tensorflow.org/tutorials/estimator/linear
-https://www.tensorflow.org/tutorials/estimator/premade
- https://www.tensorflow.org/tutorials/keras/classification



Note that these examples are for educational purposes only and models in the "Learning_ Algorithm_Examples" are based on TensorFlow documentation.

# Bitcoin Daily Price Predictor:
Author: Matthew Viafora

- Takes open Bitcoin price from past 365 days
- '0' represents a net loss for the day
- '1' represents a net gain for the day
- DNN Classifier typically outperforms 50/50 chance by average 5%
- Takes date,open price as feature input
- outputs 'result' (1,0)
- Best test set accuracy with (300,300,300) NN: 59% accuracy
