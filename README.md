# Tensorflow learning algorithms
- Some examples of learning algorithms with tensorflow
- Tensorflow 2.0 library along with all other packages (SciPy, Pandas, etc.) are distrubtuted with the Anaconda distribution: https://www.anaconda.com/


Credit for some of the code goes to https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=bQRLq4M1k1jm


# Bitcoin Daily Price Predictor:

- Takes open Bitcoin price from past 365 days
- '0' represents a net loss for the day
- '1' represents a net gain for the day
- DNN Classifier typically outperforms 50/50 chance by average 5%
- Takes date,open price as feature input
- outputs 'result' (1,0)
- Best test set accuracy with (300,300,300) NN: 59% accuracy
