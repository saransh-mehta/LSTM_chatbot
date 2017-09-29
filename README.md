# LSTM_chatbot
Implementation of a Deep Learning chatbot using Keras with Tensorflow backend
First, Google's Word2vec model has been trained with word2vec_test.py to generate 300D vector equivalent of
unique words present. These vectors are dumped into binary file which is loaded later to convert the user's query into vector form.
The LSTM model comprises of 4 layers and takes input as these vector equivalent of given sentence.
The model is trained on the data collected from chatterbox corpus.
Due to unavailabilty of good quality and quantity of data, the bot suffers in producing accurate results.
