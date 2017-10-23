import os
from utils import *
from parameters import *
from nltk.tokenize import StanfordTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dense, Input, Flatten, Dropout, Merge, AveragePooling1D, Reshape, Conv2D, MaxPooling2D, activations, GRU, Lambda, RepeatVector, Activation, Permute, Recurrent, LSTM, SimpleRNN
from keras.layers.wrappers import Bidirectional, TimeDistributed
from nltk.tokenize import TweetTokenizer
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras import backend as K
from keras.callbacks import Callback
import abc
from keras.models import load_model
from keras.engine.topology import Layer, InputSpec
from keras import initializations, regularizers
from keras.layers.recurrent import time_distributed_dense

OPTIMIZER = 'adam' #rmsprop, adam


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.sentence = Input(shape=(config['sentence_maxlen'],), dtype='int32')
        self.target = Input(shape=(config['target_maxlen'],), dtype='int32')
        self.sentence_val = Input(shape=(config['sentence_maxlen'],), dtype='int32')
        self.target_val = Input(shape=(config['target_maxlen'],), dtype='int32')
        self._models = None
        self._st_model = None
        self.trained_flag = False

    @abc.abstractmethod
    def build(self, **args):
        return

    def compile(self, **args):
        if self._models is None:
            self._models = self.build(**args)
        if self._st_model is None:
            s_output, t_output = self._models
            if t_output is not None:
                s_output = Dense(self.config['stances_num'])(s_output)
                t_output = Dense(self.config['stances_num'])(t_output)
                pred = Activation('softmax')(Activation('tanh')(Merge(mode='sum')([s_output, t_output])))
            else:
                pred = Dense(self.config['stances_num'], activation='softmax')(s_output)
            self._st_model = Model(input=[self.sentence, self.target], output=pred)
        self._st_model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['acc'])

    def fit(self, x_s_train, x_t_train, y_train, x_s_val, x_t_val, y_val, epoch):
        assert self._st_model is not None
        if epoch == 1:
            self._st_model.fit([x_s_train, x_t_train], y_train, validation_data=([x_s_val, x_t_val], y_val), nb_epoch=epoch, batch_size=self.config['batch_size'], verbose=0)
        else:
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            self._st_model.fit([x_s_train, x_t_train], y_train, validation_data=([x_s_val, x_t_val], y_val), nb_epoch=epoch, batch_size=self.config['batch_size'], verbose=0, callbacks=[early_stopping])
        self.trained_flag = True

    def predict(self, x_s_val, x_t_val):
        assert self.trained_flag is True
        y_val_pred = self._st_model.predict_on_batch([x_s_val, x_t_val])
        return self.config['classes'][y_val_pred.argmax(1)]

    def evaluation_metric_2f1(self, y_true, y_pred):
        return f1_score(self.config['classes'][y_true.argmax(1)], self.config['classes'][y_pred.argmax(1)], average='macro', labels=[self.config['classes'][0], self.config['classes'][-1]])

    def evaluation_metric_3accuracy(self, y_true, y_pred):
        return accuracy_score(self.config['classes'][y_true.argmax(1)], self.config['classes'][y_pred.argmax(1)])

    def evaluate(self, x_s_val, x_t_val, y_val):
        assert self.trained_flag is True
        y_val_pred = self._st_model.predict_on_batch([x_s_val, x_t_val])
        score = self.evaluation_metric_2f1(y_val, y_val_pred)
        return score, self.config['classes'][y_val.argmax(1)], self.config['classes'][y_val_pred.argmax(1)], y_val_pred

    def save_weights(self, filename):
        assert self.trained_flag is True
        self._st_model.save_weights(filename)

    def load_model(self, filename):
        assert self._st_model is not None
        self._st_model.load_weights(filename)


class BiGRU(BaseModel):
    def build(self, embedding_matrix=None):
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], dropout=0.2)
        embedded_sentence = embedding_layer(self.sentence)
        f_gru = GRU(self.config['dim'], consume_less='gpu', dropout_U=0.3, dropout_W=0.3)(embedded_sentence)
        b_gru = GRU(self.config['dim'], consume_less='gpu', go_backwards=True, dropout_U=0.3, dropout_W=0.3)(embedded_sentence)
        s_output = Merge(mode='concat')([f_gru, b_gru])
        s_output = Dropout(self.config['dropout'])(s_output)
        return s_output, None


class ABiGRU(BaseModel):
    def build(self, embedding_matrix=None):
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], dropout=0.2)
        embedded_target = embedding_layer(self.target)
        # Average = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda x: (x[0], x[2]))
        # Average.supports_masking = True
        # q_output = Average(embedded_target)
        q_output = Bidirectional(GRU(self.config['dim'], return_sequences=False, consume_less='gpu'))(embedded_target)

        embedded_sentence = embedding_layer(self.sentence)
        s_output = Bidirectional(GRU(self.config['dim'], return_sequences=True, consume_less='gpu', dropout_U=0.3, dropout_W=0.3))(embedded_sentence)

        tranferred_q_output = Dense(output_dim=self.config['dim']*2)(q_output)
        tranferred_q_output = RepeatVector(self.config['sentence_maxlen'])(tranferred_q_output)
        transferred_s_output = TimeDistributed(Dense(output_dim=self.config['dim']*2))(s_output)
        transferred_s_output = Merge(mode='sum')([transferred_s_output, tranferred_q_output])
        transferred_s_output = Activation('tanh')(transferred_s_output)
        transferred_s_output = TimeDistributed((Dense(output_dim=1, activation="softmax")))(transferred_s_output)
        s_output = Merge(mode='dot', dot_axes=1)([s_output, transferred_s_output])
        s_output = Flatten()(s_output)

        s_output = Dropout(self.config['dropout'])(s_output)
        return s_output, None


class BiGRUCNN(BaseModel):
    def build(self, embedding_matrix=None):
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], dropout=0.2)
        embedded_sentence = embedding_layer(self.sentence)

        s_output = Bidirectional(GRU(self.config['dim'], return_sequences=True, consume_less='gpu', dropout_U=0.3, dropout_W=0.3))(embedded_sentence)

        convs = []
        for each_filter_size in [3, 4, 5]:
            each_conv = Conv1D(100, each_filter_size, activation='relu')(s_output)
            each_conv = MaxPooling1D(self.config['sentence_maxlen']-each_filter_size+1)(each_conv)
            each_conv = Flatten()(each_conv)
            convs.append(each_conv)
        s_output = Merge(mode='concat')(convs)
        s_output = Dropout(self.config['dropout'])(s_output)
        return s_output, None


class GatedBiGRUCNN(BaseModel):
    def build(self, embedding_matrix=None):
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], dropout=0.2)
        embedded_target = embedding_layer(self.target)

        # Average = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda x: (x[0], x[2]))
        # Average.supports_masking = True
        # q_output = Average(embedded_target)

        q_output = Bidirectional(GRU(self.config['dim'], return_sequences=False, consume_less='gpu'))(embedded_target)

        embedded_sentence = embedding_layer(self.sentence)

        s_output = Bidirectional(GRU(self.config['dim'], return_sequences=True, consume_less='gpu', dropout_U=0.3, dropout_W=0.3))(embedded_sentence)

        transferred_q_output = Dense(output_dim=self.config['dim']*2)(q_output)
        transferred_q_output = RepeatVector(self.config['sentence_maxlen'])(transferred_q_output)
        transferred_s_output = TimeDistributed(Dense(output_dim=self.config['dim']*2))(s_output)
        transferred_s_output = Merge(mode='sum')([transferred_s_output, transferred_q_output])
        transferred_s_output = Activation('tanh')(transferred_s_output)
        transferred_s_output = TimeDistributed((Dense(output_dim=self.config['dim']*2, activation="sigmoid")))(transferred_s_output)  # *2
        s_output = Merge(mode='mul')([s_output, transferred_s_output])

        convs = []
        for each_filter_size in [3, 4, 5]:
            each_conv = Conv1D(100, each_filter_size, activation='relu')(s_output)
            each_conv = MaxPooling1D(self.config['sentence_maxlen']-each_filter_size+1)(each_conv)
            each_conv = Flatten()(each_conv)
            convs.append(each_conv)
        s_output = Merge(mode='concat')(convs)
        s_output = Dropout(self.config['dropout'])(s_output)
        return s_output, None