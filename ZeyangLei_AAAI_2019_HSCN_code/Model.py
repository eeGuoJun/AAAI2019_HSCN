#coding=utf-8


import numpy as np
import data_preprocess
from wordvec import train_word2vec,load_Glove
import codecs
from sklearn import metrics
import random as rn

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, subtract
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras import regularizers
import time
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from keras.layers.core import Dense,Dropout,Lambda,Activation,Reshape,RepeatVector,Permute
from keras import optimizers
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D
from keras import backend as K
from keras.layers import merge,Input,Flatten, Input ,Convolution1D,MaxPooling1D,TimeDistributed
from keras import regularizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import Bidirectional
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
import tensorflow as tf

##Reproduction
np.random.seed(0)
rn.seed(0)

# ---------------------- Parameters section -------------------

# Data source
dataset='Laptop'
# Model Hyperparameters
embedding_dim = 300
dropout_prob = (0.5, 0.8)
hidden_dims = 10
temperature = 0.7
lstm_dim= 20
# Training parameters
batch_size = 40
num_epochs = 100 #40

# ---------------------- Parameters end -----------------------


def load_data(dataset):
    x,x_aspect, y, vocabulary, vocabulary_inv_list = data_preprocess.load_data_aspect(dataset)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    # y = y.argmax(axis=1)

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    x_aspect=x_aspect[shuffle_indices]
    y = y[shuffle_indices]
    # train_len = int(len(x) * 0.9)
    train_len=2328
    x_train = x[:train_len]
    x_train_aspect=x_aspect[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    x_test_aspect = x_aspect[train_len:]
    y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv,x_train_aspect,x_test_aspect


# Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv,x_train_aspect,x_test_aspect = load_data(dataset)
x_train_final=np.concatenate((x_train,x_train_aspect),axis=1)
x_test_final=np.concatenate((x_test,x_test_aspect),axis=1)
sent_maxlen = x_test.shape[1]
aspect_maxlen=x_test_aspect.shape[1]
if sent_maxlen != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sent_maxlen = x_test.shape[1]


print("x_train shape:", x_train.shape)
print("x_train_aspect shape:", x_train_aspect.shape) 
print("x_test shape:", x_test.shape)
print("x_test_aspect shape:", x_test_aspect.shape) 
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
embedding_weights =load_Glove(vocabulary_inv)

# Build model
def get_aspect(X,sent_maxlen):
    ans = X[:,sent_maxlen:, :]
    return ans
def get_content(X,sent_maxlen):
    ans = X[:,:sent_maxlen, :]
    return ans
def get_slice_1(X,index):
	ans=X[:,index,:]
	return ans
def get_slice_2(X,index):
	ans=X[:,:,index]
	return ans

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  eps=1e-20
  y = tf.log(logits) - tf.log(-tf.log(tf.random_uniform(tf.shape(logits),minval=0,maxval=1) + eps) + eps)
  return tf.nn.softmax( y / temperature)

def transpose_matrix(X):
	X=tf.transpose(X, perm=[0,2,1])
	return X

def myloss(y_true,y_pred,e1=0.001,e2=0.01,e3=0.1,nb_classes=3):
	Loss=K.mean(K.categorical_crossentropy(y_true,y_pred),axis=-1)
	Loss1=e1*K.mean(K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/nb_classes),axis=-1)
	Loss2=e2*K.mean(K.abs(y_true-y_pred)/nb_classes,axis=-1) #应该满足其输出对于多类别的问题，应该让其类内距离较小
	Total_loss=Loss+Loss1+Loss2
	return Total_loss

model_input = Input(shape=(sent_maxlen+aspect_maxlen,))
z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sent_maxlen+aspect_maxlen, name="embedding")(model_input)
x = Dropout(0.5)(z)

w_aspect = Lambda(get_aspect,output_shape=(aspect_maxlen,embedding_dim), arguments={'sent_maxlen':sent_maxlen},name="w_aspect")(x)
print w_aspect.shape
w_content = Lambda(get_content,output_shape=(sent_maxlen,embedding_dim),arguments={'sent_maxlen':sent_maxlen}, name="w_content")(x)
print w_content.shape

# Pre-reading
print ("Excuting Pre-reading Stage")
#Equation-1
print ("Equation-1")
Attention_matrix = Lambda(lambda x:tf.matmul(x[0],x[1],transpose_b=True))([w_content,w_aspect])
print Attention_matrix.shape

#Equation-2
print ("Equation-2")
Average_attention_aspect = GlobalAveragePooling1D()(Attention_matrix)
print Average_attention_aspect.shape

#Equation-3
print ("Equation-3")
Inverse_Attention_matrix = Permute((2, 1))(Attention_matrix)
print Inverse_Attention_matrix.shape
Average_Attention_content = GlobalAveragePooling1D()(Inverse_Attention_matrix)
print Average_Attention_content.shape

#Equation-4
print ("Equation-4")
m=40
Average_attention_aspect = Reshape((-1,1))(Average_attention_aspect)
Average_attention_aspect = Lambda(lambda x,embedding_dim:K.repeat_elements(x,rep=embedding_dim,axis=2),arguments={'embedding_dim':embedding_dim})(Average_attention_aspect)
print Average_attention_aspect.shape
Average_attention_aspect = Lambda(lambda x:tf.multiply(x[0],x[1]))([Average_attention_aspect,w_aspect])
print Average_attention_aspect.shape
Concat_layer_aspect=merge([w_aspect,Average_attention_aspect], mode='sum')
print Concat_layer_aspect
Concat_layer_aspect=Dense(m,activation=None,use_bias=False)(Concat_layer_aspect)
w_aspect_1=Lambda(lambda x:K.elu(x,alpha=0.8))(Concat_layer_aspect)

#Equation-5
print ("Equation-5")
Average_Attention = Reshape((-1,1))(Average_Attention_content)
Average_Attention = Lambda(lambda x,embedding_dim:K.repeat_elements(x,rep=embedding_dim,axis=2),arguments={'embedding_dim':embedding_dim})(Average_Attention)
print Average_Attention.shape
Average_Attention = Lambda(lambda x:tf.multiply(x[0],x[1]))([Average_Attention,w_content])
print Average_Attention.shape
Concat_layer_content=merge([w_content,Average_Attention],mode='sum')
print Concat_layer_content
Concat_layer_content=Dense(m,activation=None,use_bias=False)(Concat_layer_content)
w_content_1=Lambda(lambda x:K.elu(x,alpha=0.8))(Concat_layer_content)

embedding_shape=int(Concat_layer_content.shape[2])
print embedding_shape

##Active-reading
print ("Excuting Active-reading Stage")
print ("Generating Skip-reading Module")
#Skip-reading module
#Equation_6
print ("Equation-6")
Concat_layer_aspect_transpose=Lambda(transpose_matrix)(w_aspect_1)
print Concat_layer_aspect_transpose.shape
gate_content_aspect=Lambda(lambda x:tf.matmul(x[0],x[1]))([w_content_1,Concat_layer_aspect_transpose])
gate_content_aspect=Dense(2,activation="tanh",use_bias=False)(gate_content_aspect)
gate_content_aspect=Activation('softmax')(gate_content_aspect)
print gate_content_aspect.shape
gate_content_aspect=Lambda(gumbel_softmax_sample,arguments={'temperature':temperature})(gate_content_aspect)
print gate_content_aspect.shape
get_slice=Lambda(get_slice_2,arguments={'index':1})(gate_content_aspect)
print get_slice.shape
get_slice=Reshape((-1,1))(get_slice)
print get_slice.shape
get_slice = Lambda(lambda x,embedding_dim:K.repeat_elements(x,rep=embedding_dim,axis=2),arguments={'embedding_dim':embedding_shape})(get_slice)
print get_slice.shape
Context_choose=Lambda(lambda x:tf.multiply(x[0],x[1]))([w_content_1,get_slice])
print Context_choose.shape

#Semantic Composition Module
#Equation 10-13
print ("Equation 10-13")
GRU_layer=GRU(lstm_dim,return_sequences=True,dropout=0.5,kernel_regularizer=regularizers.l2(0.000001))
GRU_layer1=GRU(lstm_dim,return_sequences=True,dropout=0.5,kernel_regularizer=regularizers.l2(0.000001))
content_encoder=GRU_layer(Context_choose)
print content_encoder.shape
content_gru=GlobalAveragePooling1D()(content_encoder)
aspect_encoder=GRU_layer1(w_aspect_1)
print aspect_encoder.shape
aspect_gru=GlobalAveragePooling1D()(aspect_encoder)
print aspect_gru.shape
aspect_gru_reshape= Reshape((-1,1))(aspect_gru)
print aspect_gru_reshape.shape
aspect_content_attention=Lambda(lambda x:tf.matmul(x[0],x[1]))([content_encoder,aspect_gru_reshape])
print aspect_content_attention.shape
aspect_content_attention=Dense(1,activation=None,use_bias=False)(aspect_content_attention)
#aspect_content_attention=TimeDistributed(Dense(1,activation='tanh',use_bias=False)(aspect_content_attention))
print aspect_content_attention.shape
aspect_content_attention=Activation('softmax')(aspect_content_attention)
content_encoder_transpose=Lambda(transpose_matrix)(content_encoder)
aspect_content_attention=Lambda(lambda x:tf.matmul(x[1],x[0]))([aspect_content_attention,content_encoder_transpose])
sentence_representation=Reshape((-1,))(aspect_content_attention)
#model_output=Dense(3, activation="softmax")(sentence_representation)

##Re-reading
print ("Re-reading")
sentence_semantic = Dense(lstm_dim, activation="tanh", use_bias=True)(sentence_representation)
sentence_semantic= Dropout(0.5)(sentence_semantic)
print sentence_semantic.shape
sentence_semantic = BatchNormalization()(sentence_semantic)
aspect_gru = BatchNormalization()(aspect_gru)
sub_semantic=subtract([aspect_gru,sentence_semantic])
print sub_semantic.shape
sub_semantic_reshape=Reshape((-1,1))(sub_semantic)
print sub_semantic_reshape.shape
sub_attention = Lambda(lambda x:tf.matmul(x[0],x[1]))([content_encoder,sub_semantic_reshape])
print sub_attention.shape
sub_attention_1 = Dense(2,activation=None, use_bias=False)(sub_attention)
print sub_attention_1.shape
sub_attention_2=Activation('softmax')(sub_attention_1)
print sub_attention_2.shape
sub_attention_3=Lambda(gumbel_softmax_sample,arguments={'temperature':temperature})(sub_attention_2)
print sub_attention_3.shape
get_slice_1=Lambda(get_slice_2,arguments={'index':1})(sub_attention_3)
print get_slice_1.shape
get_slice_2=Reshape((-1,1))(get_slice_1)
print get_slice_2.shape
get_slice_3 = Lambda(lambda x,embedding_dim:K.repeat_elements(x,rep=embedding_dim,axis=2),arguments={'embedding_dim':embedding_shape})(get_slice_2)
print get_slice_3.shape
Context_choose_1=Lambda(lambda x:tf.multiply(x[0],x[1]))([w_content_1,get_slice_3])
print Context_choose_1.shape
content_encoder_1=GRU_layer(Context_choose_1)
print content_encoder_1.shape
content_gru_1=GlobalAveragePooling1D()(content_encoder_1)

##Final representation
sentence_final= merge([sentence_representation,content_gru_1],mode='concat')
sentence_final=Dropout(0.5)(sentence_final)
model_output=Dense(3, activation="softmax")(sentence_final)

model = Model(model_input, model_output)
model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

# Initialize weights with word2vec
weights = np.array([v for v in embedding_weights.values()])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([weights])

from sklearn.metrics import precision_recall_fscore_support

def macro_f1(y_true, y_pred):
#    preds = np.argmax(y_pred, axis=-1)
#    true = np.argmax(y_true, axis=-1)
    true=y_true
    preds=y_pred
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return f_macro

# Train the model
"""
model.load_weights('output/'+dataset+'/Attention_LSTM_best_weights_20180807235611.h5')
result=model.predict(x_test_final,batch_size=638)
y_label = y_test.argmax(axis=1)
y_pred = []
for i in result:
	j=i.tolist()
	if str(j.index(max(j))) == '0':
		y_pred.append(0)
	elif str(j.index(max(j))) == '1':
		y_pred.append(1)
	else:
		y_pred.append(2)
print metrics.accuracy_score(y_label, y_pred)
print macro_f1(y_label,y_pred)
"""
date=time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
best_model=ModelCheckpoint('output/'+dataset+'/Attention_LSTM_best_weights_%s.h5'%date,monitor='val_acc',save_best_only=True,save_weights_only=False)
model.fit(x_train_final, y_train, batch_size=batch_size, epochs=num_epochs,callbacks=[best_model],shuffle=False,
          validation_split=0.1, verbose=2)
score, acc = model.evaluate(x_test_final, y_test,
								batch_size=batch_size)
print acc
best_model=load_model('output/'+dataset+'/Attention_LSTM_best_weights_%s.h5'%date,custom_objects={"tf": tf})
# best_model=load_model('output/'+dataset+'/Attention_LSTM_best_weights_20180125170634.h5',custom_objects={"tf": tf})
score, best_acc = best_model.evaluate(x_test_final, y_test,
								batch_size=638)
print best_acc
result=best_model.predict(x_test_final,batch_size=638)
y_label = y_test.argmax(axis=1)
y_pred = []
for i in result:
	j=i.tolist()
	if str(j.index(max(j))) == '0':
		y_pred.append(0)
	elif str(j.index(max(j))) == '1':
		y_pred.append(1)
	else:
		y_pred.append(2)
print metrics.accuracy_score(y_label, y_pred)
print macro_f1(y_label,y_pred)

