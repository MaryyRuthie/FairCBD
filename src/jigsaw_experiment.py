import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import layers,Input
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sentence_transformers import SentenceTransformer
import tensorflow_constrained_optimization as tfco

from utils import read_jigsaw_data, save_logs, load_logs, save_embeddings, load_embeddings, is_available
from model import get_dense_model
from evaluations import eval_report, mcnemar_test
from plot import plot_perf
from embeddings import get_bert_embeddings
from configs import config as cf
# from train import train_model, create_tensors



#read jogsaw data




data = read_jigsaw_data()

print("Overall toxicity proportion = {0:.2f}%".format(data['target'].mean() * 100))
for i in cf.identity_keys_jigsaw:
    print("\t{} proportion = {:.2f}% | toxicity proportion in {} = {:.2f}%".format(i, data[i].mean()*100, i, data[data[i]]['target'].mean()*100))


if is_available(cf.jigsaw_embeddings_path):
    sentence_embeddings = load_embeddings(dataset='jigsaw')
else:
    sentence_embeddings = get_bert_embeddings(data['comment'])
    save_embeddings(sentence_embeddings, dataset='jigsaw')

train_df, val_test_df = train_test_split(data, train_size=cf.train_size, random_state=cf.random_state, shuffle=True)
val_df, test_df = train_test_split(val_test_df, train_size=cf.val_test_ratio, random_state=cf.random_state, shuffle=True)

train_labels = np.array(train_df['target']).reshape(-1, 1).astype(float)
val_labels = np.array(val_df['target']).reshape(-1, 1).astype(float)
test_labels = np.array(test_df['target']).reshape(-1, 1).astype(float)

train_groups = np.array(train_df[cf.identity_keys_jigsaw]).astype(int)
val_groups = np.array(val_df[cf.identity_keys_jigsaw]).astype(int)
test_groups = np.array(test_df[cf.identity_keys_jigsaw]).astype(int)

train_relevant_obs_indices = np.where(train_df[cf.identity_keys_jigsaw].sum(axis=1))[0]

train, val_test = train_test_split(sentence_embeddings, train_size=cf.train_size, random_state=cf.random_state, shuffle=True)
val, test = train_test_split(val_test, train_size=cf.val_test_ratio, random_state=cf.random_state, shuffle=True)



plain_model = get_dense_model()
plain_model.load_weights('{}/{}.h5'.format(cf.MODELS_DIR,cf.jigsaw_plain_model_name))
test_probs_plain = plain_model.predict(test, batch_size=cf.hyperparams['batch_size'])
test_preds_plain = (test_probs_plain > 0.5).astype("int32")
eval_report(test_labels, test_preds_plain, test_probs_plain, test_groups)
plot_perf(test_labels, test_preds_plain, test_groups, cf.identity_keys_jigsaw, 'Plain model')


constrained_model = get_dense_model()
constrained_model.load_weights('{}/{}_87.h5'.format(cf.MODELS_DIR, cf.jigsaw_constrained_model_name))

test_probs_const = constrained_model.predict(test, batch_size=cf.hyperparams['batch_size'])
test_preds_const = (test_probs_const>0.5).astype('int32')

eval_report(test_labels, test_preds_const, test_probs_const, test_groups)
plot_perf(test_labels, test_preds_const, test_groups, cf.identity_keys_jigsaw, 'Constrained model')

eval_report(test_labels, test_preds_const, test_probs_const, test_groups)
plot_perf(test_labels, test_preds_const, test_groups, cf.identity_keys_jigsaw, 'Constrained model')

mcnemar_test(labels=test_labels.ravel(), model1_preds=test_preds_plain.ravel(), model1_name='Baseline model', 
                                         model2_preds=test_preds_const.ravel(), model2_name='Constrained model')