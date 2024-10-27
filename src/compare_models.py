import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import read_wiki_data, clean_wiki_data, read_gab_data, read_jigsaw_data, read_twitter_data, save_embeddings, load_embeddings, is_available
from embeddings import get_bert_embeddings
from model import get_dense_model
from evaluations import eval_report, mcnemar_test
from plot import plot_perf
from configs import config as cf


##compare plan and constraint JIGSAW model
data = read_jigsaw_data()

if is_available(cf.jigsaw_embeddings_path):
    sentence_embeddings = load_embeddings(dataset='jigsaw')
else:
    sentence_embeddings = get_bert_embeddings(data['comment'])
    save_embeddings(sentence_embeddings, dataset='jigsaw')
    
_, val_test_df = train_test_split(data, train_size=cf.train_size, random_state=cf.random_state, shuffle=True)
_, test_df = train_test_split(val_test_df, train_size=cf.val_test_ratio, random_state=cf.random_state, shuffle=True)
test_labels = np.array(test_df['target']).reshape(-1, 1).astype(float)
test_groups = np.array(test_df[cf.identity_keys_jigsaw]).astype(int)
_, val_test = train_test_split(sentence_embeddings, train_size=cf.train_size, random_state=cf.random_state, shuffle=True)
_, test = train_test_split(val_test, train_size=cf.val_test_ratio, random_state=cf.random_state, shuffle=True)


plain_model = get_dense_model()
plain_model.load_weights('{}/{}.h5'.format(cf.MODELS_DIR, cf.jigsaw_plain_model_name))
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