'''
evaluation related functions
'''
import pandas as pd
from mlxtend.evaluate import mcnemar_table, mcnemar
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_sm
from metrics import (accuracy, precision, recall, auc, f1, false_negative_rate,
                     false_positive_rate, group_false_negative_rates,
                     group_false_positive_rates, group_recalls, group_mccs,
                     group_precisions, false_positive_equality_diff,
                     false_negative_equality_diff, mcc)

# def false_negative_rate(y_true, y_pred):
#     fn = sum((y_true == 1) & (y_pred == 0))
#     total_positives = sum(y_true == 1)
#     return fn / total_positives if total_positives != 0 else 0

# def false_positive_rate(y_true, y_pred):
#     fp = sum((y_true == 0) & (y_pred == 1))
#     total_negatives = sum(y_true == 0)
#     return fp / total_negatives if total_negatives != 0 else 0

# def group_mccs(y_true, y_pred, groups):
#     """
#     Calculate the Matthews Correlation Coefficient (MCC) for each group.
    
#     :param y_true: 1D array of true labels
#     :param y_pred: 1D array of predicted labels
#     :param groups: 2D array where each column is a binary mask for a group
#     :return: List of MCC values for each group
#     """
#     mccs = []
#     num_groups = groups.shape[1]
    
#     for i in range(num_groups):
#         group_mask = groups[:, i] == 1  # Select the group based on the mask
#         group_y_true = y_true[group_mask]
#         group_y_pred = y_pred[group_mask]
        
#         if len(group_y_true) > 0:  # Check to avoid division by zero
#             mcc = matthews_corrcoef(group_y_true, group_y_pred)
#             mccs.append(mcc)
#         else:
#             mccs.append(None)  # No data in this group
    
#     return mccs

# def group_precisions(y_true, y_pred, groups):
#     precisions = []
#     num_groups = groups.shape[1]
    
#     for i in range(num_groups):
#         group_mask = groups[:, i] == 1
#         group_y_true = y_true[group_mask]
#         group_y_pred = y_pred[group_mask]
        
#         if len(group_y_true) > 0:
#             precision = precision_score(group_y_true, group_y_pred, zero_division=1)
#             precisions.append(precision)
#         else:
#             precisions.append(None)
    
#     return precisions

# def group_false_negative_rates(y_true, y_pred, groups):
#     fn_rates = []
#     num_groups = groups.shape[1]

#     for i in range(num_groups):
#         group_mask = groups[:, i] == 1
#         group_y_true = y_true[group_mask]
#         group_y_pred = y_pred[group_mask]

#         if len(group_y_true) > 0:
#             fn = sum((group_y_true == 1) & (group_y_pred == 0))
#             total_positives = sum(group_y_true == 1)
#             fn_rate = fn / total_positives if total_positives > 0 else None
#             fn_rates.append(fn_rate)
#         else:
#             fn_rates.append(None)  # No data in this group

#     return fn_rates

# def group_false_positive_rates(y_true, y_pred, groups):
#     fp_rates = []
#     num_groups = groups.shape[1]

#     for i in range(num_groups):
#         group_mask = groups[:, i] == 1
#         group_y_true = y_true[group_mask]
#         group_y_pred = y_pred[group_mask]

#         if len(group_y_true) > 0:
#             fp = sum((group_y_true == 0) & (group_y_pred == 1))
#             total_negatives = sum(group_y_true == 0)
#             fp_rate = fp / total_negatives if total_negatives > 0 else None
#             fp_rates.append(fp_rate)
#         else:
#             fp_rates.append(None)  # No data in this group

#     return fp_rates

# def false_negative_equality_diff(y_true, y_pred, groups):
#     fn_rates = group_false_negative_rates(y_true, y_pred, groups)
#     valid_fn_rates = [rate for rate in fn_rates if rate is not None]
#     if len(valid_fn_rates) > 0:
#         return max(valid_fn_rates) - min(valid_fn_rates)
#     else:
#         return 0

# def false_positive_equality_diff(y_true, y_pred, groups):
#     fp_rates = group_false_positive_rates(y_true, y_pred, groups)
#     valid_fp_rates = [rate for rate in fp_rates if rate is not None]
#     if len(valid_fp_rates)>0:
#         return max(fp_rates) - min(fp_rates)
#     else:
#         return 0

# def group_recalls(y_true, y_pred, groups):
#     recalls = []
#     num_groups = groups.shape[1]

#     for i in range(num_groups):
#         group_mask = groups[:, i] == 1
#         group_y_true = y_true[group_mask]
#         group_y_pred = y_pred[group_mask]

#         if len(group_y_true) > 0:
#             recall = recall_score(group_y_true, group_y_pred, zero_division=1)
#             recalls.append(recall)
#         else:
#             recalls.append(None)  # No data in this group

#     return recalls

# def eval_report(true_labels, predicted_labels,
#                 predicted_probabilities, groups):
#     '''
#     [true_labels]             : list or python 1D array of zeros or ones
#     [predicted_labels]        : list or python 1D array of zeros or ones
#     [predicted_probabilities] : list or python 1D array of floats - range [0,1]
#     [groups]                  : 2D numpy array of binary values
#     '''
#     true_labels = true_labels.astype(int)
#     print(true_labels)
#     print(predicted_labels)
#     print('Evaluation on test data:')
#     print('\tAUC = {:.4f}'.format(roc_auc_score(true_labels, predicted_probabilities)))
#     print('\tAccuracy = {:.2f} %'.format(
#                                 accuracy_score(true_labels, predicted_labels) * 100))
#     print('\tMatthews correlation coefficient = {:.4f}'.format(
#                                            matthews_corrcoef(true_labels, predicted_labels)))
#     print('\t\tGroup-specific MCCs = {}'.format(
#                             group_mccs(true_labels, predicted_labels, groups)))
#     print('\tf1 score = {:.4f}'.format(
#                                             f1_score(true_labels, predicted_labels)))
#     print('\tPrecision = {:.4f}'.format(
#                                      precision_score(true_labels, predicted_labels)))
#     print('\t\tGroup-specific precisions = {}'.format(
#                       group_precisions(true_labels, predicted_labels, groups)))
#     print('\tRecall (sensitivity) = {:.4f}'.format(
#                                         recall_score(true_labels, predicted_labels)))
#     # print('\t\tGroup-specific recalls = {}'.format(
#     #                      group_recalls(true_labels, predicted_labels, groups)))
#     print('\tFalse negative (miss) rate = {:.4f} %'.format(
#                      100 * float(false_negative_rate(true_labels, predicted_labels))))
    
#     # print('\t\tGroup-specific false negative rates = {} %'.format(
#     #   100 * group_false_negative_rates(true_labels, predicted_labels, groups)))
    
#     # Calculate false negative rates
#     fn_rates = group_false_negative_rates(true_labels, predicted_labels, groups)

# # Format each rate in the list as a percentage
#     # formatted_fn_rates = [f"{rate * 100:.2f}%" if rate is not None else "N/A" for rate in fn_rates]

#     # print('\t\tGroup-specific false negative rates = {}'.format(formatted_fn_rates))
    
#     fp_rates = group_false_positive_rates(true_labels, predicted_labels, groups)
#     # formatted_fp_rates = [f"{rate * 100:.2f}%" if rate is not None else "N/A" for rate in fp_rates]
#     # print('\t\tGroup-specific false positive rates = {}'.format(formatted_fp_rates))

#     fned = float(false_negative_equality_diff(true_labels, predicted_labels, groups))
#     fped = float(false_positive_equality_diff(true_labels, predicted_labels, groups))
#     print('\tFalse negative equality difference (per group) = {:.4f} | '
#           'total FNED = {:.4f}'.format(
#                                       fned / groups.shape[1], fned)
#           )
#     print('\tFalse positive equality difference (per group) = {:.4f} | '
#           'total FPED = {:.4f}'.format(
#                                       fped / groups.shape[1], fped)
#           )
#     print('\tTotal equality difference (bias) = {:.4f}'.format((fned + fped)))

#     return None


def eval_report(true_labels, predicted_labels,
                predicted_probabilities, groups):
    '''
    [true_labels]             : list or python 1D array of zeros or ones
    [predicted_labels]        : list or python 1D array of zeros or ones
    [predicted_probabilities] : list or python 1D array of floats - range [0,1]
    [groups]                  : 2D numpy array of binary values
    '''

    print('Evaluation on test data:')
    print('\tAUC = {:.4f}'.format(auc(true_labels, predicted_probabilities)))
    print('\tAccuracy = {:.2f} %'.format(
                                accuracy(true_labels, predicted_labels) * 100))
    print('\tMatthews correlation coefficient = {:.4f}'.format(
                                           mcc(true_labels, predicted_labels)))
    print('\t\tGroup-specific MCCs = {}'.format(
                            group_mccs(true_labels, predicted_labels, groups)))
    print('\tf1 score = {:.4f}'.format(
                                            f1(true_labels, predicted_labels)))
    print('\tPrecision = {:.4f}'.format(
                                     precision(true_labels, predicted_labels)))
    print('\t\tGroup-specific precisions = {}'.format(
                      group_precisions(true_labels, predicted_labels, groups)))
    print('\tRecall (sensitivity) = {:.4f}'.format(
                                        recall(true_labels, predicted_labels)))
    print('\t\tGroup-specific recalls = {}'.format(
                         group_recalls(true_labels, predicted_labels, groups)))
    print('\tFalse negative (miss) rate = {:.4f} %'.format(
                     100 * false_negative_rate(true_labels, predicted_labels)))
    print('\t\tGroup-specific false negative rates = {} %'.format(
      100 * group_false_negative_rates(true_labels, predicted_labels, groups)))
    print('\tFalse positive (false alarm) rate = {:.4f} %'.format(
                     100 * false_positive_rate(true_labels, predicted_labels)))
    print('\t\tGroup-specific false positive rates = {} %'.format(
      100 * group_false_positive_rates(true_labels, predicted_labels, groups)))
    fned = false_negative_equality_diff(true_labels, predicted_labels, groups)
    fped = false_positive_equality_diff(true_labels, predicted_labels, groups)
    print('\tFalse negative equality difference (per group) = {:.4f} | '
          'total FNED = {:.4f}'.format(
                                      fned / groups.shape[1], fned)
          )
    print('\tFalse positive equality difference (per group) = {:.4f} | '
          'total FPED = {:.4f}'.format(
                                      fped / groups.shape[1], fped)
          )
    print('\tTotal equality difference (bias) = {:.4f}'.format((fned + fped)))

    return None


def mcnemar_test(labels, model1_preds, model1_name, model2_preds, model2_name):
    '''
    Performs McNemar's test for paired nominal data
    Ref: McNemar, Quinn, 1947. "Note on the sampling error of the difference
         between correlated proportions or percentages".
         Psychometrika. 12 (2): 153–157.

    [labels]       : list or 1D numpy array, correct labels (0 or 1)
    [model1_preds] : list or 1D numpy array, predictions of the first model
    [model1_name]  : str, name of the first model
    [model2_preds] : list or 1D numpy array, predictions of the second model
    [model2_name]  : str, name of the second model
    '''

    contigency_table = mcnemar_table(y_target=labels.ravel(),
                                     y_model1=model1_preds.ravel(),
                                     y_model2=model2_preds.ravel())
    contigency_df = pd.DataFrame(contigency_table,
                                 columns=['{} correct'.format(model1_name),
                                          '{} wrong'.format(model1_name)],
                                 index=['{} correct'.format(model2_name),
                                        '{} wrong'.format(model2_name)])
    print(contigency_df)

    # 'mlxtend' library implementation
    print("\n'mlxtend' library implementation")
    chi2, p = mcnemar(ary=contigency_table, exact=True)
    print('\tchi-squared = {}'.format(chi2))
    print('\tp-value = {}'.format(p))

    # 'statsmodels' library implementation
    print("\n'statsmodels' library implementation")
    print(mcnemar_sm(contigency_table, exact=True))

    return None