
'''
visualization functions
'''

import matplotlib.pyplot as plt
import numpy as np


from metrics import (accuracy, precision, recall, auc, f1, false_negative_rate,
                     false_positive_rate, group_false_negative_rates,
                     group_false_positive_rates, group_recalls, group_mccs,
                     group_precisions, false_positive_equality_diff,
                     false_negative_equality_diff, mcc)

# plt.rcParams['font.sans-serif'] = "Liberation Sans"
# # plt.rcParams['font.family'] = "sans-serif"
# plt.rcParams['font.size'] = 13
cb_palette = ['#58D68D', '#C39BD3', '#F7DC6F', '#85C1E9',
              "#EC7063", '#ff9900', '#8c8c8c']


# def false_negative_rate(y_true, y_pred):
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
#     fn = sum((y_true == 1) & (y_pred == 0))
#     total_positives = sum(y_true == 1)
#     return fn / total_positives if total_positives != 0 else 0

# def false_positive_rate(y_true, y_pred):
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
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
#             fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
#             total_positives = np.sum(group_y_true == 1)
#             # Replace None with 0 if there are no positives
#             fn_rate = fn / total_positives if total_positives > 0 else None
#             fn_rates.append(fn_rate)
#         else:
#             fn_rates.append(None)  # Replace None with 0 for empty groups

#     return np.array(fn_rates)

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
#     return max(fn_rates) - min(fn_rates)

# def false_positive_equality_diff(y_true, y_pred, groups):
#     fp_rates = group_false_positive_rates(y_true, y_pred, groups)
#     return max(fp_rates) - min(fp_rates)

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


def plot_perf(labels, predictions, groups, group_names, title=''):
    '''
    [labels]      : true labels
    [predictions] : predicted labels
    [groups]      : 2D numpy array of binary values
    [group_names] : str
    [title]       : str
    '''

    num_groups = groups.shape[1]

    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    # group_fprs = [
    # rate * 100 if rate is not None else 0 for rate in group_false_positive_rates(labels, predictions, groups)
    # ]
    # overall_fpr = 100 * false_positive_rate(labels, predictions)
    # ax[0].bar(range(num_groups + 1), np.append(group_fprs, overall_fpr))
    # ax[0].set_xticks(range(num_groups + 1))
    # ax[0].set_xticklabels(group_names + ['overall'])
    # ax[0].set_xlabel("False Positive Rate (%)")
    # ax[0].set_ylim([0, max(np.append(group_fprs,
    #                                  100 * false_positive_rate(labels,
    #                                                            predictions)))
    #                 + 1])
    # ax[0].grid(axis='y', linestyle='--')

    # group_fnrs = 100 * group_false_negative_rates(labels, predictions, groups)
    # ax[1].bar(range(num_groups + 1),
    #           np.append(group_fnrs, 100 * false_negative_rate(labels,
    #                                                           predictions)),
    #           color=cb_palette)
    # ax[1].set_xticks(range(num_groups + 1))
    # ax[1].set_xticklabels(group_names + ['overall'])
    # ax[1].set_xlabel("False Negative Rate (%)")
    # ax[1].set_ylim([0, max(np.append(group_fnrs,
    #                                  100 * false_negative_rate(labels,
    #                                                            predictions)))
    #                 + 1])
    # ax[1].grid(axis='y', linestyle='--')
    group_fprs = [
    rate * 100 if rate is not None else 0 for rate in group_false_positive_rates(labels, predictions, groups)
        ]
    overall_fpr = 100 * false_positive_rate(labels, predictions)
    ax[0].bar(range(num_groups + 1), np.append(group_fprs, overall_fpr),color=cb_palette)
    ax[0].set_xticks(range(num_groups + 1))
    ax[0].set_xticklabels(group_names + ['overall'])
    ax[0].set_xlabel("Groups")
    ax[0].set_ylabel("False Positive Rate (%)")
    ax[0].set_ylim([0, max(np.append(group_fprs, overall_fpr)) + 1])
    ax[0].grid(axis='y', linestyle='--')

    # Process and scale false negative rates per group
    group_fnrs = [
        rate * 100 if rate is not None else 0 for rate in group_false_negative_rates(labels, predictions, groups)
    ]
    overall_fnr = 100 * false_negative_rate(labels, predictions)
   
    # Plot False Negative Rates
    ax[1].bar(range(num_groups + 1), np.append(group_fnrs, overall_fnr), color=cb_palette)
    ax[1].set_xticks(range(num_groups + 1))
    ax[1].set_xticklabels(group_names + ['overall'])
    ax[1].set_xlabel("Groups")
    ax[1].set_ylabel("False Negative Rate (%)")
    ax[1].set_ylim([0, max(np.append(group_fnrs, overall_fnr)) + 1])
    ax[1].grid(axis='y', linestyle='--')


    fig.suptitle('{} accuracy = {}%'.format(title,
                                            np.round(100 *
                                                     accuracy(labels,
                                                              predictions),
                                                     2)), x=0.5, y=1)
    plt.tight_layout()
    plt.show()
    return None
