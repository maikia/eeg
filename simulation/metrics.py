import numpy as np


def get_true_false(true_signal, pred_signal):
    # given true and predicted signal 1d array of 0s and 1s
    # return true_positive, true_negative, false_positive, false_negative
    unique, counts = np.unique(true_signal - pred_signal, return_counts=True)
    try:
        false_positive = counts[np.where(unique == -1)][0]
    except IndexError:
        false_positive = 0

    try:
        true_negative = counts[np.where(unique == 1)][0]
    except IndexError:
        true_negative = 0

    unique, counts = np.unique(true_signal + pred_signal, return_counts=True)
    try:
        true_positive = counts[np.where(unique == 2)][0]
    except IndexError:
        true_positive = 0

    try:
        false_negative = counts[np.where(unique == 0)][0]
    except IndexError:
        false_negative = 0

    assert len(true_signal) == (true_positive + true_negative +
                                false_positive + false_negative)
    return true_positive, true_negative, false_positive, false_negative


def calc_froc(y_true, y_score):
    """compute Free response receiver operating characteristic curve (FROC)
    Note: this implementation is restricted to the binary classification
    task.

    Parameters
    ----------
    y_true : array, shape = [n_samples x n_classes]
             true binary labels
    y_score : array, shape = [n_samples x n_classes]
             target scores: probability estimates of the positive class,
             confidence values
    Returns
    -------
    ts : array
        sensitivity: true positive normalized by sum of all true
        positives
    tfp : array
        false positive: False positive rate divided by length of
        y_true
    thresholds : array, shape = [>2]
        Thresholds on y_score used to compute ts and tfp.
        *Note*: Since the thresholds are sorted from low to high values,
        they are reversed upon returning them to ensure they
        correspond to both fpr and tpr, which are sorted in reversed order
        during their calculation.

    References
    ----------
    http://www.devchakraborty.com/Receiver%20operating%20characteristic.pdf
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3679336/pdf/nihms458993.pdf
    """

    n_samples, n_sources = y_true.shape
    classes = np.unique(y_true)

    n_pos = float(np.sum(y_true == classes[1]))  # nb of true positive

    y_true = np.ravel(y_true)
    y_score = np.ravel(y_score)

    # FROC only for binary classification
    if classes.shape[0] != 2:
        raise ValueError("FROC is defined for binary classification only")

    thresholds = np.unique(y_score)

    # sensitivity: true positive normalized by sum of all true
    # positives
    ts = np.zeros(thresholds.size, dtype=np.float)
    # false positive: False positives rate divided by length of y_true
    tfp = np.zeros(thresholds.size, dtype=np.float)

    idx = 0

    signal = np.c_[y_score, y_true]
    sorted_signal = signal[signal[:, 0].argsort(), :][::-1]
    for score, value in sorted_signal:
        t_est = sorted_signal[:, 0] >= score

        tps, _, fps, _ = get_true_false(sorted_signal[:, 1], t_est)

        ts[idx] = tps
        tfp[idx] = fps

        idx += 1

    tfp = tfp / n_samples
    ts = ts / n_pos
    return ts, tfp, thresholds[::-1]


def calc_afroc(y_true, y_score):
    """compute Alternative Free response receiver operating characteristic
    curve (FROC)
    Note: this implementation is restricted to the binary classification
    task.

    Parameters
    ----------
    y_true : array, shape = [n_samples x n_classes]
             true binary labels
    y_score : array, shape = [n_samples x n_classes]
             target scores: probability estimates of the positive class,
             confidence values
    Returns
    -------
    ts : array
        sensitivity: true positive normalized by sum of all true
        positives
    fpf : array
        false positive fraction
    thresholds : array, shape = [>2]
        Thresholds on y_score used to compute ts and tfp.
        *Note*: Since the thresholds are sorted from low to high values,
        they are reversed upon returning them to ensure they
        correspond to both fpr and tpr, which are sorted in reversed order
        during their calculation.

    References
    ----------
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3679336/pdf/nihms458993.pdf
    """

    ts, tfp, thresholds = froc_score(y_true, y_score)
    fpf = 1 - np.e**(-tfp)
    return ts, fpf, thresholds


def froc_score(y_true, y_score):
    ''' Compute Area Under the Free response receiver operating characteristic
        Curve (FROC AUC) from prediction scores
    '''

    ts, tfp, thresholds = calc_froc(y_true, y_score)

    # Compute the area using the composite trapezoidal rule.
    area = np.trapz(y=ts, x=tfp)
    return area


def afroc_score(y_true, y_score):
    ''' Compute Area Under the Alternative Free response receiver operating
        characteristic Curve (FROC AUC) from prediction scores

        True Positive fraction vs. false positive fraction (FPF) termed the
        alternative FROC (AFROC).
        Since the AFROC curve is completelycontained within the unit square,
        since both axes are probabilities analogous to the area under the ROC
        curve, the area under the AFROC be used as a figure-of-merit for FROC
        performance
        [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3679336/pdf/nihms
            458993.pdf

    '''
    ts, fpf, thresholds = calc_froc(y_true, y_score)

    # Compute the area using the composite trapezoidal rule.
    area = np.trapz(y=ts, x=fpf)
    return area

# def plot_froc():
#     """Plots the FROC curve (Free response receiver operating
#        characteristic curve)
#     """
#     threshs = thresholds[::-1]
#     plt.figure()
#     plt.plot(tfp, ts, 'ro')
#     plt.xlabel('false positives per sample', fontsize=12)
#     plt.ylabel('sensitivity', fontsize=12)
#     thresh = threshs(5).astype(str)[::100]
#     for fp, ts, t in zip(tfp, ts, thresh):
#         plt.text(fp, ts - 0.025, t, rotation=45)
#     plt.title('FROC, max parcels: ' + str(self.n_sources_))
