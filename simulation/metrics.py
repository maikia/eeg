import numpy as np


def froc_score(y_true, y_score):
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
    # XXX : what you implement is ROC-AUC then?
    if classes.shape[0] != 2:
        raise ValueError("FROC is defined for binary classification only")

    thresholds = np.unique(y_score)

    # sensitivity: true positive normalized by sum of all true
    # positives
    ts = np.zeros(thresholds.size, dtype=np.float)
    # false positive: False positive rate divided by length of y_true
    tfp = np.zeros(thresholds.size, dtype=np.float)

    idx = 0

    signal = np.c_[y_score, y_true]
    sorted_signal = signal[signal[:, 0].argsort(), :][::-1]
    for score, value in sorted_signal:
        t_est = sorted_signal[:, 0] >= score

        # false positives for this score (threshold)
        unique, counts = np.unique(sorted_signal[:, 1] - t_est,
                                   return_counts=True)
        try:
            fps = counts[np.where(unique == -1)][0]
        except IndexError:
            fps = 0
        # true positives for this score (threshold)
        unique, counts = np.unique(sorted_signal[:, 1] + t_est,
                                   return_counts=True)
        try:
            tps = counts[np.where(unique == 2)][0]
        except IndexError:
            tps = 0

        ts[idx] = tps
        tfp[idx] = fps

        idx += 1

    tfp = tfp / n_samples
    ts = ts / n_pos

    return ts, tfp, thresholds[::-1]

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
