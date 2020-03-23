import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy import linalg

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_selection import f_regression
from sklearn.utils.validation import check_is_fitted
# from sklearn.utils.validation import check_X_y, check_array
# from sklearn.utils.multiclass import unique_labels
# from sklearn.metrics import euclidean_distances


class LeadCorrelate(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, lead_field, parcel_indices_leadfield):
        self.lead_field = lead_field
        self.parcel_indices_leadfield = parcel_indices_leadfield

    def fit(self, X, y):
        """
        """
        df = self.decision_function(X)
        df = np.array(df)
        assert df.shape == y.shape
        df_mask_flat = df.ravel()[y.ravel() == 1]
        self.threshold_ = df_mask_flat.min()

        # check what is max possible number of parcels
        true_per_row = np.sum(y, axis=1).astype(int)
        self.max_active_sources_ = np.unique(true_per_row)[-1]
        self.n_sources_ = y.shape[1]

        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        n_samples, _ = X.shape
        y_pred = np.zeros((n_samples, self.n_sources_), dtype=int)

        check_is_fitted(self, 'n_sources_')

        corr = self.decision_function(X)
        corr = np.array(corr)

        # take values higher than threshold
        corr_poss = corr >= self.threshold_
        more, less, good = 0, 0, 0
        # leave only max_active_sources_ parcels
        #corr_argsort = np.argsort(corr) # sorts from the highest indices
        for idx in range(0, len(corr_poss)):
            # check if more than 0 and less than max_active_sources_
            if sum(corr_poss[idx, :]) > self.max_active_sources_:
                # take only self.max_active_sources_ highest possible corr
                corr[idx, np.logical_not(corr_poss[idx, :])] = 0
                y_pred[idx, np.argsort(
                       corr[idx, :])[-self.max_active_sources_:]] = 1
                more += 1
            elif sum(corr_poss[idx, :]) < 1:
                # take a single highest possible corr
                max_corr_idx = np.argsort(corr[idx, :])[-1]
                y_pred[idx, max_corr_idx] = 1
                less += 1
            else:
                # leave corr as selected by corr_poss
                good += 1
                y_pred[idx, corr_poss[idx, :]] = 1

        return y_pred

    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            average number of errors per sample (the more the worse)
        """

        return self.froc_score(y, self.decision_function(X))

    def decision_function(self, X):
        """ Computes the correlation of the data with the lead field
        Args:
            X: data
        Returns:
            decision: correlation of the signal from each parcel with the given
            data for each sample, dtype = DataFrame
        """
        n_samples, _ = X.shape
        L = self.lead_field
        L = L / linalg.norm(L, axis=0)  # normalize leadfield column wise
        parcel_indices = self.parcel_indices_leadfield

        for idx in range(n_samples):
            x = X.iloc[idx]
            x = x / linalg.norm(x)  # normalize x to take correlations

            corr = pd.DataFrame(np.abs(L.T.dot(x))).groupby(
                         parcel_indices).max().transpose()
            if idx:
                correlation = correlation.append(corr)
            else:
                correlation = corr

        correlation.index = range(n_samples)

        # we don't use the 0 index
        # TODO: remove passing L.idx = 0 all together
        if 0 in correlation:
            correlation = correlation.drop(columns = 0)
        return correlation

    def froc_score(self, y_true, y_score):
        # TODO: finish up

        y_true = np.ravel(y_true)
        y_score = np.ravel(y_score)
        classes = np.unique(y_true)

        # FROC only for binary classification
        if classes.shape[0] != 2:
            raise ValueError("FROC is defined for binary classification only")

        thresholds = np.unique(y_score)
        neg_value, pos_value = classes[0], classes[1]

        # total sensitivity: true positive normalized by sum of True
        ts = np.zeros(thresholds.size, dtype=np.float)
        # total false positive: False positive rate divided by lenght of y
        tfp = np.zeros(thresholds.size, dtype=np.float)

        current_pos_count = current_neg_count = sum_pos = sum_neg = idx = 0

        _pos = float(np.sum(y_true == classes[1]))  # nb of true positive
        n_neg = float(np.sum(y_true == classes[0]))  # nb of true negative

        signal = np.c_[y_score, y_true]
        sorted_signal = signal[signal[:, 0].argsort(), :][::-1]
        last_score = sorted_signal[0][0]
        for score, value in sorted_signal:
            if score == last_score:
                if value == pos_value:
                    current_pos_count += 1
                else:
                    current_neg_count += 1
                print('same score')
            else:
                import pdb; pdb.set_trace()
                ts[idx] = (sum_pos + current_pos_count) / n_pos
                tfp[idx] = (sum_neg + current_neg_count) / n_neg

                tpr[idx] = (sum_pos + current_pos_count) / n_pos
                fpr[idx] = (sum_neg + current_neg_count) / n_neg
                sum_pos += current_pos_count
                sum_neg += current_neg_count
                current_pos_count = 1 if value == pos_value else 0
                current_neg_count = 1 if value == neg_value else 0
                idx += 1
                last_score = score
        else:
            tpr[-1] = (sum_pos + current_pos_count) / n_pos
            fpr[-1] = (sum_neg + current_neg_count) / n_neg



            # for thres in thresholds:
            import pdb; pdb.set_trace()
            unique, counts = np.unique(y - y_pred, return_counts=True)
            total_FPs = counts[np.where(unique == -1)][0]
            # total_TN = counts[np.where(unique == 1)][0]
            unique, counts = np.unique(y + y_pred, return_counts=True)
            total_TP = counts[np.where(unique == 2)][0]

            sensitivity = total_TP / np.sum(y)
            false_positives = total_FPs / len(y)
            total_sensitivity.append(sensitivity)
            total_false_positives.append(false_positives)
        self.total_FPs_ = total_false_positives
        self.total_sensitivity_ = total_sensitivity
        self.thresholds_ = thresholds

        return score

    def computeFROC(self, X, y):
        """Generates the data required for plotting the FROC curve

        Args:
            X: data
            y: true classified
        Returns:
            total_FPs:  A list containing the average number of false positives
            per image for different thresholds

            total_sensitivity:  A list containig overall sensitivity of the
            system for different thresholds
        """
        # LL ≡ lesion localization, i.e., a (1)
        # lesion marked to within an agreed upon accuracy.
        # NL ≡ non-lesion localization, i.e., the mark is not close to any
        # lesion.
        # LLF ≡ lesion localization fraction ≡ # LL divided by total number of
        # lesions (0 ≤ LLF ≤ 1).
        # NLF ≡ non-lesion localization fraction ≡ # NL divided by total number
        # of images (0 ≤ NLF); note the lack of an upper bound
        thresholds = np.linspace(-0.1, -0.000001, 15)
        total_sensitivity = []
        total_false_positives = []
        for thres in thresholds:  # -0.00025)
            self.threshold = thres
            y_pred = self.predict(X)
            unique, counts = np.unique(y - y_pred, return_counts=True)
            total_FPs = counts[np.where(unique == -1)][0]
            # total_TN = counts[np.where(unique == 1)][0]
            unique, counts = np.unique(y + y_pred, return_counts=True)
            total_TP = counts[np.where(unique == 2)][0]

            sensitivity = total_TP / np.sum(y)
            false_positives = total_FPs / len(y)
            total_sensitivity.append(sensitivity)
            total_false_positives.append(false_positives)
        self.total_FPs_ = total_false_positives
        self.total_sensitivity_ = total_sensitivity
        self.thresholds_ = thresholds
        return total_false_positives, total_sensitivity, thresholds

    def plotFROC(self):
        """Plots the FROC curve (Free response receiver operating
           characteristic curve)
        """

        plt.figure()
        plt.plot(self.total_FPs_, self.total_sensitivity_, color='#000000')
        plt.plot(self.total_FPs_, self.total_sensitivity_, 'ro')
        plt.xlabel('total false positives', fontsize=12)
        plt.ylabel('total sensitivity', fontsize=12)
        thresh = self.thresholds_.round(5).astype(str)
        for fp, ts, t in zip(self.total_FPs_, self.total_sensitivity_, thresh):
            plt.text(fp, ts-0.025, t, rotation=45)
        plt.title('FROC, max parcels: ' + str(self.n_sources_))
        import pdb; pdb.set_trace()




'''
    
def predict(self, X):
    return self.decision_function(X) >= self.threshold_

def score(self, X, y):
    return froc_score(y, self.decision_function(X))
    

def froc_score(y_true, y_pred):
    ...
    
froc_score(y_test, clf.decision_function(X_test))

cross_val_score(clf, X, y, cv=3)
'''