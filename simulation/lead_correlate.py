import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy import linalg

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
# from sklearn.utils.validation import check_X_y, check_array
# from sklearn.utils.multiclass import unique_labels
# from sklearn.metrics import euclidean_distances


class LeadCorrelate(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
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

        # threshold is selected at 2% of all true
        sorted_mask = np.sort(df_mask_flat)
        thresh_idx = int(len(df_mask_flat)*(0.55))
        self.threshold_ = sorted_mask[thresh_idx]

        # check what is max possible number of parcels
        true_per_row = np.sum(y, axis=1).astype(int)
        self.max_active_sources_ = np.unique(true_per_row)[-1]
        self.n_sources_ = y.shape[1]

        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ predicting function.

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
        corr_poss = corr >= self.threshold_  # boolen
        # leave only max_active_sources_ parcels

        for idx in range(0, len(corr_poss)):
            # check if more than 0 and less than max_active_sources_
            if sum(corr_poss[idx, :]) > self.max_active_sources_:
                # take only self.max_active_sources_ highest possible corr
                corr[idx, np.logical_not(corr_poss[idx, :])] = 0
                y_pred[idx, np.argsort(
                       corr[idx, :])[-self.max_active_sources_:]] = 1
            elif sum(corr_poss[idx, :]) < 1:
                # take a single highest possible corr
                max_corr_idx = np.argsort(corr[idx, :])[-1]
                y_pred[idx, max_corr_idx] = 1
            else:
                # leave corr as selected by corr_poss
                y_pred[idx, corr_poss[idx, :]] = 1
        print(np.unique(np.sum(y_pred, 1), return_counts=True))
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
        sensitivity_list_treshold, FPavg_list_treshold, threshold_list = computeFROC(
            proba_map = np.array(self.decision_function(X)), ground_truth=y,
            nbr_of_thresholds=40, allowedDistance=0, range_threshold=[0.,1.])

         # TODO: remove:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(FPavg_list_treshold, sensitivity_list_treshold, 'bo',
                 label='other')
        plt.xlabel('average number of false positives (FP) per scan', fontsize=12)
        plt.ylabel('sensitivity', fontsize=12)
        # thresh = threshs.round(5).astype(str)[::400]
        # for fp, ts, t in zip(tfp[::400], ts[::400], thresh):
        #     plt.text(fp, ts - 0.025, t, rotation=45)

        ts, tfp, thresholds = froc_score(y, self.decision_function(X))
        plt.plot(tfp, ts, 'ro', label='ours')
        plt.xlabel('average number of false positives (FP) per scan', fontsize=12)
        plt.ylabel('sensitivity', fontsize=12)
        plt.legend()
        # thresh = threshs.round(5).astype(str)[::400]
        # for fp, ts, t in zip(tfp[::400], ts[::400], thresh):
        #     plt.text(fp, ts - 0.025, t, rotation=45)
        plt.title('FROC ')
        plt.subplot(2,1,2)
        plt.title('thresholds')
        plt.plot(ts, thresholds, 'ro')
        plt.plot(sensitivity_list_treshold, threshold_list, 'bo')
        plt.xlabel('sensitivity')
        plt.ylabel('thresholds')
        plt.tight_layout()
        plt.savefig('froc_compare.png')
        plt.show()

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

            corr = pd.DataFrame(
                np.abs(L.T.dot(x))).groupby(parcel_indices).max().transpose()
            if idx:
                correlation = correlation.append(corr)  # please run flake8
            else:
                correlation = corr

        correlation.index = range(n_samples)

        # we don't use the 0 index
        # TODO: remove passing L.idx = 0 all together
        if 0 in correlation:
            correlation = correlation.drop(columns=0)
        return correlation


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
        t = value
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

    threshs = thresholds[::-1]

    return ts, tfp, thresholds[::-1]


def computeConfMatElements(thresholded_proba_map, ground_truth, allowedDistance):
    
    if allowedDistance == 0 and type(ground_truth) == np.ndarray:
        P = np.count_nonzero(ground_truth)
        TP = np.count_nonzero(thresholded_proba_map*ground_truth)
        FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map*ground_truth))    
    else:
    
        #reformat ground truth to a list  
        if type(ground_truth) == np.ndarray:
            #convert ground truth binary map to list of coordinates
            labels, num_features = ndimage.label(ground_truth)
            list_gt = ndimage.measurements.center_of_mass(ground_truth, labels, range(1,num_features+1))   
        elif type(ground_truth) == list:        
            list_gt = ground_truth        
        else:
            raise ValueError('ground_truth should be either of type list or ndarray and is of type ' + str(type(ground_truth)))
        
        #reformat thresholded_proba_map to a list
        labels, num_features = ndimage.label(thresholded_proba_map)
        list_proba_map = ndimage.measurements.center_of_mass(thresholded_proba_map, labels, range(1,num_features+1)) 
         
        #compute P, TP and FP  
        P,TP,FP = computeAssignment(list_proba_map,list_gt,allowedDistance)
                                 
    return P,TP,FP


def computeFROC(proba_map, ground_truth, allowedDistance, nbr_of_thresholds=40,
                range_threshold=None):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}; or list of coordinates
    #allowedDistance: Integer. euclidian distance distance in pixels to consider a detection as valid (anisotropy not considered in the implementation)  
    #nbr_of_thresholds: Interger. number of thresholds to compute to plot the FROC
    #range_threshold: list of 2 floats. Begining and end of the range of thresholds with which to plot the FROC  
    #OUTPUTS
    #sensitivity_list_treshold: list of average sensitivy over the set of images for increasing thresholds
    #FPavg_list_treshold: list of average FP over the set of images for increasing thresholds
    #threshold_list: list of thresholds

    #rescale ground truth and proba map between 0 and 1
    proba_map = proba_map.astype(np.float32)
    proba_map = (proba_map - np.min(proba_map)) / (np.max(proba_map) - np.min(proba_map))
    if type(ground_truth) == np.ndarray:
        #verify that proba_map and ground_truth have the same shape
        if proba_map.shape != ground_truth.shape:
            raise ValueError('Error. Proba map and ground truth have different shapes.')
        
        ground_truth = ground_truth.astype(np.float32)    
        ground_truth = (ground_truth - np.min(ground_truth)) / (np.max(ground_truth) - np.min(ground_truth))
    
    #define the thresholds
    if range_threshold == None:
        threshold_list = (np.linspace(np.min(proba_map),np.max(proba_map),nbr_of_thresholds)).tolist()
    else:
        threshold_list = (np.linspace(range_threshold[0],range_threshold[1],nbr_of_thresholds)).tolist()
    
    sensitivity_list_treshold = []
    FPavg_list_treshold = []
    #loop over thresholds
    for threshold in threshold_list:
        sensitivity_list_proba_map = []
        FP_list_proba_map = []
        #loop over proba map
        for i in range(len(proba_map)):
                       
            #threshold the proba map
            thresholded_proba_map = np.zeros(np.shape(proba_map[i]))
            thresholded_proba_map[proba_map[i] >= threshold] = 1
            
            #save proba maps
#            imageio.imwrite('thresholded_proba_map_'+str(threshold)+'.png', thresholded_proba_map)                   
                   
            #compute P, TP, and FP for this threshold and this proba map
            P,TP,FP = computeConfMatElements(thresholded_proba_map, ground_truth[i], allowedDistance)       
            
            #append results to list
            FP_list_proba_map.append(FP)
            #check that ground truth contains at least one positive
            try:
                if (type(ground_truth) == np.ndarray and len(np.nonzero(ground_truth)[0]) > 0) or (type(ground_truth) == list and len(ground_truth) > 0):
                    sensitivity_list_proba_map.append(TP*1./P)
            except:
                import pdb; pdb.set_trace()
        
        #average sensitivity and FP over the proba map, for a given threshold
        sensitivity_list_treshold.append(np.mean(sensitivity_list_proba_map))
        FPavg_list_treshold.append(np.mean(FP_list_proba_map))    
        
    return sensitivity_list_treshold, FPavg_list_treshold, threshold_list
