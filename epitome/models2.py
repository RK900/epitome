r"""
======
Models
======
.. currentmodule:: epitome.models

.. autosummary::
  :toctree: _generate/

  VariationalPeakModel
  VLP
"""


from epitome import *
import tensorflow as tf
import tensorflow_probability as tfp

from .functions import *
from .constants import *
from .generators import *
from .metrics import *
import numpy as np

import tqdm
import logging

# for saving model
import pickle
import os
from operator import itemgetter
import time

import ray

# ray.shutdown()

#######################################################################
#################### Variational Peak Model ###########################
#######################################################################

class VariationalPeakModel():
    """ Model for learning from ChIP-seq peaks.
    Modeled from `this Bayesian Neural Network <https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py>`_.
    """

    def __init__(self,
                 assays,
                 test_celltypes = [],
                 matrix = None,
                 assaymap = None,
                 cellmap = None,
                 debug = False,
                 batch_size = 64,
                 shuffle_size = 10,
                 prefetch_size = 10,
                 l1=0.,
                 l2=0.,
                 lr=1e-3,
                 radii=[1,3,10,30],
                 similarity_assays = ['DNase'],
                 train_indices = None,
                 data = None,
                 checkpoint = None):
        """
        Initializes Peak Model

        Args:
            :param assays: list of assays to train model on
            :param test_celltypes: list of cell types to hold out for test. Should be in cellmap
            :param matrix: numpy matrix of indices mapping assay and cell to index in data
            :param assaymap: map of assays mapping assay name to row in matrix
            :param cellmap: map of cell types mapping cell name to column in matrix
            :param debug: used to print out intermediate validation values
            :param batch_size: batch size (default is 64)
            :param shuffle_size: data shuffle size (default is 10)
            :param prefetch_size: data prefetch size (default is 10)
            :param l1: l1 regularization (default is 0)
            :param l2: l2 regularization (default is 0)
            :param lr: lr (default is 1e-3)
            :param radii: radius of DNase-seq to consider around a peak of interest (default is [1,3,10,30])
            :param train_indices: option numpy array of indices to train from data[Dataset.TRAIN]
            :param data: data loaded from datapath. This option is mostly for testing, so users dont have to load in data for
            each model.
        """

        logging.getLogger("tensorflow").setLevel(logging.INFO)

        # user can provide their own assaymap information.
        if assaymap is not None:
            assert matrix is not None and cellmap is not None, "matrix, cellmap, and assaymap must all be set"
        if cellmap is not None:
            assert matrix is not None and assaymap is not None, "matrix, cellmap, and assaymap must all be set"
        if matrix is not None:
            assert assaymap is not None and cellmap is not None, "matrix, cellmap, and assaymap must all be set"

        # get cell lines to train on if not specified
        if assaymap is None:
            # assays should include similarity assays and predicted assays
            assays = list(set(assays + similarity_assays))

            # get list of TFs that have minimum number of cell lines
            matrix, cellmap, assaymap = get_assays_from_feature_file(eligible_assays = assays)
            assert len(assays) == len(list(assaymap))


        assert (set(test_celltypes) < set(list(cellmap))), \
                "test_celltypes %s must be subsets of available cell types %s" % (str(test_celltypes), str(list(cellmap)))

        # get evaluation cell types by removing any cell types that would be used in test
        self.eval_cell_types = list(cellmap).copy()
        self.test_celltypes = test_celltypes
        [self.eval_cell_types.remove(test_cell) for test_cell in self.test_celltypes]

        data_path = GET_DATA_PATH()

        # load in data, if the user has not specified it
        if data is not None:
            self.data = data
        else:
            self.data = load_epitome_data(data_path)



        self.regionsFile = os.path.join(data_path, POSITIONS_FILE)

        input_shapes, output_shape, self.train_iter = generator_to_tf_dataset(load_data(self.data[Dataset.TRAIN],
                                                self.eval_cell_types,
                                                self.eval_cell_types,
                                                matrix,
                                                assaymap,
                                                cellmap,
                                                similarity_assays = similarity_assays,
                                                radii = radii, mode = Dataset.TRAIN),
                                                batch_size, shuffle_size, prefetch_size)

        _, _,            self.valid_iter = generator_to_tf_dataset(load_data(self.data[Dataset.VALID],
                                                self.eval_cell_types,
                                                self.eval_cell_types,
                                                matrix,
                                                assaymap,
                                                cellmap,
                                                similarity_assays = similarity_assays,
                                                radii = radii, mode = Dataset.VALID),
                                                batch_size, 1, prefetch_size)

        # can be empty if len(test_celltypes) == 0
        if len(self.test_celltypes) > 0:
            _, _,            self.test_iter = generator_to_tf_dataset(load_data(self.data[Dataset.TEST],
                                                   self.test_celltypes,
                                                   self.eval_cell_types,
                                                   matrix,
                                                   assaymap,
                                                   cellmap,
                                                   similarity_assays = similarity_assays,
                                                   radii = radii, mode = Dataset.TEST),
                                                   batch_size, 1, prefetch_size)

        self.l1, self.l2 = l1, l2
        self.lr = lr
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.shuffle_size = shuffle_size
        self.optimizer =tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)

        self.num_outputs = output_shape[0]
        self.num_inputs = input_shapes

        # set self
        self.radii = radii
        self.similarity_assays = similarity_assays
        self.debug = debug
        self.assaymap = assaymap
        self.test_celltypes = test_celltypes
        self.matrix = matrix
        self.assaymap= assaymap
        self.cellmap = cellmap
        self.predict_assays = list(self.assaymap)
        [self.predict_assays.remove(i) for i in self.similarity_assays]
        self.model = self.create_model()

    def get_weight_parameters(self):
        """
        Extracts weight posterior statistics for layers with weight distributions.
        :param model: keras model

        :return triple of layer names, weight means for each layer and stddev for each layer.
        """

        names = []
        qmeans = []
        qstds = []
        for i, layer in enumerate(self.model.layers):
            try:
                q = layer.kernel_posterior
            except AttributeError:
                continue
            names.append("Layer {}".format(i))
            qmeans.append(q.mean())
            qstds.append(q.stddev())

        return (names, qmeans, qstds)

    def save(self, checkpoint_path):
        """
        Saves model.
        :param checkpoint_path: string file path to save model to.
        """
        weights_path = os.path.join(checkpoint_path, "weights.h5")
        meta_path = os.path.join(checkpoint_path, "model_params.pickle")

        # save keras model weights
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        file = h5py.File(weights_path, 'w')
        weight = self.model.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()

        # save model params to pickle file
        dict_ = {'assays': list(self.assaymap),
                         'test_celltypes':self.test_celltypes,
                         'matrix':self.matrix,
                         'assaymap':self.assaymap,
                         'cellmap':self.cellmap,
                         'debug': self.debug,
                         'batch_size':self.batch_size,
                         'shuffle_size':self.shuffle_size,
                         'prefetch_size':self.prefetch_size,
                         'radii':self.radii,
                         'similarity_assays': self.similarity_assays}

        fileObject = open(meta_path,'wb')
        pickle.dump(dict_,fileObject)
        fileObject.close()

    def body_fn(self):
        raise NotImplementedError()

    def g(self, p, a=1, B=0, y=1):
        """ Normalization Function. Normalizes loss w.r.t. label proportion.

        Constraints:
         1. g(p) = 1 when p = 1
         2. g(p) = a * p^y + B, where a, y and B are hyperparameters
        """
        return a * tf.math.pow(p, y) + B

    def loss_fn(self, y_true, y_pred, weights):
        """
        Loss function for Epitome. Calculates the weighted sigmoid cross entropy
        between logits and true values.

        Args:
          :param y_true: true binary values
          :param y_pred: logits
          :param weights: binary weights whether the true values exist for
          a given cell type/assay combination

        Returns:
          Loss summed over all TFs and genomic loci.
        """
        # weighted sum of cross entropy for non 0 weights
        # Reduction method = Reduction.SUM_BY_NONZERO_WEIGHTS
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_true,
                                                        y_pred,
                                                        weights = weights,
                                                        reduction = tf.compat.v1.losses.Reduction.NONE)

        return tf.math.reduce_sum(loss, axis=0)

    def train(self, num_steps):
        """ Trains an Epitome model for num_steps iterations.

        Args:
          :param num_steps: number of iterations to train for

        """

        tf.compat.v1.logging.info("Starting Training")

        @tf.function
        def train_step(f):
            features = f[:-2]
            labels = f[-2]
            weights = f[-1]

            with tf.GradientTape() as tape:

                logits = self.model(features, training=True)
                kl_loss = tf.reduce_sum(self.model.losses)
                neg_log_likelihood = self.loss_fn(labels, logits, weights)
                elbo_loss = neg_log_likelihood + kl_loss

            gradients = tape.gradient(elbo_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return elbo_loss, neg_log_likelihood, kl_loss

        @tf.function
        def loopiter():
            for step, f in enumerate(self.train_iter):
                loss = train_step(f)

                if step % 100 == 0:
                  tf.compat.v1.logging.info(str(step) + " " + str(tf.reduce_mean(loss[0])) +
                                            str(tf.reduce_mean(loss[1])) +
                                            str(tf.reduce_mean(loss[2])))

                  if (self.debug):
                      tf.compat.v1.logging.info("On validation")
                      _, _, _, _, _ = self.test(40000, log=False)
                      tf.compat.v1.logging.info("")

                if step > num_steps:
                  break

        loopiter()

    def test(self, num_samples, mode = Dataset.VALID, calculate_metrics=False):
        """
        Tests model on valid and test dataset handlers.
        """

        if (mode == Dataset.VALID):
            handle = self.valid_iter # for standard validation of validation cell types

        elif (mode == Dataset.TEST and len(self.test_celltypes) > 0):
            handle = self.test_iter # for standard validation of validation cell types
        else:
            raise Exception("No data exists for %s. Use function test_from_generator() if you want to create a new iterator." % (mode))
        
        # s_all = time.time_ns()
        x = self.run_predictions(num_samples, handle, calculate_metrics)
        # e_all = time.time_ns()

        # print('Total run_predictions time: %i' % (e_all - s_all))

        return x

    def test_from_generator(self, num_samples, ds, calculate_metrics=True):
        """
        Runs test given a specified data generator
        :param num_samples: number of samples to test
        :param ds: tensorflow dataset, created by dataset_to_tf_dataset
        :param cell_type: cell type to test on. Used to generate holdout indices.

        :return predictions
        """
        return self.run_predictions(num_samples, ds, calculate_metrics)

    def eval_vector(self, data, matrix, indices):
        """
        Evaluates a new cell type based on its chromatin (DNase or ATAC-seq) vector, as well
        as any other similarity assays (acetylation, methylation, etc.). len(vector) should equal
        the data.shape[1]
        :param data: data to build features from
        :param matrix: matrix of 0s/1s, where # rows match # similarity assays in model
        :param indices: indices of vector to actually score. You need all of the locations for the generator.

        :return predictions for all factors
        """
        tmps = time.time_ns()
        input_shapes, output_shape, ds = generator_to_tf_dataset(load_data(data,
                 self.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                 self.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 self.matrix,
                 self.assaymap,
                 self.cellmap,
                 radii = self.radii,
                 mode = Dataset.RUNTIME,
                 similarity_matrix = matrix,
                 similarity_assays = self.similarity_assays,
                 indices = indices), self.batch_size, 1, self.prefetch_size)
        tmpe = time.time_ns()
        print('generator: %i' % (tmpe-tmps))
        num_samples = len(indices)

        tmps = time.time_ns()
        results = self.run_predictions(num_samples, ds, calculate_metrics = False, samples = 1)
        tmpe = time.time_ns()
        print('run_predictions: %i' % (tmpe - tmps))

        return results['preds_mean'], results['preds_std']
    
    def eval_vector_2(self, data, matrix, indices):
        """
        Evaluates a new cell type based on its chromatin (DNase or ATAC-seq) vector, as well
        as any other similarity assays (acetylation, methylation, etc.). len(vector) should equal
        the data.shape[1]
        :param data: data to build features from
        :param matrix: matrix of 0s/1s, where # rows match # similarity assays in model
        :param indices: indices of vector to actually score. You need all of the locations for the generator.

        :return predictions for all factors
        """
        tmps = time.time_ns()
        input_shapes, output_shape, ds = generator_to_tf_dataset(load_data(data,
                 self.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                 self.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 self.matrix,
                 self.assaymap,
                 self.cellmap,
                 radii = self.radii,
                 mode = Dataset.RUNTIME,
                 similarity_matrix = matrix,
                 similarity_assays = self.similarity_assays,
                 indices = indices), self.batch_size, 1, self.prefetch_size)
        tmpe = time.time_ns()
        print('generator: %i' % (tmpe-tmps))
        num_samples = len(indices)

        tmps = time.time_ns()
        results = self.run_predictions(num_samples, ds, calculate_metrics = False, samples = 1)
        tmpe = time.time_ns()
        print('run_predictions: %i' % (tmpe - tmps))

        return results['preds_mean']


    def _predict(self, numpy_matrix):
        """
        Run predictions on a numpy matrix. Size of numpy_matrix should be # examples by features.
        This function is mostly used for testing, as it requires the user to pre-generate the
        features using the generator function in generators.py.
        """

        inv_assaymap = {v: k for k, v in self.assaymap.items()}

        @tf.function
        def predict_step(inputs):

            # get the shapes for the cell type specific features.
            # they are not even, because some cells have missing data.
            cell_lens = [i.shape[-1] for i in self.train_iter.element_spec[:-2]]

            # split matrix inputs into tuple of cell line specific features
            split_inputs = tf.split(inputs.astype(np.float32), cell_lens, axis=1)

            # predict
            tmp = self.model(split_inputs)
            y_pred = tf.sigmoid(tmp)
            return y_pred

        return predict_step(numpy_matrix)

    def run_predictions(self, num_samples, iter_, calculate_metrics = True, samples = 50):
        """
        Runs predictions on num_samples records
        :param num_samples: number of samples to test
        :param iter_: output of self.sess.run(generator_to_one_shot_iterator()), handle to one shot iterator of records
        :param log: if true, logs individual factor accuracies

        :return preds, truth, assay_dict, auROC, auPRC, False
            preds = predictions,
            truth = actual values,
            sample_weight: 0/1 weights on predictions.
            assay_dict = if log=True, holds predictions for individual factors
            auROC = average macro area under ROC for all factors with truth values
            auPRC = average area under PRC for all factors with truth values
        """

        inv_assaymap = {v: k for k, v in self.assaymap.items()}

        batches = int(num_samples / self.batch_size)+1

        # empty arrays for concatenation
        truth = []
        preds_mean = []
        preds_std = []
        sample_weight = []

        @tf.function
        def predict_step(inputs_b):

            # sample n times by tiling batch by rows, running
            # predictions for each row
            inputs_tiled = [tf.tile(i, (samples, 1)) for i in inputs_b]
            tmp = self.model(inputs_tiled)
            y_pred = tf.sigmoid(tmp)
            # split up batches into a third dimension and stack them in third dimension
            preds = tf.stack(tf.split(y_pred, samples, axis=0), axis=0)
            return tf.math.reduce_mean(preds, axis=0), tf.math.reduce_std(preds, axis=0)

        for f in tqdm.tqdm(iter_.take(batches)):
            inputs_b = f[:-2]
            truth_b = f[-2]
            weights_b = f[-1]
            # Calculate epistemic uncertainty for batch by iterating over a certain number of times,
            # getting y_preds. You can then calculate the mean and sigma of the predictions,
            # and use this to gather uncertainty: (see http://krasserm.github.io/2019/03/14/bayesian-neural-networks/)
            # inputs, truth, sample_weight
            preds_mean_b, preds_std_b = predict_step(inputs_b)

            preds_mean.append(preds_mean_b)
            preds_std.append(preds_std_b)
            truth.append(truth_b)
            sample_weight.append(weights_b)

        # concat all results
        preds_mean = tf.concat(preds_mean, axis=0)
        preds_std = tf.concat(preds_std, axis=0)

        truth = tf.concat(truth, axis=0)
        sample_weight = tf.concat(sample_weight, axis=0)

        # trim off extra from last batch
        truth = truth[:num_samples, :].numpy()
        preds_mean = preds_mean[:num_samples, :].numpy()
        preds_std = preds_std[:num_samples, :].numpy()
        sample_weight = sample_weight[:num_samples, :].numpy()

        # reset truth back to 0 to compute metrics
        # sample weights will rule these out anyways when computing metrics
        truth_reset = np.copy(truth)
        truth_reset[truth_reset < Label.UNBOUND.value] = 0

        # do not continue to calculate metrics. Just return predictions and true values
        if (not calculate_metrics):
            return {
                'preds_mean': preds_mean,
                'preds_std': preds_std,
                'truth': truth,
                'weights': sample_weight,
                'assay_dict': None,
                'auROC': None,
                'auPRC': None
            }

        assert(preds_mean.shape == sample_weight.shape)

        try:

            # try/accept for cases with only one class (throws ValueError)
            assay_dict = get_performance(self.assaymap, preds_mean, truth_reset, sample_weight, self.predict_assays)

            # calculate averages
            auROC = np.nanmean(list(map(lambda x: x['AUC'],assay_dict.values())))
            auPRC = np.nanmean(list(map(lambda x: x['auPRC'],assay_dict.values())))
            avgGINI = np.nanmean(list(map(lambda x: x['GINI'],assay_dict.values())))

            tf.compat.v1.logging.info("macro auROC:     " + str(auROC))
            tf.compat.v1.logging.info("auPRC:     " + str(auPRC))
            tf.compat.v1.logging.info("GINI:     " + str(avgGINI))
        except ValueError as v:
            auROC = None
            auPRC = None
            tf.compat.v1.logging.info("Failed to calculate metrics")

        return {
            'preds_mean': preds_mean,
            'preds_std': preds_std,
            'truth': truth,
            'weights': sample_weight,
            'assay_dict': assay_dict,
            'auROC': auROC,
            'auPRC': auPRC
        }

    def score_whole_genome(self, similarity_peak_files,
                       file_prefix,
                       chrs=None,
                       all_data = None):
        """
        Runs a whole genome scan for all available genomic regions in the dataset (about 3.2Million regions)
        Takes about 1 hour.

        Args:
            :param similarity_peak_files: list of similarity_peak_files corresponding to similarity_assays
            :param file_prefix: path to save compressed numpy file to. Adds '.npz' extension.
            :param chroms: list of chromosome names to score. If none, scores all chromosomes.
            :param all_data: for testing. If none, generates a concatenated matrix of all data when called.

        """

        # get peak_vector, which is a vector matching train set. Some peaks will not overlap train set,
        # and their indices are stored in missing_idx for future use
        peak_vectors = [bedFile2Vector(f, self.regionsFile)[0] for f in similarity_peak_files]
        peak_matrix = np.vstack(peak_vectors)
        del peak_vectors

        liRegions = list(enumerate(load_bed_regions(self.regionsFile)))

        # filter liRegions by chrs
        if chrs is not None:
            liRegions = [i for i in liRegions if i[1].chrom in chrs]

        # get indices to score
        idx = np.array([i[0] for i in liRegions])
        liRegions = [i[1] for i in liRegions]

        print("scoring %i regions" % idx.shape[0])

        if all_data is None:
            all_data = concatenate_all_data(self.data, self.regionsFile)

        # tuple of means and stds
        predictions = self.eval_vector(all_data, peak_matrix, idx)
        print("finished predictions...", predictions[0].shape)

        # zip together means and stdevs for each position in idx

        # return matrix of region, TF information
        npRegions = np.array(list(map(lambda x: np.array([x.chrom, x.start, x.end]),liRegions)))
        # TODO turn into right types (all strings right now)
        # predictions[0] is means of size n regions by # ChIP-seq peaks predicted
        means = np.concatenate([npRegions, predictions[0]], axis=1)
        stds = np.concatenate([npRegions, predictions[1]], axis=1)

        # can load back in using:
        # > loaded = np.load('file_prefix.npz')
        # > loaded['means'], loaded['stds']
        # TODO: save the right types!  (currently all strings!)
        np.savez_compressed(file_prefix, means = means, stds=stds,
                            names=np.array(['chr','start','end'] + list(self.assaymap)[1:]))

        print("columns for matrices are chr, start, end, %s" % ", ".join(list(self.assaymap)[1:]))

    def score_matrix(self, accessilibility_peak_matrix, regions_peak_file, all_data = None):
        """ Runs predictions on a matrix of accessibility peaks, where columns are samples and
        rows are regions from regions_peak_file. rows in accessilibility_peak_matrix should matching

        Args:
            :param accessilibility_peak_matrix: numpy matrix of (samples by genomic regions)
            :param regions_peak_file: narrowpeak or bed file containing regions to score. Number of regions Should
              match rows in accessilibility_peak_matrix
            :param all_data: for testing. If none, generates a concatenated matrix of all data when called.

        Returns:
            3-dimensional numpy matrix of predictions: sized (samples by regions by ChIP-seq targets)
        """
        print("in function")
        s = time.time_ns()
        if all_data is None:
            all_data = concatenate_all_data(self.data, self.regionsFile)
        e = time.time_ns()
        print('concat: %i' % (e-s))

        s = time.time_ns()
        regions_bed = bed2Pyranges(regions_peak_file)
        all_data_regions = bed2Pyranges(self.regionsFile)
        e = time.time_ns()
        print('bed2pyranges: %i' % (e - s))

        joined = regions_bed.join(all_data_regions, how='left',suffix='_alldata').df

        # select regions with data to score
        idx = joined['idx_alldata']

        results_tmp = []

        s = time.time_ns()
        # TODO 9/10/2020: should do something more efficiently than a for loop
        # Wrap this for loop in a function
        for sample_i in range(accessilibility_peak_matrix.shape[0]):
            # tuple of means and stds
            peaks_i = np.zeros((len(all_data_regions)))
            peaks_i[idx] = accessilibility_peak_matrix[sample_i, joined['idx']]

            # means, _ = self.eval_vector(all_data, peaks_i, idx)

            # group means by joined['idx']
            results_tmp.append(self.eval_vector_2(all_data, peaks_i, idx))
        
        results = results_tmp
        
        e = time.time_ns()
        print("total eval_vector time: %i" % (e - s))

        # stack all samples along 0th axis
        s = time.time_ns()
        tmp = np.stack(results)

        # get the index break for each region_bed region
        reduce_indices = joined.drop_duplicates('idx',keep='first').index.values

        # get the number of times there was a scored region for each region_bed region
        # used to calculate reduced means
        indices_counts = joined['idx'].value_counts(sort=False).values[:,None]

        # reduce means on middle axis
        final = np.add.reduceat(tmp, reduce_indices, axis = 1)/indices_counts

        # TODO 9/10/2020: code is currently scoring missing values and setting to nan.
        # You could be more efficient and remove these so you are not taking
        # the time to score garbage data.

        # fill missing indices with nans
        missing_indices = joined[joined['idx_alldata']==-1]['idx'].values
        final[:,missing_indices, :] = np.NAN
        e = time.time_ns()
        print('the rest: %i' % (e - s))

        return final


    def score_peak_file(self, similarity_peak_files, regions_peak_file, all_data = None):
        """ Runs predictions on a set of peaks defined in a bed or narrowPeak file.

        Args:
            :param similarity_peak_files: narrowpeak or bed files containing chromatin accessibility to score
            :param regions_peak_file: narrowpeak or bed file containing regions to score.
            :param all_data: for testing. If none, generates a concatenated matrix of all data when called.

        Returns:
            pandas dataframe of genomic regions and predictions
        """


        # get peak_vector, which is a vector matching train set. Some peaks will not overlap train set,
        # and their indices are stored in missing_idx for future use
        peak_vectors = [bedFile2Vector(f, self.regionsFile)[0] for f in similarity_peak_files]
        peak_matrix = np.vstack(peak_vectors)
        del peak_vectors

        peak_vector_regions, all_peaks_regions = bedFile2Vector(regions_peak_file, self.regionsFile)

        print("finished loading peak file")

        # only select peaks to score
        idx = np.where(peak_vector_regions == True)[0]

        print("scoring %i regions" % idx.shape[0])

        if len(idx) == 0:
            raise ValueError("No positive peaks found in %s" % regions_peak_file)

        if all_data is None:
            all_data = concatenate_all_data(self.data, self.regionsFile)

        # tuple of means and stds
        means, stds = self.eval_vector(all_data, peak_matrix, idx)
        print("finished predictions...", means.shape)

        assert type(means) == type(stds), "Means and STDs variables not of the same type"
        if not isinstance(means, np.ndarray):
            means = means.numpy()
            stds = stds.numpy()

        means_df =  pd.DataFrame(data=means, columns=list(self.assaymap)[1:])
        std_cols = list(map(lambda x: x + "_stds",list(self.assaymap)[1:]))
        stds_df =  pd.DataFrame(data=stds, columns=std_cols)

        # read in regions file and filter by indices that were scored
        p = pd.read_csv(self.regionsFile, sep='\t',header=None)[[0,1,2]]
        p['idx']=p.index # keep original bed region ordering using idx column
        p.columns = ['Chromosome', 'Start','End','idx']
        prediction_positions = p[p['idx'].isin(idx)] # select regions that were scored
        # reset index to match predictions shape
        prediction_positions = prediction_positions.reset_index()
        prediction_positions['idx'] = prediction_positions.index
        prediction_positions = pd.concat([prediction_positions,means_df,stds_df],axis=1)
        prediction_positions_pr = pr.PyRanges(prediction_positions).sort()

        original_file = bed2Pyranges(regions_peak_file)
        # left join
        joined = original_file.join(prediction_positions_pr, how='left')
        # turn into dataframe
        joined_df = joined.df
        joined_df = joined_df.drop(['index','Start_b','End_b','idx_b'],axis=1)
        # set missing values to 0
        num = joined_df._get_numeric_data()
        num[num < 0] = 0
        mean_results = joined_df.groupby('idx').mean()

        tmp = original_file.df
        tmp = tmp.set_index(tmp['idx']) # correctly set index to original ordering

        assert np.all(tmp.sort_index()['Start'].values == mean_results['Start'].values), "Error: genomic position start sites in mean results and original bed file do not match!"
        assert np.all(tmp.sort_index()['End'].values == mean_results['End'].values), "Error: genomic position end sites in mean results and original bed file do not match!"

        return pd.concat([tmp[['Chromosome']],mean_results], axis=1)

class VLP(VariationalPeakModel):
    def __init__(self,
             *args,
             **kwargs):
        """ Creates a new model with 4 layers with 100 unites each.
            To resume model training on an old model, call:

            .. code-block:: python

                model = VLP(checkpoint=path_to_saved_model)
        """
        self.activation = tf.tanh
        self.layers = 2

        if "checkpoint" in kwargs.keys():
            fileObject = open(kwargs["checkpoint"] + "/model_params.pickle" ,'rb')
            metadata = pickle.load(fileObject)
            fileObject.close()
            # remove checkpoint from kwargs
            VariationalPeakModel.__init__(self, **metadata, **kwargs)
            file = h5py.File(os.path.join(kwargs["checkpoint"], "weights.h5"), 'r')

            # load model weights back in
            weights = []
            for i in range(len(file.keys())):
                weights.append(file['weight' + str(i)][:])
            self.model.set_weights(weights)
            file.close()

        else:
            VariationalPeakModel.__init__(self, *args, **kwargs)

    def create_model(self, **kwargs):
        """ Creates an Epitome model.
        """
        cell_inputs = [tf.keras.layers.Input(shape=(self.num_inputs[i],))
                       for i in range(len(self.num_inputs))]

        # make a channel for each cell type
        cell_channels = []

        # TODO resize by max iterations. 5000 is an estimate for data size
        kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /
                            tf.cast(self.batch_size * 5000, dtype=tf.float32))
        for i in range(len(self.num_inputs)):
            # make input layer for cell
            last = cell_inputs[i]
            for j in range(self.layers):
                num_units = int(self.num_inputs[i]/(2 * (j+1)))
                d = tfp.layers.DenseFlipout(num_units,
                                                kernel_divergence_fn=kl_divergence_function,
                                                activation = self.activation)(last)
                last = d
            cell_channels.append(last)


        # merge together all cell channels
        if (len(cell_channels) > 1):
            last = tf.keras.layers.concatenate(cell_channels)
        else:
            last = cell_channels[0]

        outputs = tfp.layers.DenseFlipout(self.num_outputs,
                                        kernel_divergence_fn=kl_divergence_function,
                                        activity_regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2),
                                        name="output_layer")(last)

        model = tf.keras.models.Model(inputs=cell_inputs, outputs=outputs)
        return model


if __name__ == '__main__':
    x = np.random.rand(25, 30_000)
    # vlps = [VLP.remote(['CEBPB'], test_celltypes=['K562']) for _ in range(4)]
    # VLPActor = VLP.remote(['CEBPB'], test_celltypes=['K562'])
    
    # result = VLPActor.score_matrix.remote(x, 'data/test_regions.bed', all_data = None)
    # ray.get(results)
    
    # results = ray.get([v.score_matrix.remote(x, 'data/test_regions.bed', all_data = None) for v in vlps])

    model = VLP(['CEBPB'], test_celltypes=['K562'])
    # score_matrix_2(model, x, 'data/test_regions.bed', all_data = None)

    x = model.score_matrix(x, 'data/test_regions.bed', all_data = None)

    print(x)