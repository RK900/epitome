"""
Functions for data generators.
"""

import numpy as np
import tensorflow as tf
from .constants import *
from .functions import *
import epitome.iio as iio
import glob

######################### Original Data Generator: Only peak based #####################

def gen_from_peaks_to_tf_records(data, 
                 label_cell_types,  # used for labels. Should be all for train/eval and subset for test
                 eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 matrix,
                 assaymap,
                 cellmap,
                 radii,
                 **kwargs):

    """
    Takes Deepsea data and calculates distance metrics from cell types whose locations
    are specified by label_cell_indices, and the other cell types in the set. Label space is only one cell type.
     TODO AM 3/7/2019
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n by 919 labels.
    :param label_cell_types: list of cell types to be rotated through and used as labels (subset of eval_cell_types)
    :param eval_cell_types: list of cell types to be used in evaluation (includes label_cell_types)
    :param matrix: matrix of celltype, assay positions
    :param assaymap: map of column assay positions in matrix
    :param cellmap: map of row cell type positions in matrix
    :param radii: radii to compute dnase distances from
    :param kwargs: kargs

    :returns: generator of data with three elements:
        1. record features
        2. record labels for a given cell type
        3. 0/1 mask of labels that have validation data. For example, if this record is for celltype A549,
        and A549 does not have data for ATF3, there will be a 0 in the position corresponding to the label space.
    """

    # Running in TRAIN, VALID, or TEST?    
    mode = kwargs.get("mode")
    # specifies the indices to generate records.
    # can be used for debug purposes, or to evaluate
    # only specific regions in a vector
    # TODO AM 4/17/2019: move this to an actual parameter
    indices = kwargs.get("indices")
    
    if (not isinstance(indices, np.ndarray) and not isinstance(indices, list)):
        indices = range(0, data["y"].shape[-1]) # if not defined, set to all points
    
    if (not isinstance(mode, Dataset)):
        raise ValueError("mode is not a Dataset enum")
        
    if (mode == Dataset.RUNTIME):
        label_cell_types = ["PLACEHOLDER_CELL"]
        dnase_vector = kwargs.get("dnase_vector")
        random_cell = list(cellmap)[0] # placeholder to get label vector length
        
    print("using %s as labels for mode %s" % (label_cell_types, mode))
    
    # string of radii for meta data labeling
    radii_str = list(map(lambda x: "DNASE_RADII_%i" % x, radii))
        
    if (mode == Dataset.TEST or mode == Dataset.RUNTIME):
        # Drop cell types with the least information (TODO AM 4/1/2019 this could do something smarter)
        
        # make dictionary of eval_cell_type: assay count and sort in decreasing order
        tmp = matrix.copy()
        tmp[tmp>= 0] = 1
        tmp[tmp== -1] = 0
        sums = np.sum(tmp, axis = 1)
        cell_assay_counts = zip(list(cellmap), sums)
        cell_assay_counts = sorted(cell_assay_counts, key = lambda x: x[1])
        # filter by eval_cell_types
        cell_assay_counts = list(filter(lambda x: x[0] in eval_cell_types, cell_assay_counts))
        
        # remove cell types with smallest number of factors
        eval_cell_types = eval_cell_types.copy()
        [eval_cell_types.remove(i[0]) for i in cell_assay_counts[0:len(label_cell_types)]]
        del tmp
        del cell_assay_counts
        
    def g():
                
        for i in indices: # for all records specified
            
            for (cell) in label_cell_types: # for all cell types to be used in labels
                dnases = [] 
                
                # cells to be featurized
                feature_cells = eval_cell_types.copy()
                
                # try to remove cell if it is in the possible list of feature cell types
                try:
                    feature_cells.remove(cell) 
                except ValueError:
                    pass  # do nothing!
                
                # features from all remaining cells not in label set
                feature_cell_indices_list = list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c), 
                                                     feature_cells))
                feature_cell_indices = np.array(feature_cell_indices_list).flatten()
                
                # labels for this cell
                if (mode != Dataset.RUNTIME):
                    label_cell_indices = get_y_indices_for_cell(matrix, cellmap, cell)
                    label_cell_indices_no_dnase = np.delete(label_cell_indices, [0])

                    # Copy assay_index_no_dnase and turn into mask of 0/1 for whether data for this cell type for
                    # a given label is available.
                    assay_mask = np.copy(label_cell_indices_no_dnase)
                    assay_mask[assay_mask == -1] = 0
                    assay_mask[assay_mask > 0] = 1
                    
                else:
                    label_count = len(get_y_indices_for_cell(matrix, cellmap, random_cell))-1
                    
                    # Mask and labels are all 0's because labels are missing during runtime
                    garbage_labels = assay_mask = np.zeros(label_count)

                # get dnase indices for cell types that are going to be features
                dnase_indices = [x[0] for x in feature_cell_indices_list]
                            
                for radius in radii:
                    min_radius = max(0, i - radius)
                    max_radius = i+radius+1
                    
                    # use DNase vector, if it is provided
                    if (mode == Dataset.RUNTIME):

                        # within the radius, fraction of places where they are both 1
                        dnase_double_positive = np.average(data["y"][dnase_indices,min_radius:max_radius]*
                                                 dnase_vector[min_radius:max_radius], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data["y"][dnase_indices,min_radius:max_radius]==
                                                 dnase_vector[min_radius:max_radius], axis=1)

                    else:
                        # within the radius, fraction of places where they are both 1
                        # label_cell_index[0] == DNase location for specific cell type
                        dnase_double_positive = np.average(data["y"][dnase_indices,min_radius:max_radius]*
                                                 data["y"][label_cell_indices[0],min_radius:max_radius], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data["y"][dnase_indices,min_radius:max_radius]==
                                                 data["y"][label_cell_indices[0],min_radius:max_radius], axis=1)
                        
                        
                    dnases.extend(dnase_double_positive)
                    dnases.extend(dnase_agreement)
                    
                    

                # Handle missing values 
                # Set missing values to 0. Should be handled later.
                features = data["y"][feature_cell_indices,i]
                

                # one hot encoding (ish). First row is 1/0 for known/unknown. second row is value.
                binding_features_n  = len(features)
                
                feature_n = binding_features_n + len(dnases)
                
                # two row matrix where first row is feature mask and second row is features
                x_data = np.concatenate([features, dnases])
                x_mask = np.ones([feature_n])
                
                x_mask[np.where(feature_cell_indices == -1)[0]] = 0 # assign UNKs to missing features

                # There can be NaNs in the DNases for edge cases (when the radii extends past the end of the chr).
                # Mask these values in the first row of tmp
                x_mask[np.where(np.isnan(x_data))[0]] = 0 # assign UNKs to missing DNase values
                x_data[np.where(np.isnan(x_data))[0]] = 0 # set NaNs to 0
                
                assert(x_mask.shape == x_data.shape)
                
                feature = {}
                # TODO maybe nest these for readability
                feature['x/data'] = iio.make_float_feature(x_data)
                feature['x/shape'] = iio.make_int64_feature(x_data.shape)
                feature['x/mask'] =iio.make_int64_feature(x_mask.astype(np.bool))
                
                feature['eval_cell_types'] = iio.make_bytes_feature(map(lambda t: bytes(t, 'utf-8'), eval_cell_types))
                feature['label_cell_types'] = iio.make_bytes_feature(map(lambda t: bytes(t, 'utf-8'), label_cell_types))
                
                feature["y/labels"] = iio.make_bytes_feature(map(lambda t: bytes(t, 'utf-8'), list(assaymap)[1:])) # label assays, drop DNase
                feature["y/celltype"] = iio.make_bytes_feature([bytes(cell, 'utf-8')]) # label cell
                feature["y/mask"] = iio.make_int64_feature(assay_mask) # mask for labels
                
                # Save assay and cell type ordering
                # cell types for assays ["cell1", "cell1", cell2", ...]
                x_celltype_for_assay = np.repeat(list(feature_cells),len(list(assaymap)))
                # cell types for chromatin similarity ["cell1", "cell2", cell1", ...]
                x_celltype_for_similarity = np.tile(feature_cells, len(radii * 2))
                # assays for features ["DNase", "ATF3", ... , "DNase", "ATF3", ..]
                x_label_for_assay = np.repeat(list(feature_cells),len(list(assaymap)))
                # radius for chromatin similariry [1, 3, 10, 30, ... ]
                x_label_for_similarity = np.repeat(radii_str, len(feature_cells * 2))
                
                x_labels_celltype = np.concatenate([x_celltype_for_assay, x_celltype_for_similarity])
                x_labels_label = np.concatenate([x_label_for_assay, x_label_for_similarity ])
                                                       
                assert(len(x_celltype_for_assay) == binding_features_n)
                assert(len(x_celltype_for_similarity) == len(dnases))
                
                assert(len(x_labels_label) == len(x_labels_celltype))             
                assert(len(x_labels_celltype) == x_data.shape[0])                                              
                     
                feature["x/celltypes"] = iio.make_bytes_feature(map(lambda t: bytes(t, 'utf-8'), x_labels_celltype))
                feature["x/labels"] = iio.make_bytes_feature(map(lambda t: bytes(t, 'utf-8'), x_labels_label))
                    
                if (mode != Dataset.RUNTIME):
                    # yield features, labels, label mask
                    labels = data["y"][label_cell_indices_no_dnase,i]
                    
                    # The features going into the example.
                    feature['y/data'] = iio.make_int64_feature(labels)
                    feature['y/shape'] = iio.make_int64_feature(labels.shape)

                    feature['has_real_labels'] = iio.make_int64_feature([True])

                else:
                    # The features going into the example.
                    feature['y/data'] = iio.make_int64_feature(garbage_labels)
                    feature['y/shape'] = iio.make_int64_feature(garbage_labels.shape)

                    feature['real_labels'] = iio.make_int64_feature([False])
                    
                yield iio.make_example(feature)

    return g

############################################################################################
######################## Functions for running data generators #############################
############################################################################################
def _parse_function(example_proto, example):
    """ 
    A helper function for extracting data from an example_proto
    
    :param single instance of tensorflow.core.example.example_pb2.Example
    :return (data, labels and label mask) for example
    """
    x_shape = example.features.feature['x/shape'].int64_list.value[0]
    y_shape = example.features.feature['y/shape'].int64_list.value[0]
    eval_cell_count = len(example.features.feature['eval_cell_types'].bytes_list.value)
    label_cell_count = len(example.features.feature['label_cell_types'].bytes_list.value)
    features = {
        'x/data': tf.FixedLenFeature((x_shape,), tf.float32),
        'x/shape': tf.FixedLenFeature((1,), tf.int64),
        'x/mask': tf.FixedLenFeature((x_shape,), tf.int64),
        'eval_cell_types': tf.FixedLenFeature((eval_cell_count,), tf.string),
        'label_cell_types': tf.FixedLenFeature((label_cell_count,), tf.string),
        'y/labels': tf.FixedLenFeature((y_shape,), tf.string),
        'y/celltype': tf.FixedLenFeature((1,), tf.string),
        'y/mask': tf.FixedLenFeature((y_shape,), tf.int64),
        'x/celltypes': tf.FixedLenFeature((x_shape,), tf.string),
        'x/labels': tf.FixedLenFeature((x_shape,), tf.string),
        'y/data': tf.FixedLenFeature((y_shape,), tf.int64),
        'y/shape': tf.FixedLenFeature((1,), tf.int64),
        'has_real_labels': tf.FixedLenFeature((1,), tf.int64)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    
    feature_tensor = tf.concat([tf.cast(parsed_features["x/mask"], tf.float32), parsed_features["x/data"]], 0)
    feature_tensor = tf.reshape(feature_tensor, (2, x_shape))
    
    return (feature_tensor, tf.cast( parsed_features["y/data"], tf.float32), 
            tf.cast(parsed_features["y/mask"], tf.float32))



def tf_records_to_one_shot_iterator(path, dataset, batch_size, shuffle_size, prefetch_size):
    """
    Generates a one shot iterator from a list of filenames pointing to tf records.
    
    :param g: data generator
    :param batch_size: number of elements in generator to combine into a single batch
    :param shuffle_size: number of elements from the  generator fromw which the new dataset will shuffle
    :param prefetch_size: maximum number of elements that will be buffered  when prefetching
    :param radii: where to calculate DNase similarity to.
    
    :returns: tuple of (label shape, one shot iterator)
    """
    
    files = glob.glob(os.path.join(path, "*"))
    
    if dataset == Dataset.TRAIN:
        file_prefix = "train.tfrecord"
    elif dataset == Dataset.VALID:
        file_prefix = "valid.tfrecord"
    elif dataset == Dataset.TEST:
        file_prefix = "test.tfrecord"
        
    else:
        raise Exception("invalid dataset specified for generating iterator: %s" % dataset)
        
    filtered_files = list(filter(lambda x: file_prefix in x, files))

    # get 1 example to get sample shape information
    compression_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    t = iio.read_tfrecord(filtered_files[0], proto=None, options=compression_options)
    example = next(t)
    
    try: 
        dataset = tf.data.TFRecordDataset(filenames=filtered_files,
                                  compression_type='GZIP')

    except NameError as e:
        print("Error: no data, %s" % e)

    dataset = dataset.map(lambda x: _parse_function(x, example))
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_size)

    try: 
        return example.features.feature["y/shape"].int64_list.value[0], dataset.make_one_shot_iterator()
    except NameError as e:
        return None, dataset.make_one_shot_iterator()

    
def _parse_function_from_generator(example):
    """ 
    A helper function for extracting data from an example_proto
    
    :param single instance of tensorflow.core.example.example_pb2.Example
    :return (data, labels and label mask) for example
    """
    x_shape = example.features.feature['x/shape'].int64_list.value[0]

    # TODO this is SOOOO inefficient but I am at a loss here for what to do.
    x_mask = np.array(example.features.feature["x/mask"].int64_list.value)
    x_data = np.array(example.features.feature["x/data"].float_list.value)
    y_data = np.array(example.features.feature["y/data"].int64_list.value)
    y_mask = np.array(example.features.feature["y/mask"].int64_list.value)
    
    return (np.vstack([x_mask, x_data]), y_data, y_mask)
    
def generator_to_one_shot_iterator(g, batch_size, shuffle_size, prefetch_size):
    """
    Generates a one shot iterator from a data generator.
    
    :param g: data generator
    :param batch_size: number of elements in generator to combine into a single batch
    :param shuffle_size: number of elements from the  generator fromw which the new dataset will shuffle
    :param prefetch_size: maximum number of elements that will be buffered  when prefetching
    :param radii: where to calculate DNase similarity to.
    
    :returns: tuple of (label shape, one shot iterator)
    """
    
    for example in g():
        break
        
    x_shape = example.features.feature['x/shape'].int64_list.value[0]
    y_shape = example.features.feature['y/shape'].int64_list.value[0]

    def g_mapped():
        for x in g():
            yield _parse_function_from_generator(x)
    
    for values in g_mapped():
        break
        
    # only set output_shapes if g() has data
    dataset = tf.data.TFRecordDataset.from_generator( # generator of tensorflow.core.example.example_pb2.Example
        g_mapped,
        output_types=(tf.float32,)*3, # 3 = features, labels, and missing indices
        output_shapes=((2, x_shape), (y_shape,), (y_shape,),)
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_size)
    
    try: 
        example
        return example.features.feature["y/shape"].int64_list.value[0], dataset.make_one_shot_iterator()
    except NameError as e:
        return None, dataset.make_one_shot_iterator()