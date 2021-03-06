from epitome.test import EpitomeTestCase
from epitome.test import *
from epitome.functions import *
from epitome.dataset import EpitomeDataset, REQUIRED_KEYS, EPITOME_H5_FILE
from epitome.constants import Dataset

import os
import numpy as np
import pytest
import warnings


class DatasetTest(EpitomeTestCase):

    def __init__(self, *args, **kwargs):
        super(DatasetTest, self).__init__(*args, **kwargs)
        self.dataset = EpitomeDataset()

    def test_user_data_path(self):
        # user data path should be able to be explicitly set
        datapath = GET_DATA_PATH()
        self.assertTrue(datapath == os.environ["EPITOME_DATA_PATH"])

    def test_save(self):

        out_path = self.tmpFile()
        print(out_path)

        # generate 100 records for each chromosome
        PER_CHR_RECORDS=10
        chroms = []
        newStarts = []
        binSize = 200

        for i in range(1,22): # for all 22 chrs
            chrom = 'chr' + str(i)
            for j in range(1,PER_CHR_RECORDS+1):
                chroms.append(chrom)
                newStarts.append(j * binSize)

        regions_df = pd.DataFrame({'Chromosome': chroms, 'Start': newStarts})

        targets = ['DNase', 'CTCF', 'Rad21', 'SCM3', 'IRF3']
        cellTypes = ['K562'] * len(targets) + ['A549'] * len(targets) + ['GM12878'] * len(targets)
        targets = targets * 3

        row_df = pd.DataFrame({'cellType': cellTypes, 'target': targets})

        # generate random data of 0/1s
        np.random.seed(0)
        rand_data = np.random.randint(2, size=(len(row_df), len(regions_df))).astype('i1')

        source = 'rand_source'
        assembly = 'hg19'
        EpitomeDataset.save(out_path,
                rand_data,
                row_df,
                regions_df,
                binSize,
                assembly,
                source,
                valid_chrs = ['chr7'],
                test_chrs = ['chr8','chr9'])


        # load back in dataset and make sure it checks out
        dataset = EpitomeDataset(data_dir = out_path)

        assert dataset.assembly == assembly
        assert dataset.source == source
        assert np.all(dataset.valid_chrs == ['chr7'])
        assert np.all(dataset.test_chrs == ['chr8','chr9'])
        assert(len(dataset.targetmap)==5)
        assert(len(dataset.cellmap)==3)


    def test_gets_correct_matrix_indices(self):
        eligible_cells = ['IMR-90','H1','H9']
        eligible_targets = ['DNase','H4K8ac']

        matrix, cellmap, targetmap = EpitomeDataset.get_assays(
				targets = eligible_targets,
				cells = eligible_cells, min_cells_per_target = 3, min_targets_per_cell = 1)

        self.assertTrue(matrix[cellmap['IMR-90']][targetmap['H4K8ac']]==0) # data for first row


    def test_get_assays_single_target(self):
        TF = ['DNase', 'JUND']

        matrix, cellmap, targetmap = EpitomeDataset.get_assays(targets = TF,
                min_cells_per_target = 2,
                min_targets_per_cell = 2)

        targets = list(targetmap)
        # Make sure only JUND and DNase are in list of targets
        self.assertTrue(len(targets) == 2)

        for t in TF:
            self.assertTrue(t in targets)

    def test_get_targets_without_DNase(self):
        TF = 'JUND'

        matrix, cellmap, targetmap = EpitomeDataset.get_assays(targets = TF,
                similarity_targets = ['H3K27ac'],
                min_cells_per_target = 2,
                min_targets_per_cell = 1)

        targets = list(targetmap)
        # Make sure only JUND and is in list of targets
        self.assertTrue(len(targets) == 2)
        self.assertTrue(TF in targets)
        self.assertTrue('H3K27ac' in targets)

    def test_list_targets(self):
        targets =self.dataset.list_targets()
        self.assertTrue(len(targets) == len(self.dataset.targetmap))


    def test_get_assays_without_DNase(self):
        TF = 'JUND'

        matrix, cellmap, targetmap = EpitomeDataset.get_assays(targets = TF,
                similarity_targets = ['H3K27ac'],
                min_cells_per_target = 2,
                min_targets_per_cell = 1)

        targets = list(targetmap)
        # Make sure only JUND and is in list of assays
        self.assertTrue(len(targets) == 2)
        self.assertTrue(TF in targets)
        self.assertTrue('H3K27ac' in targets)

    def test_targets_SPI1_PAX5(self):
        # https://github.com/YosefLab/epitome/issues/22
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            matrix, cellmap, targetmap = EpitomeDataset.get_assays(targets = ['DNase','SPI1', 'PAX5'],min_cells_per_target=2, min_targets_per_cell=2)
            self.assertTrue(len(warning_list) == 1) # one for SPI1 and PAX5
            self.assertTrue(all(item.category == UserWarning for item in warning_list))


    def test_get_data(self):

        train_data = self.dataset.get_data(Dataset.TRAIN)
        self.assertTrue(train_data.shape[0] == np.where(self.dataset.matrix!= -1)[0].shape[0])
        self.assertTrue(train_data.shape[1] == 1800)

        valid_data = self.dataset.get_data(Dataset.VALID)
        self.assertTrue(valid_data.shape[1] == 100)

        test_data = self.dataset.get_data(Dataset.TEST)
        self.assertTrue(test_data.shape[1] == 200)

        all_data = self.dataset.get_data(Dataset.ALL)
        self.assertTrue(all_data.shape[1] == 2100)

        # Make sure you are getting the right data in the right order
        alldata_filtered = self.dataset.get_data(Dataset.ALL)
        dataset = h5py.File(os.path.join(self.dataset.data_dir, EPITOME_H5_FILE), 'r')

        alldata= dataset['data'][:]
        dataset.close()

        self.assertTrue(np.all(alldata[self.dataset.full_matrix[:, self.dataset.targetmap['DNase']],:] == alldata_filtered[self.dataset.matrix[:, self.dataset.targetmap['DNase']],:]))

    def test_order_by_similarity(self):
        cell = list(self.dataset.cellmap)[0]
        mode = Dataset.VALID
        sim = self.dataset.order_by_similarity(cell, mode, compare_target = 'DNase')

        self.assertTrue(len(sim) == len(self.dataset.cellmap))
        self.assertTrue(sim[0] == cell)

    def test_all_keys(self):
        data = h5py.File(os.path.join(self.dataset.data_dir, EPITOME_H5_FILE), 'r')
        keys = sorted(set(EpitomeDataset.all_keys(data)))

        # bin size should be positive
        self.assertTrue(data['columns']['binSize'][:][0] > 0)

        data.close()
        self.assertTrue(np.all([i in REQUIRED_KEYS for i in keys]))

    def test_reserve_validation_indices(self):
        # new dataset because we are modifying it
        dataset = EpitomeDataset()
        self.assertTrue(dataset.get_data(Dataset.TRAIN).shape == (746, 1800))
        self.assertTrue(dataset.get_data(Dataset.TRAIN_VALID).shape == (746,0))

        old_indices = dataset.indices[Dataset.TRAIN]
        self.assertTrue(dataset.indices[Dataset.TRAIN].shape[0] == 1800)

        dataset.set_train_validation_indices("chr1")
        self.assertTrue(dataset.get_data(Dataset.TRAIN).shape == (746, 1700))
        self.assertTrue(dataset.get_data(Dataset.TRAIN_VALID).shape == (746, 100))

        # check indices
        self.assertTrue(dataset.indices[Dataset.TRAIN].shape[0] == 1700)
        self.assertTrue(dataset.indices[Dataset.TRAIN_VALID].shape[0] == 100)
        self.assertTrue(dataset.indices[Dataset.TRAIN_VALID][0] == 0) # first chr
        self.assertTrue(dataset.indices[Dataset.TRAIN][0] == 100) # start of chr2

        joined_indices = np.concatenate([dataset.indices[Dataset.TRAIN_VALID], dataset.indices[Dataset.TRAIN]])
        joined_indices.sort()
        self.assertTrue(len(np.setdiff1d(joined_indices, old_indices)) == 0 and len(np.setdiff1d(old_indices, joined_indices)) == 0)
