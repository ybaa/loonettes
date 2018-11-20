from src.constants import CIFAR_10_PATH
import pickle


class CifarLoader:

    def load_data(self):
        dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3',
                'data_batch_4', 'data_batch_5', 'test_batch']

        all_data = [0, 1, 2, 3, 4, 5, 6]

        for i, direc in zip(all_data, dirs):
            all_data[i] = self.unpickle(CIFAR_10_PATH + direc)

        # batch[n], test_batch, batch_meta
        return [all_data[1], all_data[2], all_data[3], all_data[4], all_data[5]], all_data[6], all_data[0]

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            cifar_dict = pickle.load(fo, encoding='bytes')
        return cifar_dict