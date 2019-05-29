from src.models.cnnMyDataset import CNNMyDataset
from src.models.divisionDetector import DivisionDetector
from src.models.cnnMyDatasetMultiLabel import CNNMyDatasetMultiLabel
import tensorflow as tf
import time
import csv
from src.constants import CURR_CHANNELS


class MyDatasetAndDivisionDetectorManager:

    def __init__(self):
        pass

    # works both for 4 and 3 channels (set in constants.py)
    def test_detection(self,save_csv=False):
        correct_answers = 0                     # all objects recognized
        correct_min_one_object_recognized = 0   # minimum one object found
        tested_images = 1
        my_cnn = CNNMyDataset()
        my_dataset_helper = my_cnn.load_and_prepare_set(reshape_test_images=False, for_classification=False)
        division_detector = DivisionDetector()

        data_to_csv = []

        with tf.Session() as sess:
            my_cnn.restore_model(sess, '../models/myConvo/' + str(CURR_CHANNELS) + 'ch/model.ckpt')
            for test_image, test_labels in zip(my_dataset_helper.test_images, my_dataset_helper.test_batches_encoded):
                division_detector.divided_image = []
                division_detector.divided_image.append(test_image)
                division_detector.divide_recursively(2, test_image)
                single_img_divided_rescaled = my_dataset_helper.resize_images(division_detector.divided_image)

                single_img_predictions = []

                start = time.time()

                for part_of_image in single_img_divided_rescaled:
                    part_img_pred_val, prob_val = my_cnn.predict_single_image(part_of_image, sess)

                    if part_img_pred_val is not None:
                        single_img_predictions.append(part_img_pred_val[0])

                end = time.time()

                if len(single_img_predictions) > 0:

                    recognized_objects_on_single_photo = 0
                    for test_label in test_labels:
                        if test_label in single_img_predictions:
                            recognized_objects_on_single_photo += 1

                    if recognized_objects_on_single_photo == len(test_labels):
                        correct_answers += 1

                    if recognized_objects_on_single_photo > 0:
                        correct_min_one_object_recognized += 1

                    data_to_csv.append([tested_images,
                                        correct_answers / tested_images,
                                        correct_min_one_object_recognized / tested_images,
                                        end - start])

                    print('current acc for all objects: ' + str(correct_answers / tested_images) + ' on step ' + str(tested_images))
                    print('current acc for min 1 object: ' + str(correct_min_one_object_recognized / tested_images) + ' on step ' + str(tested_images))
                    print('time of prediciton: ' + str(end-start))
                    tested_images += 1
                    print('----------------------------------------')

        print('final acc for all objects: ' + str(correct_answers / len(my_dataset_helper.test_batches_encoded)))
        print('final acc for min 1 object: ' + str(correct_min_one_object_recognized / len(my_dataset_helper.test_batches_encoded)))

        if save_csv:
            with open('../reports/detection_eval_' + str(CURR_CHANNELS) + 'ch2.csv', mode='w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in data_to_csv:
                    writer.writerow(row)


    def test_detection_for_mlc(self, save_csv=False):
        correct_answers = 0  # all objects recognized
        correct_min_one_object_recognized = 0  # minimum one object found
        tested_images = 1
        my_cnn = CNNMyDatasetMultiLabel()
        my_dataset_helper = my_cnn.load_and_prepare_set(reshape_test_images=False, for_classification=False)
        division_detector = DivisionDetector()

        data_to_csv = []

        with tf.Session() as sess:
            for test_image, test_labels in zip(my_dataset_helper.test_images, my_dataset_helper.test_labels):
                division_detector.divided_image = []
                division_detector.divided_image.append(test_image)
                division_detector.divide_recursively(2, test_image)
                single_img_divided_rescaled = my_dataset_helper.resize_images(division_detector.divided_image)

                single_img_predictions = []
                start = time.time()

                for part_of_image in single_img_divided_rescaled:
                    part_img_pred_val, prob_val = my_cnn.predict_single_image(part_of_image, sess)

                    if part_img_pred_val is not None:
                        single_img_predictions.append(part_img_pred_val)

                end = time.time()

                indices = [i for i, x in enumerate(test_labels) if x == '1']

                recognized_objects_on_single_photo = 0
                for test_label in indices:
                    if test_label in single_img_predictions:
                        recognized_objects_on_single_photo += 1

                if recognized_objects_on_single_photo == len(indices):
                    correct_answers += 1

                if recognized_objects_on_single_photo > 0:
                    correct_min_one_object_recognized += 1

                data_to_csv.append([tested_images,
                                    correct_answers / tested_images,
                                    correct_min_one_object_recognized / tested_images,
                                    end - start])

                print('current acc for all objects: ' + str(correct_answers / tested_images) + ' on step ' + str(tested_images))
                print('current acc for min 1 object: ' + str(correct_min_one_object_recognized / tested_images) + ' on step ' + str(tested_images))
                print('time of prediciton: ' + str(end - start))
                tested_images += 1
                print('----------------------------------------')

                if tested_images % 10 == 0:
                    if save_csv:
                        with open('../reports/detection_eval_' + str(CURR_CHANNELS) + 'ch_MLC.csv',
                                  mode='w') as csv_file:
                            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            for row in data_to_csv:
                                writer.writerow(row)

        print('final acc for all objects: ' + str(correct_answers / len(my_dataset_helper.test_batches_encoded)))
        print('final acc for min 1 object: ' + str(correct_min_one_object_recognized / len(my_dataset_helper.test_batches_encoded)))

        if save_csv:
            with open('../reports/detection_eval_' + str(CURR_CHANNELS) + 'ch_MLC.csv', mode='w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in data_to_csv:
                    writer.writerow(row)
