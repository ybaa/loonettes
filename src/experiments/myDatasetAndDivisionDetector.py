from src.models.cnnMyDataset import CNNMyDataset
from src.models.divisionDetector import DivisionDetector


class MyDatasetAndDivisionDetectorManager:

    def __init__(self):
        pass

    # works both for 4 and 3 channels
    def test(self):
        correct_answers = 0     # all objects recognized
        correct_min_one_object_recognized = 0 # minimum one object found
        tested_images = 1
        my_cnn = CNNMyDataset()
        my_dataset_helper = my_cnn.load_and_prepare_set(reshape_test_images=False, for_classification=False)
        division_detector = DivisionDetector()

        for test_image, test_labels in zip(my_dataset_helper.test_images, my_dataset_helper.test_batches_encoded):
            division_detector.divided_image = []
            division_detector.divided_image.append(test_image)
            division_detector.divide_recursively(2, test_image)
            single_img_divided_rescaled = my_dataset_helper.resize_images(division_detector.divided_image)

            single_img_predictions = []
            for part_of_image in single_img_divided_rescaled:
                part_img_pred_val = my_cnn.predict_single_image(part_of_image)

                if part_img_pred_val is not None:
                    single_img_predictions.append(part_img_pred_val[0])

            recognized_objects_on_single_photo = 0
            for test_label in test_labels:
                if test_label in single_img_predictions:
                    recognized_objects_on_single_photo += 1

            if recognized_objects_on_single_photo == len(test_labels):
                correct_answers += 1

            if recognized_objects_on_single_photo > 0:
                correct_min_one_object_recognized += 1

            print('current acc for all objects: ' + str(correct_answers / tested_images) + ' on step ' + str(tested_images))
            print('current acc for min 1 object: ' + str(correct_min_one_object_recognized / tested_images) + ' on step ' + str(tested_images))
            tested_images += 1
            print('----------------------------------------')

        print('final acc for all objects: ' + str(correct_answers / len(my_dataset_helper.test_batches_encoded)))
        print('final acc for min 1 object: ' + str(correct_min_one_object_recognized / len(my_dataset_helper.test_batches_encoded)))
