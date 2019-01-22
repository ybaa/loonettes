from src.models.cnnMyDataset import CNNMyDataset
from src.models.divisionDetector import DivisionDetector


class MyDatasetAndDivisionDetectorManager:

    def __init__(self):
        pass

    def test_for_3_channels(self):
        correct_answers = 0
        my_cnn = CNNMyDataset()
        my_dataset_helper = my_cnn.load_and_prepare_set(reshape_test_images=False)
        division_detector = DivisionDetector()

        for test_image, test_label in zip(my_dataset_helper.test_images, my_dataset_helper.test_batches_encoded):
            division_detector.divided_image.append(test_image)
            division_detector.divide_recursively(2, test_image)
            single_img_divided_rescaled = my_dataset_helper.resize_images(division_detector.divided_image)

            single_img_predictions = []
            for part_of_image in single_img_divided_rescaled:
                part_img_pred_val = my_cnn.predict_single_image(part_of_image)
                single_img_predictions.append(part_img_pred_val[0])

            if test_label in single_img_predictions:
                correct_answers += 1

        print('acc: ' + str(correct_answers / len(my_dataset_helper.test_batches_encoded)))