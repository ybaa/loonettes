from src.experiments.myDatasetAndDivisionDetector import MyDatasetAndDivisionDetectorManager
from src.models.cnnMyDataset import CNNMyDataset
from src.models.cnnMyDatasetMultiLabel import CNNMyDatasetMultiLabel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # turn off warnings



# # 1 CNN
# # detection test
# exp1 = MyDatasetAndDivisionDetectorManager()
# exp1.test_detection()
#
# # learning sess
# cnn = CNNMyDataset()
# cnn.run_learning_session(save=True, restore=False)
#
#
# # MLC
# # learning sess
#
# classes = ['backpack', 'bike', 'book', 'chair', 'coach', 'cup', 'phone', 'skateboard']
# for cls in classes:
#     mlc = CNNMyDatasetMultiLabel(cls)
#     mlc.run_learning_session(save=True)


# detectiont est
exp2 = MyDatasetAndDivisionDetectorManager()
exp2.test_detection_for_mlc()
