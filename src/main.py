from src.experiments.myDatasetAndDivisionDetector import MyDatasetAndDivisionDetectorManager
from src.models.cnnMyDataset import CNNMyDataset
from src.models.cnnMyDatasetMultiLabel import CNNMyDatasetMultiLabel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # turn off warnings
from src.visualization.deviceManager import DeviceManager

dm = DeviceManager()
dm.run()

# # 1 CNN
# # detection test
# exp1 = MyDatasetAndDivisionDetectorManager()
# exp1.test_detection(save_csv=False)

# # learning sess
# cnn = CNNMyDataset()
# cnn.run_learning_session(save=False, restore=False, save_csv=False)
#
#
# # MLC
# # learning sess

# classes = ['backpack', 'bike', 'book', 'chair', 'coach', 'cup', 'phone', 'skateboard']
# for cls in classes:
#     mlc = CNNMyDatasetMultiLabel(cls)
#     mlc.run_learning_session(save=True, restore=False, save_csv=False)


# detectiont est
# exp2 = MyDatasetAndDivisionDetectorManager()
# exp2.test_detection_for_mlc(save_csv=True)


