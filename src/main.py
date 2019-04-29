from src.experiments.myDatasetAndDivisionDetector import MyDatasetAndDivisionDetectorManager
from src.models.cnnMyDatasetMultiLabel import CNNMyDatasetMultiLabel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # turn off warnings


# detection testing
# exp1 = MyDatasetAndDivisionDetectorManager()
# exp1.test_detection()

classes = ['backpack', 'bike', 'book', 'chair', 'coach', 'cup', 'phone', 'skateboard']
for cls in classes:
    mlc = CNNMyDatasetMultiLabel(cls)
    mlc.run_learning_session(save=True)
