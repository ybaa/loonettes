from src.experiments.myDatasetAndDivisionDetector import MyDatasetAndDivisionDetectorManager
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # turn off warnings


# detection testing
exp1 = MyDatasetAndDivisionDetectorManager()
exp1.test_detection()
