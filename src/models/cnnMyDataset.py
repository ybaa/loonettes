from src.models.cnnBase import CNNBase


class CNNMyDataset(CNNBase):

    def __init__(self):
        pass

    def predict_single_image(self, img):
        print('hi! i am supposed to do prediction but i only print img shape: ' + str(img.shape))