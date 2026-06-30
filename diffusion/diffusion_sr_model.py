import sys
import os

from simplesr.models.base_model import BaseModel

class DiffusionSRModel(BaseModel):
    def __init__(self,opt):
        super().__init__(opt)

    def feed_data(self, data):
        ...

    def optimize_parameters(self, current_iter):
        ...

    def test(self):
        ...

    def get_current_visuals(self):
        ...

    def nondist_validation(self, dataloader, current_iter, wandb_logger, save_img):
        ...

    def save(self, epoch, current_iter):
        ...

if __name__ == '__main__':
    pass
