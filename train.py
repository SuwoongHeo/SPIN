from utils import TrainOptions
from train import Trainer
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
