import sys
import yaml
from getData import get_generator
from context_ae import CEncoder
import numpy as np
import tensorflow as tf
import os

def main(arg):

    EXP_FOLDER= arg[0]

    print(arg[0])

    with open(EXP_FOLDER+"/custom.yaml", 'r') as stream:
        custom_data = yaml.safe_load(stream)

    
                
    JSON_PATHS_TRAIN = custom_data["JSON_PATHS_TRAIN"]
    CKPT_PERIOD = int(custom_data["CKPT_PERIOD"])
    BATCH_SIZE =  int(custom_data["BATCH_SIZE"])

    generator = get_generator(JSON_PATHS_TRAIN,BATCH_SIZE,128,damaged=False,dim_missing=64)

    ce = CEncoder(batch_size=BATCH_SIZE,dir_path = EXP_FOLDER,cpkt_period= CKPT_PERIOD)

    ce.train(generator)




if __name__ == "__main__":
    
    main(sys.argv[1:])

