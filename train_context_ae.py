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

    path_discriminator_=""
    path_generator=""
    start_iter=0

    if('checkpoint' in os.listdir(EXP_FOLDER)):
        with open(EXP_FOLDER+"/checkpoint", 'r') as stream:
            try:
                ckpts = yaml.safe_load(stream)
                last_ckpt = ckpts["model_checkpoint_path"]
                start_iter = int(last_ckpt.replace('ckpt_context_discriminator',''))

                path_discriminator_ = 'ckpt_context_discriminator'+str(start_iter)
                path_generator = 'ckpt_context_generator'+str(start_iter)
                #wn.load_weights(EXP_FOLDER+'/'+last_ckpt)
                

            except yaml.YAMLError as exc:
                print(exc)
    
                
    JSON_PATHS_TRAIN = custom_data["JSON_PATHS_TRAIN"]
    CKPT_PERIOD = int(custom_data["CKPT_PERIOD"])
    BATCH_SIZE =  int(custom_data["BATCH_SIZE"])
    MAX_ITER =  int(custom_data["MAX_ITER"])

    print("start_iter",start_iter)
    generator = get_generator(JSON_PATHS_TRAIN,BATCH_SIZE,128,damaged=False,dim_missing=64)

    ce = CEncoder(batch_size=BATCH_SIZE,dir_path = EXP_FOLDER,cpkt_period= CKPT_PERIOD,path_discriminator=path_discriminator_,path_generator=path_generator)

    ce.train(generator,start_iter=start_iter,max_iter=MAX_ITER)




if __name__ == "__main__":
    
    main(sys.argv[1:])

