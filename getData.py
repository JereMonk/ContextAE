import numpy as np
from monk import BBox
import tensorflow as tf
from monk import Dataset
import json
import PIL
from monk.utils.s3.s3path import S3Path

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,json_paths, batch_size=10, dim=(128,128),dim_missing=(50,50), n_channels=3,shuffle=True,damaged=False):
        
        
        
        self.shuffle = shuffle 
        self.dim = dim 
        self.batch_size = batch_size  
        self.n_channels = n_channels
        self.damaged=damaged
        self.dim_missing=dim_missing
        
        jsons_data=[]
        
        for json_path in json_paths:
            with open(json_path) as f:
                json_data = json.load(f)
            jsons_data.append(json_data)
            
        self.filter_json(jsons_data,damaged)

    
    def mask_randomly(self, imgs,overlap=7):
        imgs=imgs+1
        y1 = np.random.randint(0, self.dim[0] - self.dim_missing[0])
        y2 = y1 + self.dim_missing[0]
        x1 = np.random.randint(0, self.dim[1]  - self.dim_missing[1])
        x2 = x1 + self.dim_missing[1]
   
        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty(( self.dim_missing[0], self.dim_missing[1], 3))
        
        masked_img = imgs.copy()

        _y1, _y2, _x1, _x2 = y1, y2, x1, x2
        missing_parts = masked_img[_y1:_y2, _x1:_x2].copy()
        masked_img[_y1 + overlap :_y2 - overlap, _x1 + overlap:_x2 - overlap] = 0
        masked_imgs = masked_img

        masked_imgs=masked_imgs-1
        missing_parts= missing_parts-1
        return masked_imgs, missing_parts, (y1, y2, x1, x2)
        
    def filter_json(self,jsons_data,damaged):
        
        
        filtered_json =[]

        for json_data in jsons_data :
        
            for i in range(0,len(json_data)):
                
                
                if json_data[i]["repair_action"]=='not_damaged' and damaged==False :
                    filtered_json.append(json_data[i])
                elif json_data[i]["repair_action"]!='not_damaged' and json_data[i]["repair_action"]!=None and damaged==True and (json_data[i]["label"]=='scratch' or json_data[i]["label"]=='dent' ) :
                    filtered_json.append(json_data[i])
               
        
        self.filtered_json=filtered_json
        self.list_IDs = np.arange(len(filtered_json)) 
        self.indexes = np.arange(len(filtered_json))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        

    
    def load_image(self,id):
        
        data = self.filtered_json[id]
        
        
        if('s3:/monk-client-images/' in data["path"]):
            bucket = "monk-client-images"
            key = data["path"].replace("s3:/monk-client-images/","")
            s3 = S3Path(bucket,key)
            im = PIL.Image.open(s3.download())
        else:
            im = PIL.Image.open(data["path"])
        
        bbox =  data["part_bbox"]
        img_crop = im.crop(bbox)
        img_crop = img_crop.resize(self.dim)
        
        img_crop = ((((np.array(img_crop)/255)*2)-1)).astype(np.float32)
        masked_img, missing, _ = self.mask_randomly(img_crop)

        return( img_crop,masked_img, missing )
        #return(np.array(img_crop).astype(np.float32))
        
    def __len__(self):
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        IMG,MASKED,MISSING = self.__data_generation(list_IDs_temp)

        return IMG,MASKED,MISSING

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        IMG = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
        MASKED = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
        MISSING = np.empty((self.batch_size, *self.dim_missing, self.n_channels),dtype=np.float32)
        
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img_crop,masked_img, missing = self.load_image(self.indexes[ID])
            IMG[i,] = img_crop
            MASKED[i,] = masked_img
            MISSING[i,] = missing

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        IMG = tf.convert_to_tensor(IMG)
        MASKED = tf.convert_to_tensor(MASKED)
        MISSING= tf.convert_to_tensor(MISSING)
        return IMG,MASKED,MISSING

def get_generator(json_paths,batch_size,size,dim_missing,damaged=False):
    
    generator = DataGenerator(json_paths,batch_size=batch_size,dim=(size,size),dim_missing=(dim_missing,dim_missing),damaged=damaged)

    return(generator)