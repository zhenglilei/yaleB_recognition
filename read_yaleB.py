# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:56:12 2015

@author: root
"""

import leveldb
import os

# Make sure that caffe is on the python path:
caffe_root = '/home/liris/Downloads/caffe/'  
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

def main():

#    db_path = 'yaleb_train_leveldb'
 
    db_path = 'yaleb_test_leveldb'
    
    #---------------- readin data and write out database-----------
    
    if not os.path.exists(db_path):    
        raise Exception('no db_path exist.')
    
    db = leveldb.LevelDB(db_path)
    
    keystr = '{:0>8d}'.format(3)
    
    value = db.Get(keystr)
    
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    print(datum.channels, datum.height, datum.width)
    
    image = caffe.io.datum_to_array(datum)
    print(image)
    print(datum.label)

if __name__ == '__main__':
    main()