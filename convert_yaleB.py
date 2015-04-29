# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:20:13 2015

@author: root
"""

import leveldb
import struct
import os
import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '/home/liris/Downloads/caffe/'  
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

def main():
    data_filename = 'soclbp_train_yaleB'
    db_path = 'yaleb_train_leveldb'
    
#    data_filename = 'soclbp_test_yaleB'
#    db_path = 'yaleb_test_leveldb'
    
    #---------------------- reading header --------------------
    
    data_file = open(data_filename, 'rb')
    num, = struct.unpack('i', data_file.read(4))    
    print num
    dim, = struct.unpack('i', data_file.read(4))
    print dim
    
    #---------------- readin data and write out database-----------
    print ('Opening db')    
    
    if not os.path.exists(db_path):    
        os.mkdir(db_path)    
    
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = dim   
    datum.height = 1
    datum.width = 1    
    
    db = leveldb.LevelDB(db_path)
    batch = leveldb.WriteBatch()    
    
    for nid in range(num):
        label, = struct.unpack('i', data_file.read(4))
#        print label
        strfmt = '{:0}d'.format(dim)
#        print strfmt
        data = struct.unpack(strfmt, data_file.read(8*dim))
        data = np.asarray(data)
#        print data
        
        # pseudo-image
        image = data.reshape((datum.channels, datum.height, datum.width))        
        
        datum = caffe.io.array_to_datum(image,label)            
        keystr = '{:0>8d}'.format(nid)
        
        batch.Put( keystr, datum.SerializeToString() )
        
        if (nid+1)%1000 == 0: # write down the buffer every 1000 data
            print ( str(nid+1) + ' data passed')
            db.Write(batch, sync=True)
            batch = leveldb.WriteBatch()
        
    print ('----------')
    if (nid+1)%1000 != 0:
        db.Write(batch, sync=True)
        print ( 'totally ' + str(nid+1) + ' data' )
        

if __name__ == '__main__':
    main()