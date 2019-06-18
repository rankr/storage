#coding: utf-8
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py

import os
import cv2

def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def mat2xml(mat_path, xml_dir):
    train_data = h5py.File(mat_path + '/digitStruct.mat', 'r')
    for img_id in range(len(train_data['/digitStruct/bbox'])):
        if img_id % 1000 == 0:
            print("achieved", img_id)
        info = get_box_data(img_id, train_data)
        name = get_name(img_id, train_data)

        im = cv2.imread((mat_path) + '/' + str(img_id + 1) + '.png')
        height, width, rgb  = im.shape
        xml_file = open((xml_dir + '/' + str(img_id) + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>svhn</folder>\n')
        xml_file.write('    <filename>' + str(img_id) + '.png' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')
        for j in range(len(info['height'])):
            x,y,w,h,la = info['left'][j], info['top'][j], info['width'][j], info['height'][j], info['label'][j], 

            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + str("svhn-num") + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(x) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(y) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(x+width) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(y+height) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')
        xml_file.close()

def prepare_(data_path):
    #根据数据集准备yolo数据
    train_data = h5py.File(data_path + '/digitStruct.mat', 'r')

    for img_id in range(0, len(train_data['/digitStruct/bbox'])):
        if img_id % 1000 == 0:
            print("achieved", img_id)
        info = get_box_data(img_id, train_data)
        name = get_name(img_id, train_data)

        im = cv2.imread((data_path) + '/' + str(img_id + 1) + '.png')
        height, width, rgb  = im.shape

        to_write_path = os.path.join(data_path, "%d.txt"%(img_id+1))
        #if os.path.exists(to_write_path):
        #    continue
        with open(to_write_path, 'w') as wr:
            for j in range(len(info['height'])):
                x,y,w,h,la = info['left'][j], info['top'][j], info['width'][j], info['height'][j], info['label'][j], 
                if la == 10:#原标注把0标注成了10， 详见test/2000.txt & test/2000.png
                    la = 0
                wr.write('%d %f %f %f %f\n'%(la, (x + w)/width, (y + h)/height, w/width, h/height))
        #exit()

def prepare(train_dir, test_dir, txt_path):
    prepare_(train_dir)
    prepare_(test_dir)
    with open(os.path.join(txt_path, 'svhn-train.txt'), 'w') as w:
        train_data = h5py.File(train_dir + '/digitStruct.mat', 'r')
        for img_id in range(1, len(train_data['/digitStruct/bbox'])+1):
            w.write(os.path.abspath(train_dir + '/%d.png\n'%(img_id)))
    with open(os.path.join(txt_path, 'svhn-test.txt'), 'w') as w:
        test_data = h5py.File(test_dir + '/digitStruct.mat', 'r')
        for img_id in range(1, len(test_data['/digitStruct/bbox'])+1):
            w.write(os.path.abspath(test_dir + '/%d.png\n'%(img_id)))


#mat2xml("/Users/apple/Documents/file5/deep learning/hw3/train", "data/Annotations")

import sys
if __name__ == '__main__':
    a = sys.argv
    prepare(a[1], a[2], a[3])


