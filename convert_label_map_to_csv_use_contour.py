#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from collections import namedtuple
import csv
import os.path
import random
import sys

import numpy as np
import PIL.Image
import shapely.geometry
import shapely.wkt
import skimage.segmentation
import skimage.measure as sk_measuer
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

import scipy.io as sio

def detect(model, image_path):
    '''
    Find few potential building candidates within an image, and return a list of tuple
    (polygon_wkt, confidence) of detections
    '''
    detections = []
    image = sio.loadmat(image_path)
    segments = image['inst_img']
    confiden = image['pred_dict']['boxes'][0][0]
    #print('{}'.format(confiden.shape))
    max_label = np.max(np.max(segments))
    #print('{}'.format(max_label))
    for i in range(1, max_label + 1):
        tmp_label_map = np.zeros(segments.shape, dtype=np.uint8)
        x, y = np.nonzero(segments == i)
        if len(x) < 4:
            #print('xlen{}'.format(image_path))
            continue
        tmp_label_map[x, y] = 1
        contours = sk_measuer.find_contours(tmp_label_map, 0)
        if len(contours) == 0:
            continue
        contour = contours[0]
	ring = ogr.Geometry(ogr.wkbLinearRing)
        pt_num = contour.shape[0]
        for j in range(0, pt_num):
            ring.AddPoint(int(contour[j][1]), int(contour[j][0])) #(x, y, 0)
        ring.CloseRings()
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        wkt = poly.ExportToWkt()
	#print('kml:{}'.format(wkt))
        # simple area and confidence heuristic based on polygon area and length
        # confidence formula
        confidence = confiden[i-1][4]
        if confidence >= 0.4:
        #confidence = 1
	# use shapely.wkt to build the polygon string
        #print('detect polygon:{}'.format((shapely.wkt.loads(ring))))
        #print('detect polygon:{}'.format((shapely.wkt.dumps(kml, old_3d = True, rounding_precision = 2))))
        #detections.append((shapely.wkt.dumps(wkt, old_3d = True, rounding_precision = 2), confidence))
            detections.append((wkt,confidence))
    # limit to max_detections per image based on confidence scores
    if len(detections) > model.max_detections:
        detections.sort(key=lambda x: x[1], reverse = True)
        detections = detections[:model.max_detections]
    return detections


def main():
    '''
    For all mat file, detect buildings and write results to a CSV file
    '''
    # image_folder = '../spacenet_TrainData/3band'  # folder with test images
    image_folder = '../MNC_post/results'  # folder with test images
    image_prefix_length = 6 # prefix to be removed is "3band_"
    image_subset = 0      # number of images to process (or 0 for all images)

    # load offline trained parameters (could also be a disk file with millions of parameters)
    Model = namedtuple('Model',
        ['max_size', 'min_size', 'hratio0', 'hratio1', 'max_detections'])
    model = Model(max_size = 2000, min_size = 20, hratio0 = 1.25,
        hratio1 = 3.0, max_detections = 300)

    image_list = [x for x in os.listdir(image_folder) if x[-4:]=='.mat']

    # create a csv file, which will comply with the contest format
    with open('../MNC_post/results.csv', 'w') as dest:
        writer = csv.writer(dest)
        #print('writer')
        # header line
        writer.writerow(['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])
        #print('writerrow')
        # loop over test images and detect building based on model

        for j, basename in enumerate(image_list):
            #print('j = {}'.format(j))
            #print('{}'.format(basename))
            #image_name = basename[image_prefix_length:-4]
            image_name = 'AOI_2_RIO_img'+str(j+1)
            basename = '3band_' + image_name + '.mat'
            print('{}'.format(basename))
	    #print(image_name)
            detections = []
            #print('detections')
            #print(len(detections))
            #print('len(detc)')
            detections = detect(model, os.path.join(image_folder, basename))
            # write to submission file
            if len(detections) == 0 :
                writer.writerow([image_name, -1, "POLYGON EMPTY", 1])
                #print('empty {}'.format(image_name))
            else :
                for i, detection in enumerate(detections):
                    polygon_wkt = detection[0]
                    print('polygon_wkt:{}'.format(polygon_wkt))
                    confidence = detection[1]
                    writer.writerow([image_name, i+1, polygon_wkt, confidence])
            # summary data for the image
            #print("{}, {}, {} polygons".format(j, image_name, len(detections)))
            #print('printttt')
            #if j == 2794:
    		#return

if __name__ == '__main__':
    #sys.exit(main(sys.argv))
    main()
