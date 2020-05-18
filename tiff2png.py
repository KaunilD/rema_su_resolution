import shapefile
import numpy as np
import glob
import math
from osgeo import gdal, ogr, osr
import os
import argparse

OUT_SUFFIX = '../data/pre-processed/landsat/'
GEOTIF_PATH = '../data/geotiffs/landsat/tiles/*.tif'
SHP_PATH = '../data/shapefiles/shapefiles/landsat'

SHP_MASTER = '../data/shapefiles/PGC_LIMA_VALID_3031-84.shp'

def get_extent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform
        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext

def reproject_coords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.
        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

class Image(object):
    def __init__(self, path):
        self._path = path
        self._id = path.split("_")[1]
        self._polygons = []
        self._bbox = None
        self._image = None
        self._map = None

    def get_polygons(self):
        return self._polygons

    def get_poly(self, idx):
        return self._polygons[idx]

    def add_poly(self, poly):
        self.get_polygons().append(poly)
        return None

    def get_id(self):
        return self._id

    def get_path(self):
        return self._path

    def get_bbox(self):
        return self._bbox

    def set_image(self, data):
        self._image = data

    def set_bbox(self, data):
        self._bbox = data


def create_args():
    parser = argparse.ArgumentParser(
        description="Converts DEM Tiffs to PNG"
    )
    parser.add_argument(
        "--inp",
        type=str,
        help="input DEM.",
    )

    parser.add_argument(
        "--out",
        type=str,
        help="out file name to write the processed DEM to.",
    )

    parser.add_argument(
        "--max",
        type=float,
        default=1.0,
        help="max value for min-max normalization"
    )

    parser.add_argument(
        "--min",
        type=float,
        default=0.0,
        help="min value for min-max normalization"
    )

    return parser.parse_args()

def get_image_name(string):
    fn = os.path.basename(string)
    return fn.split(".")

if __name__ == "__main__":
    args = create_args()

    driver = ogr.GetDriverByName("ESRI Shapefile")

    print("Processing: {}".format(args.inp))

    name, ext = get_image_name(args.inp)

    tif_ds = gdal.Open(args.inp)
    print(tif_ds.GetRasterBand(1).GetStatistics(0, 0))
    tif_ds = gdal.Translate(
        "{}_proc.tif".format(name), args.inp,
        format='GTiff', outputType=gdal.GDT_Byte,
        bandList=[1],
        scaleParams=[
            [tif_ds.GetRasterBand(1).GetStatistics(0, 1)[0], tif_ds.GetRasterBand(1).GetStatistics(0, 1)[1]],
        ],
    )

    gt = tif_ds.GetGeoTransform()
    w = tif_ds.RasterXSize
    h = tif_ds.RasterYSize

    ext_ = get_extent(gt, w, h)
    # get raster data
    print("Rasterizing {} to {}".format(args.inp, args.out))
    print("\t",ext_)
    print("\tWIDTH = {} HEIGHT = {}".format(w, h))
    ds = gdal.Translate(
        destName=args.out, srcDS=tif_ds,
        options=gdal.TranslateOptions(
            width = w,
            height = h,
            format='PNG'
            # scaleParams=[args]
        )
    )
