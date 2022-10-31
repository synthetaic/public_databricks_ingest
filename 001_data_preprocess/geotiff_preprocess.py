# Databricks notebook source
# !rm -r /dbfs/mnt/dev-databricks/output/dbricks_full_pipeline_test_upload0002_8lite/SLH_50378_ICEYE_X8_GRD_SLH_50378_20210403T183125/17/*
# !ls /dbfs/mnt/dev-databricks/output/dbricks_full_pipeline_test_upload0002_8lite/SLH_50378_ICEYE_X8_GRD_SLH_50378_20210403T183125/17/75870

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "100000")

try:
    cluster_count = dbutils.widgets.get("cluster_count")
    if cluster_count is None:
        cluster_count = 4
    else:
        cluster_count = int(cluster_count)
except:
    cluster_count = 4
    
    
import uuid
from typing import *
import glob
import os
import sys
import json
import subprocess
import math
from time import time
import tqdm

import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2

import gdal
import pygeotile
from pygeotile.point import Point
from pygeotile.tile import Tile

import pyspark
from pyspark import TaskContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import *


# COMMAND ----------

def mount_folder(
    storageAccount:str, 
    containerName:str, 
    folderName:str, 
    appRegistrationId:str="5da2fe00-7ce9-4be8-8f11-57befc2cf831", 
    directoryId:str="628574bc-9c62-4216-afdd-7278cbd0123a"
) -> None:
    clientSecret = dbutils.secrets.get(scope = "azure-ingest-key-vault", key = "dev-raic-ingest-adlsgen2-001")

    storageSource = f"abfss://{containerName}@{storageAccount}.dfs.core.windows.net/{folderName}"
    fullMountLocationName = f"/mnt/{containerName}/{folderName}"

    configs = {
        "fs.azure.account.auth.type"              : "OAuth"
        , "fs.azure.account.oauth.provider.type"    : "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
        , "fs.azure.account.oauth2.client.id"       : appRegistrationId
        , "fs.azure.account.oauth2.client.secret"   : clientSecret
        , "fs.azure.account.oauth2.client.endpoint" : f"https://login.microsoftonline.com/{directoryId}/oauth2/token"
    }

    dbutils.fs.mount(
        source = storageSource,
        mount_point = fullMountLocationName,
        extra_configs = configs
    )

def list_mounts() -> None:
    display(dbutils.fs.mounts()) 

def unmount_folder(
    dest_folder_path:str
) -> None:
    dbutils.fs.unmount(dest_folder_path)

# COMMAND ----------

TILE_SIZE = 256
def print_time(
    s:float,
    e:float
) -> Tuple[float, float, float, float]:
    "Print elapsed time in days:hh:mm:ss format"
    dif = e-s
    d = dif // 86400
    h = dif // 3600 % 24
    m = dif // 60 % 60
    s = dif % 60
    print("@"*50,"Elapsed time ==> Days: {} Hours: {:}   Minutes:{:}    Seconds:{:.2f}".format(d,h,m,s))
    return d,h,m,s


@pandas_udf(IntegerType())
def main_process_uniques(
    batch_row_id:pd.Series,
    batch_row_path:pd.Series,
    batch_index:pd.Series,
    batch_outputDirectory:pd.Series, 
    batch_level:pd.Series,
    mod:pd.Series=['png4']
) -> pd.Series:
    spark_ret = []
    for row_id, row_path, index, outputDirectory, level in tqdm.tqdm(zip(batch_row_id, batch_row_path, batch_index, batch_outputDirectory, batch_level), total=len(batch_index), position=0):
        if index%100000 ==0:
            print(f"   Processing file {index}=> {row_id}")
        parts = row_id.split('-')
        x = parts[1]
        y = parts[-1]
        assert len(row_path) ==1,f"Error: {row_path} is not a unique tile" # means there is no overlapping tile=> So just copy it over to the target folder
        c_img = cv2.imread(row_path[0], cv2.IMREAD_UNCHANGED)
        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(c_img)
        imgarr = np.array(img)
        try:
            a1 = np.where(imgarr[:,:,3]==0) # blank (transparent) pixels
            if a1[0].shape[0]!=img.size[0]**2:#then it is not a completely transparent
                if mod[0]=='jpg':
                    img.convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.jpg",format = 'JPEG',quality = 95)
                elif mod[0]=='png3':
                    img.convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.png",format = 'PNG')
                elif mod[0] == 'png4':
                    img.convert('RGBA').save(f"{outputDirectory}/{level}/{x}/{y}.png",format = 'PNG')
                else:
                    raise ValueError(" mod must be 'jpg', 'png3', or 'png4'")
                    spark_ret.append(-1)
                    continue
            spark_ret.append(0)
        except:
            spark_ret.append(-1)
            continue
            
    return pd.Series(spark_ret)
        
@pandas_udf(IntegerType())
def main_process_duplicates(
    batch_row_id:pd.Series,
    batch_row_path:pd.Series,
    batch_index:pd.Series,
    batch_outputDirectory:pd.Series, 
    batch_level:pd.Series,
    mod:pd.Series=['png4']
) -> pd.Series:
    spark_ret = []
    for row_id, row_path, index, outputDirectory, level in tqdm.tqdm(zip(batch_row_id, batch_row_path, batch_index, batch_outputDirectory, batch_level), total=len(batch_index), position=0):
        if index%100000 ==0:
            print(f"   Processing file {index}=> {row_id}")
        parts = row_id.split('-')
        x = parts[1]
        y = parts[-1]
        assert len(row_path)>1, f"Error: {row_path} is not a duplicate tile" # means there is no overlapping tile=> So just copy it over to the target folder
        #Process overlapping tiles
        tiles = []
        valued_pixels = []
        ch1_zeros = []
        ch2_zeros = []
        ch3_zeros = []
        mean = []
        std=[]
        for path in row_path:
            c_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            c_img = cv2.cvtColor(c_img, cv2.COLOR_BGRA2RGBA)
            img = Image.fromarray(c_img)
            imgarr = np.array(img)
            try:
                a1 = np.where(imgarr[:,:,3]!=0)
                if a1[0].shape[0] !=0: # if the tile is not transparent
                    tiles.append(img)
                    valued_pixels.append(a1[0].shape[0])
                    ch1_zeros.append(np.where(imgarr[:,:,0]==0)[0].shape[0])
                    ch2_zeros.append(np.where(imgarr[:,:,1]==0)[0].shape[0])
                    ch3_zeros.append(np.where(imgarr[:,:,2]==0)[0].shape[0])
                    mean.append(imgarr[:,:,:3].flatten().mean())
                    std.append(imgarr[:,:,:3].flatten().std())
            except:
                continue
        if len(tiles)>0: # if there exist at least one overlapping tile which is not transparent
            df = pd.DataFrame(
                    {'images': tiles,
                    'valued_pixels': valued_pixels,
                    'ch1_zeros': ch1_zeros,
                    'ch2_zeros': ch2_zeros,
                    'ch3_zeros': ch3_zeros,
                    'mean':mean,
                    'std':std
                    })
            df = df.sort_values('valued_pixels',ascending=False)
            if len(df['valued_pixels'].unique()) == 1: #and df.iloc[0]['valued_pixels'] == TILE_SIZE * TILE_SIZE: # duplicate images that all of them have meaningful values according to their 4th channel
                df = df.sort_values('mean',ascending=False)# sort the images according to their mean value across RGB and copy the one with highest mean
            #     df.iloc[0]['images'].convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.jpg",format = 'JPEG',quality = 95)
            # else:# any other overlapping tiles with different number of meaningful values in their 4th channel
            #     df.iloc[0]['images'].convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.jpg",format = 'JPEG',quality = 95)
            if mod[0]=='jpg':
                df.iloc[0]['images'].convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.jpg",format = 'JPEG',quality = 95)
            elif mod[0] == 'png3':
                df.iloc[0]['images'].convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.png",format = 'PNG')
            elif mod[0] == 'png4':
                df.iloc[0]['images'].convert('RGBA').save(f"{outputDirectory}/{level}/{x}/{y}.png",format = 'PNG')
            else:
                raise ValueError(" mod must be 'jpg', 'png3', or 'png4'")
                spark_ret.append(-1)
            spark_ret.append(0)
        else:
            spark_ret.append(-1)

    return pd.Series(spark_ret)

def mergeTiles_databricks(
    inputDirectory:str,
    outputDirectory:str,
    level:str
) -> None:    
    print("processing folder "+inputDirectory)
    if not os.path.exists(outputDirectory+f"/{level}"):
        os.makedirs(outputDirectory+f"/{level}")

    pngfiles=glob.glob(inputDirectory + f'/**/{level}/**/*.png', recursive=True)# To consider level <level> only
    print(f"\ntotal number of files being processed: {len(pngfiles)}")
    st=time()
    print(f"\nDataframe is being created...")
    df = pd.DataFrame(pngfiles, columns = ['Path'])
    print(f"finished: {time()-st} sec")

    st=time()
    print(f"\ncolumn Z is being added...")
    df['z'] = df['Path'].apply(lambda x: str(x).rsplit('/',4)[2])
    print(f"finished: {time()-st} sec")

    st=time()
    print(f"\ncolumn X is being added...")
    df['x'] = df['Path'].apply(lambda x: str(x).rsplit('/',4)[3])
    print(f"finished: {time()-st} sec")

    st=time()
    print(f"\ncolumn Y is being added...")
    df['y'] = df['Path'].apply(lambda x: str(x).rsplit('/',4)[4].split('.')[0])
    print(f"finished: {time()-st} sec")

    st=time()
    print(f"\ncolumn ID is being added...")
    df['ID']=df['z']+'-'+df['x']+'-'+df['y']
    print(f"finished: {time()-st} sec")

    st=time()
    print(f"\ncalculating unique ids...")
    xs = df['x'].unique()
    print(f"finished: {time()-st} sec")

    st=time()
    print(f"\nmaking directories...")
    [os.makedirs(f"{outputDirectory}/{level}/{tmp}", exist_ok=True) for tmp in xs]
    print(f"finished: {time()-st} sec")

    st=time()
    print(f"\ngrouping df by ID...")
    aggdf = df.groupby('ID').agg({'Path': list}).reset_index()
    print(f"finished: {time()-st}")

    st=time()
    print(f"\nSeparating duplicate and unique dataframes...")
    aggdf['duplicate'] = aggdf['Path'].apply(lambda x: len(x))
    len_aggdf = len(aggdf.index)
    duplicates = aggdf[aggdf['duplicate']!=1].reset_index(drop=True)
    num_duplicates = len(duplicates)
    uniques = aggdf[aggdf['duplicate']==1].reset_index(drop=True)
    num_uniques = len(uniques)
    print(f"finished: {time()-st}")

    print("\n\n{:20s}: {:10d} \n{:20s}: {:10d}\n{:20s}: {:10d}".format("unique tiles",num_uniques,"duplicate tiles",num_duplicates,"all",len_aggdf))    
    assert num_uniques + num_duplicates == len_aggdf,f"num_uniques + num_duplicates not equal all => {num_uniques} + {num_duplicates} != {len_aggdf}"
    global single_blank
    single_blank = 0
    global duplicate_blank
    duplicate_blank = 0
    global copied
    copied =0

    fls = glob.glob(os.path.join(outputDirectory,"**/*.jpg"), recursive=True)
    print(f"{len(fls)} jpg files already exist in the output diretory {outputDirectory}/{level}")
    fls = glob.glob(os.path.join(outputDirectory,"**/*.png"), recursive=True)
    print(f"{len(fls)} png files already exist in the output diretory {outputDirectory}/{level}")


    print(f"\n\nProcessing tiles started...")    
    print(f"Total files (aggregated): {len_aggdf}")
    cpu_count = int(os.cpu_count())
    print(f"Available cpus: {cpu_count}")

    # for unique tiles
    print(f"Running the code for {num_uniques} *unique* tiles:")
    s = time()

    print(uniques.columns)
    unique_df = spark.createDataFrame(([(row['ID'], row['Path'], index, outputDirectory, level) for index, row in uniques.iterrows()]), ["batch_row_id", "batch_row_path", "batch_index", "batch_outputDirectory", "batch_level"])
    unique_df = unique_df.coalesce(cluster_count)
    unique_df = unique_df.select(main_process_uniques(col("batch_row_id"), col("batch_row_path"), col("batch_index"), col("batch_outputDirectory"), col("batch_level"),).alias("res"))
    res_df = unique_df.toPandas()
    
    e = time()
    print_time(s,e)

    # for duplicate tiles
    print(f"\n\n Running the code for {num_duplicates} *overlapping* tile:")
    s = time()

    duplicates_df = spark.createDataFrame(([(row['ID'], row['Path'], index, outputDirectory, level) for index, row in duplicates.iterrows()]), ["batch_row_id", "batch_row_path", "batch_index", "batch_outputDirectory", "batch_level"])
    duplicates_df = duplicates_df.coalesce(cluster_count)
    duplicates_df = duplicates_df.select(main_process_duplicates(col("batch_row_id"), col("batch_row_path"), col("batch_index"), col("batch_outputDirectory"), col("batch_level"),).alias("res"))
    res_df = duplicates_df.toPandas()

    e = time()
    print_time(s,e)

    print("\n\n\n","+_"*15)
    print(f"Total files (initial): {len(pngfiles)}")
    print(f"Total files (aggregated): {len_aggdf}")
    print(f"Created folder under level {df['z'].iloc[0]}: {len(xs)}")
    print(f"Output directory: {outputDirectory}/{level}")

# COMMAND ----------

def convert_coords_to_latlong(
    x:int, 
    y:int, 
    z:int, 
    standard:str="TMS"
) -> Tuple[float, float]:
    """Convert coordinates to latitude and longitude.

    Args:
        x (int): X coordinate.
        y (int): Y coordinate.
        z (int): Z coordinate.
        standard (str, optional): Geospatial reference standard of the coordinates. Defaults to "TMS".

    Returns:
        Tuple[float, float]: Centered latitude, and Centered longitude of the tile.
    """
    tile = Tile.from_tms(x, y, z) if standard == "TMS" else Tile.from_google(x, y, z)
    sw, ne = tile.bounds
    cen_lat = (sw[0] + ne[0]) / 2
    cen_long = (sw[1] + ne[1]) / 2
    return (cen_lat, cen_long)

def get_lat_long_tms_tile(
    latitude:float, 
    longitude:float, 
    z:int
) -> pygeotile.tile.Tile:
    """Get a TMS-standard Tile object in from a latitude and longitude.

    Args:
        latitude (float): Point's latitude.
        longitude (float): Point's longitude.
        z (int): Zoom level to get tile coordinates.

    Returns:
        pygeotile.tile.Tile: TMS-standard Tile object containing coordinates.
    """
    tms_tile = Tile.for_latitude_longitude(latitude, longitude, z) 
    return tms_tile

def convert_latlong_to_coords(
    latitude:float, 
    longitude:float, 
    z:int, 
    standard:str="TMS"
) -> Tuple[int, int, int]:
    """Convert latitude and longitude to coordinates.

    Args:
        latitude (float): Latitude.
        longitude (float): Longitude.
        z (int): Zoom level for coordaintes.
        standard (str, optional): Geospatial reference standard to get. Defaults to "TMS".

    Returns:
        Tuple[int, int, int]: Z, X, Y coordinates.
    """
    tms_tile = get_lat_long_tms_tile(latitude, longitude, z)
    return tms_tile.tms if standard == "TMS" else tms_tile.google

def get_zxy_latlong_from_fpath(
    tile_fpath:str,
    standard:str="TMS"
) -> Tuple[int, int, int, float, float]:
    """Get z,x,y coordinates and latitude and longitude from a file path.

    Args:
        tile_fpath (str): File path.
        standard (str, optional): Geospatial reference standard of the file path. Defaults to "TMS".

    Returns:
        Tuple[int, int, int, float, float]: Z, X, Y, latitude, and longitude.
    """
    z = tile_fpath.split('/')[-3]
    x = tile_fpath.split('/')[-2]
    y = os.path.splitext(os.path.split(tile_fpath)[1])[0]
    central_latitude, central_longitude = convert_coords_to_latlong(int(x), int(y), int(z), standard=standard)
    return (int(z), int(x), int(y), central_latitude, central_longitude)

def convert_fpath_geostandards(
    cur_fpath:str,
    cur_standard="WMS"
) -> Tuple[str, str, str, str]:
    """Convert filepath in one geospatial standard to the opposite.

    Args:
        cur_fpath (str): File path to convert.
        cur_standard (str, optional): Geospatial reference standard of the input file path. Defaults to "WMS".

    Returns:
        Tuple[str, str, str, str]: Converted filepath, Z, X, and Y.
    """
    z = cur_fpath.split('/')[-3]
    x = cur_fpath.split('/')[-2]
    y = os.path.splitext(os.path.split(cur_fpath)[1])[0]
    img_ext = os.path.splitext(os.path.split(cur_fpath)[1])[1]
    map_root_dir = cur_fpath.split(f'/{z}/')[0]
    cen_lat, cen_long = convert_coords_to_latlong(int(x), int(y), int(z), standard=cur_standard)
    convert_standard = "TMS" if cur_standard == "WMS" else "WMS"
    conv_z = z
    conv_x, conv_y = convert_latlong_to_coords(cen_lat, cen_long, int(z), standard=convert_standard)
    conv_fpath = os.path.join(map_root_dir, str(conv_z), str(conv_x), f"{str(conv_y)}.{img_ext}")
    return conv_fpath, conv_z, conv_x, conv_y

def work_merge_anchor_x(
    args:List[object]
    ):
    """Distributed function.

    Args:
        args (List[object]): Arguments.
    """
    build_x_anchor(*args)

def merge_anchor_x(
    zoom_dir:str, 
    next_zoom_dir:str, 
    src_anchor_x:int, 
    ext:str, 
    tile_size:int, 
    next_anchor_y:int, 
    channels:int, 
    standard:str="TMS"
) -> int:
    """Merge 4 component tiles into one higher zoom tile.

    Args:
        zoom_dir (str): Zoom directory of source tiles.
        next_zoom_dir (str): Zoom directory of saved tiles.
        src_anchor_x (int): X anchor of the 4 tiles.
        ext (str): File extension to save as.
        tile_size (int): Tile size.
        next_anchor_y (int): Y anchor to use for the next higher zoom.
        channels (int): Image channel.
        standard (str, optional): Geospatial reference standard. Defaults to "TMS".

    Returns:
        int: 1 on success and 0 on failure.
    """
    nxt_anchor_x = int(src_anchor_x / 2)
    src_anchor_y = next_anchor_y * 2

    # Find neighbor composites in MMS Form
    cur_zoom_anchor = os.path.join(zoom_dir, str(src_anchor_x), f'{str(src_anchor_y)}.{ext}')
    right = os.path.join(zoom_dir, str(src_anchor_x), f'{str(src_anchor_y + 1)}.{ext}')
    down = os.path.join(zoom_dir, str(src_anchor_x + 1), f'{str(src_anchor_y)}.{ext}')
    down_right = os.path.join(zoom_dir, str(src_anchor_x + 1), f'{str(src_anchor_y + 1)}.{ext}')
        
    anchor_file_name = os.path.join(next_zoom_dir, str(nxt_anchor_x), f'{str(next_anchor_y)}.{ext}')
    try:
        new_image = np.zeros((tile_size * 2, tile_size * 2, channels), np.uint8)

        if os.path.exists(cur_zoom_anchor):
            cur_zoom_anchor = cv2.imread(cur_zoom_anchor, cv2.IMREAD_UNCHANGED)
            new_image[0:tile_size, 0:tile_size, :] = cur_zoom_anchor

        if os.path.exists(down):
            down = cv2.imread(down, cv2.IMREAD_UNCHANGED)
            new_image[0:tile_size, tile_size:tile_size * 2, :] =  down

        if os.path.exists(right):
            right = cv2.imread(right, cv2.IMREAD_UNCHANGED)
            new_image[tile_size:tile_size * 2, 0:tile_size, :] = right

        if os.path.exists(down_right):
            down_right = cv2.imread(down_right, cv2.IMREAD_UNCHANGED)
            new_image[tile_size:tile_size * 2, tile_size:tile_size * 2, :] = down_right
        
        if standard == "TMS":
            tophalf = new_image[0:tile_size, : , :]
            bothalf = new_image[tile_size:tile_size*2:, : , :]

            new_image = np.zeros((tile_size * 2, tile_size * 2, channels), np.uint8)

            new_image[tile_size:tile_size*2, :, :] = tophalf
            new_image[0:tile_size,:, :] = bothalf

        new_image = cv2.resize(new_image, (tile_size, tile_size))
 
        if ext in ['jpg', 'jpeg']:
            cv2.imwrite(anchor_file_name, new_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(anchor_file_name, new_image)

        return 1
    except:
        return 0


def valid_xy_anchor(
    zoom_dir:str, 
    src_anchor_x:str, 
    next_anchor_y:str, 
    ext:str
) -> bool:
    """Filtering function to determine if the next y anchor is valid. At least one of the four tile exists to make it valid.

    Args:
        zoom_dir (str): Directory of the current zoom.
        src_anchor_x (str): X anchor of the current zoom.
        next_anchor_y (str): Possible Y anchor of the next zoom.
        ext (str): Image file extension.

    Returns:
        bool: True if one of the four components exist, False if none do.
    """
    src_anchor_y = next_anchor_y * 2

    cur_zoom_anchor = os.path.join(zoom_dir, str(src_anchor_x), f'{str(src_anchor_y)}.{ext}')
    right = os.path.join(zoom_dir, str(src_anchor_x), f'{str(src_anchor_y + 1)}.{ext}')
    down = os.path.join(zoom_dir, str(src_anchor_x + 1), f'{str(src_anchor_y)}.{ext}')
    down_right = os.path.join(zoom_dir, str(src_anchor_x + 1), f'{str(src_anchor_y + 1)}.{ext}')
    if os.path.exists(cur_zoom_anchor) or os.path.exists(right) or os.path.exists(down) or os.path.exists(down_right): 
        return True
    else:
        return False

@pandas_udf(IntegerType())
def build_x_anchor(
    batch_x:pd.Series, 
    batch_next_zoom_dir:pd.Series, 
    batch_cur_zoom_dir:pd.Series, 
    batch_ext:pd.Series, 
    batch_tile_size:pd.Series, 
    batch_channels:pd.Series, 
    batch_standard:pd.Series
) -> pd.Series:
    """Build the tiles for an X anchor of the next level. Filters good Y anchors of the next level first.

    Args:
        x (int): X anchor to build.
        next_zoom_dir (str): Next zoom save directory.
        cur_zoom_dir (str): Current zoom directory.
        ext (str): Image file extension.
        tile_size (int): Tile size.
        channels (int): Image channels.
        standard (str): Geospatial reference standard.
    """
    spark_ret = []
    for x, next_zoom_dir, cur_zoom_dir, ext, tile_size, channels, standard in tqdm.tqdm(zip(batch_x, batch_next_zoom_dir, batch_cur_zoom_dir, batch_ext, batch_tile_size, batch_channels, batch_standard), total=len(batch_x), position=0):
        cur_y_anchors = set()
        try:
            cur_y_anchors_x1 = glob.glob(os.path.join(cur_zoom_dir, str(x + 1), '*'))
            cur_y_anchors.update(set(cur_y_anchors_x1))

            if os.path.exists(os.path.join(cur_zoom_dir, str(x))):
                if len(cur_y_anchors) < len(os.listdir(os.path.join(cur_zoom_dir, str(x)))):
                    cur_y_anchors_x0 = glob.glob(os.path.join(cur_zoom_dir, str(x), '*'))
                    cur_y_anchors.update(set(cur_y_anchors_x0))
        # Left boundary has one xtile row
        except:
            cur_y_anchors2 = glob.glob(os.path.join(cur_zoom_dir, str(x), '*'))
            cur_y_anchors.update(set(cur_y_anchors2))

        cur_y_anchors = [int((tmp.split('/')[-1]).split('.')[0]) for tmp in list(cur_y_anchors)]

        if len(cur_y_anchors) > 0:
            min_cur_y_anchors = np.amin(cur_y_anchors) - 1
            max_cur_y_anchors = np.amax(cur_y_anchors) + 1
            actual_cur_y_anchors = list(range(min_cur_y_anchors, (max_cur_y_anchors+1)))
            nxt_anchor_y = [math.floor(tmp / 2) for tmp in actual_cur_y_anchors]
            nxt_anchor_y = list(set(nxt_anchor_y))
            valid_nxt_anchor_y = [a_y for a_y in nxt_anchor_y if valid_xy_anchor(cur_zoom_dir, x, a_y, ext)]

            s = [merge_anchor_x(cur_zoom_dir, next_zoom_dir, x, ext, tile_size, ancy, channels, standard) for ancy in valid_nxt_anchor_y]
            spark_ret.append(0)
        else:
            spark_ret.append(-1)
    
    return pd.Series(spark_ret)

class LayerBuilder():
    def __init__(
        self, 
        root_data_dir:str, 
        start_zoom:str, 
        end_zoom:str, 
        workers:int, 
        ext:str='png', 
        tile_size:int=256, 
        standard:str='TMS'
    ):
        """Layer Building class to build higher zoom levels.

        Args:
            root_data_dir (str): Source data directory.
            start_zoom (str): Zoom to start tiling at.
            end_zoom (str): Zoom to end tiling at.
            workers (int): Workers to use
            ext (str, optional): Image file extension. Defaults to 'png'.
            tile_size (int, optional): Tile size. Defaults to 256.
            standard (str, optional): Geospatial reference standard. Defaults to 'TMS'.
        """
        self.root_data_dir = root_data_dir
        self.start_zoom = start_zoom
        self.end_zoom = end_zoom
        self.workers = workers
        self.ext = ext
        self.tile_size = tile_size
        self.standard = standard

        _, _, self.channels = cv2.imread(glob.glob(os.path.join(self.root_data_dir, self.start_zoom, '**', f'*.{ext}'))[0], cv2.IMREAD_UNCHANGED).shape

    def create_x_columns(
        self, 
        zoom_dir:str, 
        next_zoom_dir:str
    ) -> List[int]:
        """Create the possible X anchor directory for the next level.

        Args:
            zoom_dir (str): Current zoom directory.
            next_zoom_dir (str): Next zoom level directory.

        Returns:
            List[int]: List of the good X anchors for the next level.
        """
        cur_x = os.listdir(zoom_dir)
        cur_x = [int(tmp) for tmp in cur_x]
        cur_x = sorted(cur_x)
        next_x = [math.floor(tmp / 2) for tmp in cur_x]
        next_x = list(set(next_x))
        edges = [(np.amin(next_x) + 1), (np.amax(next_x) + 1)]
        next_x = next_x + edges

        [os.makedirs(os.path.join(next_zoom_dir, str(an_x)), exist_ok=True) for an_x in next_x]

        return next_x

    def build_one_layer(
        self, 
        root_data_dir:str, 
        z:int
    ) -> None:
        """Engine function to build one zoom layer.

        Args:
            root_data_dir (str): Source data directory.
            z (int): Next zoom level to build.
        """
        print(f"Building Layer Zoom {str(z)}")
        tile_files = glob.glob(os.path.join(root_data_dir, str(z), '**', '*'))
        cur_zoom_dir = os.path.join(root_data_dir, str(z))
        next_zoom_dir = os.path.join(root_data_dir, str(z - 1))
        
        next_x = self.create_x_columns(cur_zoom_dir, next_zoom_dir)
        cur_x_anchors = [anchor * 2 for anchor in next_x]
        
        schema = [ArrayType(StringType()), ArrayType(StringType()), ArrayType(StringType()), ArrayType(StringType()), ArrayType(IntegerType()), ArrayType(IntegerType()), ArrayType(StringType()), ]
        
        xargs_df = spark.createDataFrame(([(int(anchor_x), next_zoom_dir, cur_zoom_dir, self.ext, self.tile_size, self.channels, self.standard) for anchor_x in cur_x_anchors]), ["anchor_x", "next_zoom_dir", "cur_zoom_dir", "ext", "tile_size", "channels", "standard"])
        xargs_df = xargs_df.coalesce(cluster_count)
        res_df = xargs_df.select(build_x_anchor(col("anchor_x"), col("next_zoom_dir"), col("cur_zoom_dir"), col("ext"), col("tile_size"), col("channels"), col("standard"),).alias("res"))
        
    def build_all_layer(
        self, 
        start_zoom:str
    ) -> None:
        """Runner function to recursively build all zoom layers from start to end.

        Args:
            start_zoom (str): Start zoom level.
        """
        if int(start_zoom) > int(self.end_zoom):
            self.build_one_layer(self.root_data_dir, int(start_zoom))
            next_zoom = int(start_zoom) - 1

            self.build_all_layer(next_zoom)
        else:
            return

# COMMAND ----------

def get_gtiff_histogram(
    fname:str, 
    nbuckets:int=4096, 
    percentiles:List[float]=[0.0005, 95.0]
) -> Tuple[List[float], List[float], float]:
    gdal.AllRegister()
    src = gdal.Open(fname)
    band = src.GetRasterBand(1)
    (lo, hi, avg, std) = band.GetStatistics(True, True)
    rawhist = band.GetHistogram(min=lo, max=hi, buckets=nbuckets)
    binEdges = np.linspace(lo, hi, nbuckets+1)
    pmf = rawhist / (np.sum(rawhist) * np.diff(binEdges[:2]))
    distribution = np.cumsum(pmf) * np.diff(binEdges[:2])
    idxs = [np.sum(distribution < p / 100.0) for p in percentiles]
    vals = [binEdges[i] for i in idxs]
    percentiles = [0] + percentiles + [100]
    vals = [lo] + vals + [hi]
    upper_limit = avg + std + std
    return (vals, percentiles, upper_limit)

@pandas_udf(ArrayType(StringType()))
def process_geotiff(
    batch_geotiff:pd.Series,
    batch_save_geotiff_dir:pd.Series,
    batch_zooms:pd.Series,
    batch_is_rgb:pd.Series,
    batch_map_region_guid:pd.Series
) -> pd.DataFrame:        
    ctx = TaskContext()
    spark_pid = ctx.partitionId()
    print(f"Spark Worker {spark_pid} processing samples: {len(batch_geotiff)}")
    spark_ret = []
    for geotiff, save_geotiff_dir, zooms, is_rgb, map_region_guid in tqdm.tqdm(zip(batch_geotiff, batch_save_geotiff_dir, batch_zooms, batch_is_rgb, batch_map_region_guid), total=len(batch_geotiff), position=0):
        if not is_rgb:
            save_scaled_geotiff_dir = f"/mnt/{map_region_guid}"
            if not os.path.exists(save_scaled_geotiff_dir):
                os.makedirs(save_scaled_geotiff_dir, exist_ok=True)

            scaled_geotiff_region = os.path.basename(os.path.split(geotiff)[0])
            scaled_geotiff_part = os.path.splitext(os.path.split(geotiff)[1])[0]
            scaled_save_geotiff_regionpart_dir = f"{scaled_geotiff_region}_{scaled_geotiff_part}"
            scaled_save_geotiff_regionpart_dir = os.path.join(save_scaled_geotiff_dir, scaled_save_geotiff_regionpart_dir)
            if not os.path.exists(scaled_save_geotiff_regionpart_dir):
                os.makedirs(scaled_save_geotiff_regionpart_dir, exist_ok=True)

            scaled_geotiff_fpath = os.path.join(scaled_save_geotiff_regionpart_dir, f"scaled_{os.path.basename(geotiff)}")

            vals, percentiles, upper_lim = get_gtiff_histogram(geotiff)
            scaling_command = f"gdal_translate -ot Byte -scale {vals[1] } {vals[2]} 0 255 -r cubic -quiet --config GDAL_PAM_ENABLED NO {geotiff} {scaled_geotiff_fpath}"
            res = os.system(scaling_command)

            output = subprocess.check_output(f"gdalinfo -json {scaled_geotiff_fpath}", shell=True)
            gdalinfo = json.loads(output)
            if "cornerCoordinates" in gdalinfo.keys():
                geotiff_region = os.path.basename(os.path.split(geotiff)[0])
                geotiff_part = os.path.splitext(os.path.split(geotiff)[1])[0]
                save_geotiff_regionpart_dir = f"{geotiff_region}_{geotiff_part}"
                save_geotiff_regionpart_dir = os.path.join(save_geotiff_dir, save_geotiff_regionpart_dir)
                if not os.path.exists(save_geotiff_regionpart_dir):
                    os.makedirs(save_geotiff_regionpart_dir, exist_ok=True)
                res = os.system(f'gdal2tiles.py --zoom={zooms} --processes={os.cpu_count()} --resampling="cubic" --quiet --webviewer="none"  {scaled_geotiff_fpath} {save_geotiff_regionpart_dir}')
                
                spark_ret.append([str(res), json.dumps(gdalinfo)])
            else:
                print(f"{scaled_geotiff_fpath} is invalid")    
                spark_ret.append([str(-1), json.dumps(gdalinfo)])
        else:
            output = subprocess.check_output(f"gdalinfo -json {geotiff}", shell=True)
            gdalinfo = json.loads(output)
            if "cornerCoordinates" in gdalinfo.keys():
                geotiff_region = os.path.basename(os.path.split(geotiff)[0])
                geotiff_part = os.path.splitext(os.path.split(geotiff)[1])[0]
                save_geotiff_regionpart_dir = f"{geotiff_region}_{geotiff_part}"
                save_geotiff_regionpart_dir = os.path.join(save_geotiff_dir, save_geotiff_regionpart_dir)
                if not os.path.exists(save_geotiff_regionpart_dir):
                    os.makedirs(save_geotiff_regionpart_dir, exist_ok=True)
                res = os.system(f'gdal2tiles.py --processes={workers} --zoom={zooms} --quiet --webviewer="none"  {geotiff} {save_geotiff_regionpart_dir}')
                spark_ret.append([str(res), json.dumps(gdalinfo)])
            else:
                spark_ret.append([str(-1), json.dumps(gdalinfo)])
    
    spark_ret = pd.Series(spark_ret)
    
    return spark_ret

# COMMAND ----------

global MANIFEST

LOCAL_DATA_DIR = '/dbfs/mnt'
# LOCAL_DATA_DIR = os.path.join(LOCAL_DATA_DIR, samp_guid)
# os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# with open(f"/dbfs/mnt/maps/{samp_guid}/datasetuploadmanifest.json", "r") as fp:
#     MANIFEST = json.load(fp)
# input_file_zoom_lvl = MANIFEST["InputFileZoomLevel"]    
# is_rgb = MANIFEST["IsRgb"]
# display_container_name = MANIFEST['DisplayContainerName']
# display_zoom_levels = MANIFEST["DisplayZoomLevels"]

# input_data_storage_account = "rebellion"
# input_data_container = "upload-00002"
# input_data_folder = "CS-5343"

# output_data_storage_account = "rebellion"
# output_data_container = "dev-databricks"
# output_data_folder_process_one = "output"
# output_data_folder_process_two = "postoutput"
# output_extraction_folder = "extractionoutput"

# input_file_zoom_lvl = 12
# is_rgb = False
# display_container_name = 'dbricks_full_pipeline_test_upload0002_4lite_lv15_v2'


input_data_storage_account = dbutils.widgets.get("input_data_storage_account")
input_data_container = dbutils.widgets.get("input_data_container")
input_data_folder = dbutils.widgets.get("input_data_folder")

output_data_storage_account = dbutils.widgets.get("output_data_storage_account")
output_data_container = dbutils.widgets.get("output_data_container")
output_data_folder_process_one = dbutils.widgets.get("output_data_folder_process_one")
output_data_folder_process_two = dbutils.widgets.get("output_data_folder_process_two")
output_extraction_folder = dbutils.widgets.get("output_extraction_folder")
display_container_name = dbutils.widgets.get("display_container_name")

input_file_zoom_lvls = dbutils.widgets.get("input_file_zoom_lvls").split(',')
input_file_zoom_lvls = np.array([int(z) for z in input_file_zoom_lvls])
input_file_zoom_lvl = int(np.amax(input_file_zoom_lvls))

is_rgb = False if dbutils.widgets.get("is_rgb") == "False" else True

test_run = True if dbutils.widgets.get("test_run") == "True" else False

print(is_rgb)

# unmount_folder(f"/mnt/{input_data_container}/{input_data_folder}")
# unmount_folder(f"/mnt/{output_data_container}/{output_data_folder_process_one}")
# unmount_folder(f"/mnt/{output_data_container}/{output_data_folder_process_two}")
# unmount_folder(f"/mnt/{output_data_container}/{output_extraction_folder}")

# mount_folder(input_data_storage_account, input_data_container, input_data_folder)
# mount_folder(input_data_storage_account, output_data_container, output_data_folder_process_one)
# mount_folder(input_data_storage_account, output_data_container, output_data_folder_process_two)
# mount_folder(input_data_storage_account, output_data_container, output_extraction_folder)

display_zoom_levels = [int(i) for i in range(5,(input_file_zoom_lvl+1))]
geotile_domain = "rgb" if is_rgb else "sar"


# COMMAND ----------

input_geotiffs = glob.glob(f'/dbfs/mnt/{input_data_container}/{input_data_folder}/*/**.tif')

if test_run:
    input_geotiffs = input_geotiffs[40:44]

output_geotiffs_dir = f"/dbfs/mnt/{output_data_container}/{output_data_folder_process_one}/{display_container_name}"

print(len(input_geotiffs))

geotiff_df = spark.createDataFrame(map(lambda input_geotiff: (input_geotiff, output_geotiffs_dir, input_file_zoom_lvl, is_rgb, display_container_name,), input_geotiffs), ["input_geotiff", "output_geotiffs_dir", "input_file_zoom_lvl", "is_rgb", "display_container_name"])


# COMMAND ----------

geotiff_df_partitioned = geotiff_df.coalesce(cluster_count)

print(geotiff_df_partitioned.show())

res_geotiff_df = geotiff_df_partitioned.select(process_geotiff(col("input_geotiff"), col("output_geotiffs_dir"), col("input_file_zoom_lvl"), col("is_rgb"), col("display_container_name"), ).alias('results')).toPandas()

print(res_geotiff_df)


# COMMAND ----------

res = res_geotiff_df.results.to_list()
return_codes = [z[0] for z in res]
geotiffinfos = [json.loads(z[1]) for z in res]


# COMMAND ----------

# results = [row.res for row in result_rows]
# return_codes = [row.ret for row in results]
# geotiffinfos = [row.geotiffinfo for row in results]
print(len(geotiffinfos))

# COMMAND ----------

src_geotiff_fpaths = [os.path.basename(geotiff_fpath) for geotiff_fpath in input_geotiffs]
geotiff_success = [{fpath:return_codes[idx]} for idx, fpath in enumerate(src_geotiff_fpaths)]
geotiff_gdalinfo = [{fpath:geotiffinfos[idx]} for idx, fpath in enumerate(src_geotiff_fpaths)]
### An output
geotiff_metadata_output_dir = f"/dbfs/mnt/{output_data_container}/{output_data_folder_process_two}/{display_container_name}"
if not os.path.exists(geotiff_metadata_output_dir):
    os.makedirs(geotiff_metadata_output_dir)
geotiff_processing_output = {"geotiff_success": geotiff_success, "geotiff_gdalinfo": geotiff_gdalinfo}
with open(os.path.join(geotiff_metadata_output_dir, "geotiffProcessingResults.json"), "w") as fp:
    json.dump(geotiff_processing_output, fp)
    

# COMMAND ----------

# Tile Remediation.

# COMMAND ----------

tmp_premediation_save_dir = f"{LOCAL_DATA_DIR}/{output_data_container}/{output_data_folder_process_one}/{display_container_name}"
os.makedirs(tmp_premediation_save_dir, exist_ok=True)

display_image_domain_directory = f"{display_container_name}/{geotile_domain}"
tile_image_domain_save_dir = f"{LOCAL_DATA_DIR}/{output_data_container}/{output_data_folder_process_two}/{display_image_domain_directory}"
os.makedirs(tile_image_domain_save_dir, exist_ok=True)

print(f"{tmp_premediation_save_dir}\n{tile_image_domain_save_dir}\n{input_file_zoom_lvl}")

mergeTiles_databricks(tmp_premediation_save_dir, tile_image_domain_save_dir, str(input_file_zoom_lvl))


# COMMAND ----------

# Layer Building

# COMMAND ----------

pm = LayerBuilder(tile_image_domain_save_dir, str(input_file_zoom_lvl), str(np.amin(display_zoom_levels)), os.cpu_count(), "png", 256, "TMS")
pm.build_all_layer(int(input_file_zoom_lvl))


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


