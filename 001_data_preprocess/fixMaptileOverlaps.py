# Databricks notebook source
# ###############################################################
# Module      : Tile remediation                                #
# Date        : 11/30/2021                                      #
# Contributors: Corey Jaskolski (corey@synthetaic.com)          # 
#               Reihaneh Rostami (reihaneh@synthetaic.com)    #
# Summary     : Takes a folder maptile directories and          #
# merges all of the tiles at same z,y,x tiel coordinate         #
# to fix overlap issues. Saves every tile (including            #
# non-overlappiing ones) into a single maptile directory        #  
# input(s)    : (1) a source directory that has this structure: #
#      "/**/{level}/**/*.png" => "{county}/{z}/{x}/{y}.png"     #
#        (2) an output directory                                #
#        (3) a zoom level (e.g. 18)                             #
# ###############################################################

import sys, os, glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from time import time
import multiprocessing
from multiprocessing import Pool,Value
from joblib import Parallel, delayed
import tqdm
import cv2

TILE_SIZE = 256
    
def print_time(s,e):
    "Print elapsed time in days:hh:mm:ss format"
    dif = e-s
    d = dif // 86400
    h = dif // 3600 % 24
    m = dif // 60 % 60
    s = dif % 60
    print("@"*50,"Elapsed time ==> Days: {} Hours: {:}   Minutes:{:}    Seconds:{:.2f}".format(d,h,m,s))
    return d,h,m,s
def main_process_uniques(row
                 ,index
                 , outputDirectory
                 , level
                 , mod = 'png4'):
    if index%100000 ==0:
        print(f"   Processing file {index}=> {row['ID']}")
    parts = row['ID'].split('-')
    x = parts[1]
    y = parts[-1]
    assert len(row['Path']) ==1,f"Error: {row['Path']} is not a unique tile" # means there is no overlapping tile=> So just copy it over to the target folder
    # img = Image.open(row['Path'][0]) 
    # imgarr = np.array(img)
    c_img = cv2.imread(row['Path'][0], cv2.IMREAD_UNCHANGED)
    c_img = cv2.cvtColor(c_img, cv2.COLOR_BGRA2RGBA)
    img = Image.fromarray(c_img)
    imgarr = np.array(img)
    try:
        a1 = np.where(imgarr[:,:,3]==0) # blank (transparent) pixels
        if a1[0].shape[0]!=img.size[0]**2:#then it is not a completely transparent
            if mod=='jpg':
                img.convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.jpg",format = 'JPEG',quality = 95)
            elif mod=='png3':
                img.convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.png",format = 'PNG')
            elif mod == 'png4':
                img.convert('RGBA').save(f"{outputDirectory}/{level}/{x}/{y}.png",format = 'PNG')
            else:
                raise ValueError(" mod must be 'jpg', 'png3', or 'png4'")
    except:
        return
        
def main_process_duplicates(row
                ,index
                , outputDirectory
                , level
                , mod = 'png4'):

    if index%100000 ==0:
        print(f"   Processing file {index}=> {row['ID']}")
    parts = row['ID'].split('-')
    x = parts[1]
    y = parts[-1]
    assert len(row['Path'])>1, f"Error: {row['Path']} is not a duplicate tile" # means there is no overlapping tile=> So just copy it over to the target folder
    #Process overlapping tiles
    tiles = []
    valued_pixels = []
    ch1_zeros = []
    ch2_zeros = []
    ch3_zeros = []
    mean = []
    std=[]
    for path in row['Path']:
        # img = Image.open(path)
        # imgarr = np.array(img)
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
        if mod=='jpg':
            df.iloc[0]['images'].convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.jpg",format = 'JPEG',quality = 95)
        elif mod == 'png3':
            df.iloc[0]['images'].convert('RGB').save(f"{outputDirectory}/{level}/{x}/{y}.png",format = 'PNG')
        elif mod == 'png4':
            df.iloc[0]['images'].convert('RGBA').save(f"{outputDirectory}/{level}/{x}/{y}.png",format = 'PNG')
        else:
            raise ValueError(" mod must be 'jpg', 'png3', or 'png4'")

            
def mergeTiles(inputDirectory,outputDirectory,level):
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
    cpu_count = int(multiprocessing.cpu_count())
    if cpu_count>2:
        cpu_count = cpu_count-2# to leave 2 cpus free
    print(f"Available cpus: {cpu_count}")

    # for unique tiles
    print(f"Running the code for {num_uniques} *unique* tiles:")
    s = time()
    Parallel(n_jobs=cpu_count)(delayed(main_process_uniques)(row,index,outputDirectory,level) for index,row in uniques.iterrows())
    e = time()
    print_time(s,e)

    # for duplicate tiles
    print(f"\n\n Running the code for {num_duplicates} *overlapping* tile:")
    s = time()
    Parallel(n_jobs=cpu_count)(delayed(main_process_duplicates)(row, index, outputDirectory, level) for index,row in duplicates.iterrows())
    e = time()
    print_time(s,e)
    
    print("\n\n\n","+_"*15)
    print(f"Total files (initial): {len(pngfiles)}")
    print(f"Total files (aggregated): {len_aggdf}")
    print(f"Created folder under level {df['z'].iloc[0]}: {len(xs)}")
    print(f"Output directory: {outputDirectory}/{level}")

def mergeTiles_databricks(inputDirectory,outputDirectory,level):
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
    cpu_count = int(multiprocessing.cpu_count())
    if cpu_count>2:
        cpu_count = cpu_count-2# to leave 2 cpus free
    print(f"Available cpus: {cpu_count}")

    # for unique tiles
    print(f"Running the code for {num_uniques} *unique* tiles:")
    s = time()
    
    unique_rdd = sc.parallelize([(row, index, outputDirectory, level) for index, row in uniques.iterrows()])
    unique_df = unique_rdd.toDF()
    udf_main_process_uniques = udf(main_process_uniques)
    res = unique_df.select(udf_main_process_uniques(col("_1"), col("_2"), col("_3"), col("_4"))).alias("result").collect()

    e = time()
    print_time(s,e)
    
    # for duplicate tiles
    print(f"\n\n Running the code for {num_duplicates} *overlapping* tile:")
    s = time()
    
    duplicates_rdd = sc.parallelize([(row, index, outputDirectory, level) for index, row in duplicates.iterrows()])    
    duplicates_df = duplicates_rdd.toDF()
    udf_main_process_duplicates = udf(main_process_duplicates)
    res = duplicates_df.select(udf_main_process_duplicates(col("_1"), col("_2"), col("_3"), col("_4"))).alias("result").collect()
    
    e = time()
    print_time(s,e)
    
    
    print("\n\n\n","+_"*15)
    print(f"Total files (initial): {len(pngfiles)}")
    print(f"Total files (aggregated): {len_aggdf}")
    print(f"Created folder under level {df['z'].iloc[0]}: {len(xs)}")
    print(f"Output directory: {outputDirectory}/{level}")

    
# Sample call:
# python3 fixMaptileOverlaps_mp_df.py /data_32TB/CA_out_split /data_32TB/CANaipNew 18 
def main(argv):
    if len(argv) < 4:
        print("Usage: fixMaptileOverlaps.py inputDirectory outputDirectory zoom")
        return 1
    inputDirectory=str(argv[1])
    outputDirectory=str(argv[2])
    level=str(argv[3])
    mergeTiles(inputDirectory,outputDirectory,level)
    return 0
