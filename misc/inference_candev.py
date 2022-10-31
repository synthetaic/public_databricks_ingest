# Databricks notebook source


# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000000")

import uuid
from typing import *
import glob
import tqdm
import os 
import io
from io import *
import pickle
import time

import numpy as np
import faiss
import pandas as pd
import torch
import PIL
from PIL import Image
import cv2
import ray
from dotenv import load_dotenv

import pyspark
from pyspark import TaskContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import *

import math

from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision
from pygeotile.tile import Tile
from platform import python_version

from raicV1.models import Mdl
from raicV1.preprocess.coreai_utils import (BallTreeIndexLatentFeatures,
                          DistributedBaseExtractLatentFeatures,
                          FaissIndexLatentFeatures, get_mean_vector,
                          whiten_matrix_numpy)

from raicV1.preprocess.geotile_utils import get_zxy_latlong_from_fpath

cuda = True
print(python_version())

# COMMAND ----------

use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# COMMAND ----------

# model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
# bc_model_state = sc.broadcast(model.state_dict())


# COMMAND ----------

def get_model_for_eval():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
    model.load_state_dict(bc_model_state.value)
    model.eval()
    return model

# COMMAND ----------

# Raic-V1 feature extraction process phase

# COMMAND ----------

# Pytorch dataset class.
class TensorPatchesDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_imgs:List[str], 
        data_patch_coords:np.ndarray,
        transform:torchvision.transforms.Compose=None,
        kernel_size:int=256,
        loaded_bytes:bool=False,
    ) -> None:
        """Generic inherited pytorch dataset class to process RAM tensor-form of individual patches.

        Args:
            data_imgs (List[str]): List of file paths or bytearrays.
            data_patch_coords (np.ndarray): Filepath for the frame image.
            transform (torchvision.transforms.Compose, optional): Pytorch tranformations to do on the tensor patches. IE: Normalizing. Defaults to None.
            kernel_size (int): kernel size of for patching.
        """
        self.data_imgs = data_imgs
        self.data_patch_coords = data_patch_coords
        self.kernel_size = kernel_size
        self.transform = transform
        self.loaded_bytes = loaded_bytes

    def __len__(
        self
    ) -> int:
        """Return the length of the dataset, which is the number of total patches in the dataset.

        Returns:
            int: Length of dataset.
        """
        return len(self.data_imgs)
    
    def __getitem__(
        self, 
        index:int
    ) -> Tuple[torch.tensor, np.ndarray]:
        """Returns patch tensor and its corresponding coords.

        Args:
            index (int): Index corresponding to pair.

        Returns:
            Tuple[torch.tensor, np.ndarray]: Patch image, corresponding patch coord.
        """
        patch = self.data_imgs[index]
        if not self.loaded_bytes:
            patch = Image.open(patch)
        else:
            patch = np.asarray(bytearray(patch), dtype=np.uint8).reshape(self.kernel_size,self.kernel_size, 4)
            patch = Image.fromarray(patch)

        patch = patch.convert('RGB')

        cur_patch_coords = self.data_patch_coords[index]

        if self.transform:
            patch = self.transform(patch)

        return patch, cur_patch_coords


# COMMAND ----------

@pandas_udf(ArrayType(ArrayType(FloatType())))
def extract_features(
    data:pd.Series,
    z:pd.Series,
    x:pd.Series,
    y:pd.Series,
    load_images_dataframe:pd.Series,
    raic_method:pd.Series,
    raic_zoom:pd.Series,
    batches_data_save_dir:pd.Series=None,
) -> pd.Series:
    total_time = time.time()
    ctx = TaskContext()
    spark_pid = ctx.partitionId()
    print(f"Working on partition: {spark_pid} on {len(data)} data inputs with {os.cpu_count() - 1} dataloader workers.")
    if batches_data_save_dir is not None:
        batches_data_save_dir = batches_data_save_dir[0]
    
    m_time = time.time()
    
#     model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])
    mdl = Mdl()
    model_string = mdl.load_mdl(raic_method, int(raic_zoom))
    model = mdl.mdl[model_string]["model"]
    transform = mdl.mdl[model_string]["txf"]

    model.eval()
    print(f"{spark_pid}:\tDirect model loading time: {time.time() - m_time}")

    model.to(device)
 
    batch_size = 142 if raic_method == "B" else 1536 # For 16gb GPU and 11gb GPU

    coords_npy = np.column_stack((z,x,y))
                             
    tile_dataset = TensorPatchesDataset(data, coords_npy, transform, 256, loaded_bytes=load_images_dataframe[0])
    loader = torch.utils.data.DataLoader(tile_dataset, batch_size=batch_size, shuffle=False, num_workers=(os.cpu_count() - 1), pin_memory=True, drop_last=False)

    all_latents = []
    all_metadata = []
    one_data = []
    
    extract_time = time.time()
    
    with torch.no_grad():
        for (input_images, input_metadata) in tqdm.tqdm(loader, position=0):
            images = input_images.to(device)
            input_metadata = input_metadata.cpu().numpy()
            output_features = model(images).cpu().numpy()

            if len(output_features.shape) == 1:
                output_features = output_features[None, :]
            for meta, lat in zip(list(input_metadata), list(output_features)):
                if batches_data_save_dir is not None:
                    all_latents.append(lat)
                    all_metadata.append(meta)
                    one_data.append([[0.0], [0.0]])
                else:
                    one_data.append([meta, lat])

    print(f"{spark_pid}:\tFeature extraction time: {time.time() - extract_time}")
    
    # Save to disk to save on RAM.
    if batches_data_save_dir is not None:
        latents_save = os.path.join(batches_data_save_dir, 'latents', f'pid_batch_{str(spark_pid).zfill(9)}.npy')
        np.save(latents_save, np.array(all_latents))
        metadata_save = os.path.join(batches_data_save_dir, 'coords', f'pid_batch_{str(spark_pid).zfill(9)}.npy')
        np.save(metadata_save, np.array(all_metadata))

    spark_ret = pd.Series(one_data)
    
    print(f"{spark_pid}:\tFinished work total time: {time.time() - total_time}")
    
    return spark_ret
    

# COMMAND ----------

# Hard code environment varibles for local development.

# output_data_storage_account = "rebellion"
# output_data_container = "dev-databricks"
# output_data_folder_process_two = "postoutput"
# output_extraction_folder = "extractionoutput"
# input_file_zoom_lvls = ["17","19"]
# display_container_name = "dbricks_test_guid3"
# is_rgb = False
# raic_methods = ["C"]

# Use .env file for azure devop pipeline build and release integration.

# load_dotenv()

# output_data_storage_account = os.getenv("output_data_storage_account")
# output_data_container = os.getenv("output_data_container")
# output_data_folder_process_two = os.getenv("output_data_folder_process_two")
# output_extraction_folder = os.getenv("output_extraction_folder")
# display_container_name = os.getenv("display_container_name")
# input_file_zoom_lvl = os.getenv("input_file_zoom_lvl")
# is_rgb = os.getenv("is_rgb")

# Use databricks widgets for jobs api json parameters.

output_data_storage_account = dbutils.widgets.get("output_data_storage_account")
output_data_container = dbutils.widgets.get("output_data_container")
output_data_folder_process_two = dbutils.widgets.get("output_data_folder_process_two")
output_extraction_folder = dbutils.widgets.get("output_extraction_folder")
display_container_name = dbutils.widgets.get("display_container_name")
is_rgb = False if dbutils.widgets.get("is_rgb") == "False" else True

input_file_zoom_lvls = dbutils.widgets.get("input_file_zoom_lvls").split(',')
input_file_zoom_lvls = [int(z) for z in input_file_zoom_lvls]
raic_methods = ["C"]

# unmount_folder(f"/mnt/{output_data_container}/{output_extraction_folder}")
# mount_folder(input_data_storage_account, output_data_container, output_extraction_folder)

LOCAL_DATA_DIR = '/dbfs/mnt'
# LOCAL_DATA_DIR = os.path.join(LOCAL_DATA_DIR, samp_guid)
# os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# Download the manifest from azure blob instead of mounting folder.
# with open(f"/dbfs/mnt/maps/{samp_guid}/datasetuploadmanifest.json", "r") as fp:
#     MANIFEST = json.load(fp)
# input_file_zoom_lvl = MANIFEST["InputFileZoomLevel"]    
# is_rgb = MANIFEST["IsRgb"]
# display_container_name = MANIFEST['DisplayContainerName']
# display_zoom_levels = MANIFEST["DisplayZoomLevels"]

geotile_domain = "rgb" if is_rgb else "sar"

for cur_raic_method in raic_methods:
    for input_file_zoom_lvl in input_file_zoom_lvls:
        batches_data_save_dir = f"/dbfs/mnt/tmp_extract/{display_container_name}/{geotile_domain}/{cur_raic_method}/{input_file_zoom_lvl}/"
        os.makedirs(os.path.join(batches_data_save_dir, 'latents'), exist_ok=True)
        os.makedirs(os.path.join(batches_data_save_dir, 'coords'), exist_ok=True)

        dataset_input_fpaths = glob.glob(f'/dbfs/mnt/{output_data_container}/{output_data_folder_process_two}/{display_container_name}/{geotile_domain}/{input_file_zoom_lvl}/*/**')

        print(dataset_input_fpaths[0:3])
        print(f"Total Dataset filepaths for zoom {input_file_zoom_lvl}: {len(dataset_input_fpaths)}")

        load_images_dataframe = False
        # Load images directly to spark dataframes.
        if load_images_dataframe:
            dbfs_input_fpaths = [z.split('dbfs') for z in dataset_input_fpaths]
            dbfs_input_fpaths = [f"dbfs:{z[1]}" for z in dbfs_input_fpaths]
            img_df = spark.read.format("image").option("dropInvalid", True).load(dbfs_input_fpaths)
            img_origins = img_df.select('image.origin').toPandas().origin.to_list()
            dataset_input_fpaths = [z.split('dbfs:') for z in img_origins]
            dataset_input_fpaths = [f"/dbfs{z}" for z in dataset_input_fpaths]
            img_df = img_df.select('image.data')
        else:
            img_df = spark.createDataFrame(map(lambda data: (data,), dataset_input_fpaths), ["data"])

        geo_data = [get_zxy_latlong_from_fpath(z) for z in tqdm.tqdm(dataset_input_fpaths)]
        z = [inp[0] for inp in geo_data]
        x = [inp[1] for inp in geo_data]
        y = [inp[2] for inp in geo_data]
        geo_coords = [(zz, xx, yy) for zz,xx,yy in zip(z,x,y)] 

        write_disk = True
        # Save partitions to Disk.
        if write_disk:
            geo_coords_df = spark.createDataFrame(map(lambda coord: (coord[0],coord[1],coord[2],load_images_dataframe, cur_raic_method, input_file_zoom_lvl, batches_data_save_dir,), geo_coords), ["z", "x", "y", "load_images_dataframe", "raic_method", "raic_zoom", "batches_data_save_dir"])
        else:
        # Return partitions in RAM.
            geo_coords_df = spark.createDataFrame(map(lambda coord: (coord[0],coord[1],coord[2],load_images_dataframe, cur_raic_method, input_file_zoom_lvl,), geo_coords), ["z", "x", "y", "raic_method", "raic_zoom", "load_images_dataframe"])
        
        w=Window.orderBy(lit(1))
        geo_coords_df2 = geo_coords_df.withColumn("rn",row_number().over(w)-1)
        img_df2 = img_df.withColumn("rn",row_number().over(w)-1)

        # Find RAM-limited max partition size for loaded images.
        total_size = len(dataset_input_fpaths)
        # max_partition_size = 8 # 56 # 896
        max_partition_size = 2 if math.ceil(total_size / 53760) < 2 else math.ceil(total_size / 53760)
        # max_partition_size = math.ceil(total_size / spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch"))

        dataset_df = img_df2.join(geo_coords_df2, ["rn"]).drop("rn").coalesce(max_partition_size)
        
        if write_disk:
            predictions_df = dataset_df.select(extract_features(col('data'), col('z'), col('x'), col('y'), col('load_images_dataframe'), col('raic_method'), col('raic_zoom'), col('batches_data_save_dir'),).alias("results"))
        else:
            predictions_df = dataset_df.select(extract_features(col('data'), col('z'), col('x'), col('y'), col('load_images_dataframe'), col('raic_method'), col('raic_zoom'),).alias("results"))

        predictions_df = predictions_df.toPandas()

        data_save_dir = f"/dbfs/mnt/{output_data_container}/{output_extraction_folder}/{display_container_name}/{geotile_domain}/{cur_raic_method}/{input_file_zoom_lvl}/"
        os.makedirs(data_save_dir, exist_ok=True)

        if write_disk:
            print(predictions_df)
            res = predictions_df.results.to_numpy()
            coords_meta = np.array([z[0] for z in res]).astype(np.uint8)
            latents = np.array([z[1] for z in res])

            print(f"coords: {coords_meta.shape}\tlatents: {latents.shape}")

            np.save(os.path.join(data_save_dir, 'mat.npy'), latents)
            np.save(os.path.join(data_save_dir, 'coords.npy'), coords_meta)

            print(coords_meta[0:3])
        else:
            latents_fpaths = sorted(glob.glob(os.path.join(batches_data_save_dir, "latents", "*")))
            all_lats = [np.load(z, allow_picklef=True) for z in latents_fpaths]
            all_lats = np.concatenate(all_lats)
            np.save(os.path.join(data_save_dir, 'mat.npy'), all_lats)

            coords_fpaths = sorted(glob.glob(os.path.join(batches_data_save_dir, "coords", "*")))
            all_coords = [np.load(z, allow_pickle=True) for z in coords_fpaths]
            all_coords = np.concatenate(all_coords)
            np.save(os.path.join(data_save_dir, 'coords.npy'), all_coords)

        faiser = FaissIndexLatentFeatures()
        faiss_save = os.path.join(data_save_dir, 'faissindex.pickle')
        latents = np.load(os.path.join(data_save_dir, 'mat.npy'), allow_pickle=True)
        coords = np.load(os.path.join(data_save_dir, 'coords.npy'), allow_pickle=True)
        index = faiser(faiss_save, latents, coords)



# COMMAND ----------

# all_lats = np.load(f'/dbfs/mnt/dev-databricks/extractionoutput/{display_container_name}/{geotile_domain}/{input_file_zoom_lvl}/mat.npy', allow_pickle=True)
# print(all_lats.shape)

# all_lats = np.load(f'/dbfs/mnt/dev-databricks/extractionoutput/{display_container_name}/{geotile_domain}/{input_file_zoom_lvl}/coords.npy', allow_pickle=True)
# print(all_lats.shape)

# COMMAND ----------



# COMMAND ----------

# Basic Feature Extraction

# COMMAND ----------

# class PDDataset(Dataset):
#     def __init__(self, fpaths, transform=None):
#         self.fpaths = fpaths
#         self.transform = transform
#     def __len__(self):
#         return len(self.fpaths)
#     def __getitem__(self, index):
#         image = Image.open(self.fpaths[index]).convert("RGB")
#         return self.transform(image)
 

# COMMAND ----------

# @pandas_udf(ArrayType(FloatType()))
# def base_feature_extract(paths: pd.Series) -> pd.Series:
#     transform = transforms.Compose([
#     #      transforms.Resize((256,256)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])
    
#     images = PDDataset(paths, transform=transform)
#     loader = torch.utils.data.DataLoader(images, batch_size=1500, num_workers= 4)
#     model = get_model_for_eval()
#     model.to(device)
#     all_predictions = []
#     with torch.no_grad():
#         for batch in loader:
#             predictions = list(model(batch.to(device)).cpu().numpy())
#             for prediction in predictions:
#                 all_predictions.append(prediction)
#     return pd.Series(all_predictions)

# COMMAND ----------

# dataset_input_fpaths = glob.glob('/dbfs/mnt/dev-databricks/postoutput/dbricks_test_guid3/sar/17/*/**')[512:768]
# dataset_df = spark.createDataFrame(
#     map(lambda path: (path,), dataset_input_fpaths), ["path"]
# ).repartition(6)
# print(len(dataset_input_fpaths))

# predictions_df = dataset_df.select(col('path'), base_feature_extract(col('path')).alias("prediction"))

# output_file_path = "/dbfs/mnt/dev-databricks/postoutput/output.parquet"
# predictions_df = predictions_df.toPandas()
# predictions_df.to_parquet(output_file_path)

# COMMAND ----------

# df = pd.read_parquet(output_file_path)
# print(df)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

!ls -la /dbfs/mnt/dev-databricks/

# COMMAND ----------


