# Databricks notebook source
from typing import Dict, List, Tuple

# def mount_folder(
#   storageAccount:str, 
#   containerName:str, 
#   folderName:str, 
#   appRegistrationId:str="5fc88396-5f37-4469-be79-1ce0821a1e2a", 
#   directoryId:str="628574bc-9c62-4216-afdd-7278cbd0123a"
# ) -> None:

#   clientSecret = dbutils.secrets.get(scope = "azure-key-vault", key = "dev-raic-adlsgen2-001")
#   storageSource = f"abfss://{containerName}@{storageAccount}.dfs.core.windows.net/{folderName}"
#   fullMountLocationName = f"/mnt/{containerName}/{folderName}"

#   configs = {
#           "fs.azure.account.auth.type"              : "OAuth"
#         , "fs.azure.account.oauth.provider.type"    : "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
#         , "fs.azure.account.oauth2.client.id"       : appRegistrationId
#         , "fs.azure.account.oauth2.client.secret"   : clientSecret
#         , "fs.azure.account.oauth2.client.endpoint" : f"https://login.microsoftonline.com/{directoryId}/oauth2/token"
#         }
  
#   dbutils.fs.mount(
#     source = storageSource,
#     mount_point = fullMountLocationName,
#     extra_configs = configs
#   )
  
def mount_folder(
  storageAccount:str, 
  containerName:str, 
  folderName:str, 
  sas_key:str="?sv=2020-08-04&ss=bfqt&srt=co&sp=rwlacupitf&se=2028-02-09T05:38:27Z&st=2022-02-08T21:38:27Z&spr=https&sig=LqDunHmud2LsmoU5LmpswsE2jWrOGVn7Pzm6Ehl32OM%3D"
#   appRegistrationId:str="5fc88396-5f37-4469-be79-1ce0821a1e2a", 
#   directoryId:str="628574bc-9c62-4216-afdd-7278cbd0123a"
) -> None:
  config = {"fs.azure.sas." + containerName + "." + storageAccount + ".blob.core.windows.net" : sas_key}
#   clientSecret = dbutils.secrets.get(scope = "azure-key-vault", key = "dev-raic-adlsgen2-001")
  storageSource = f"wasbs://{containerName}@{storageAccount}.blob.core.windows.net/{folderName}"
  fullMountLocationName = f"/mnt/{containerName}/{folderName}"

#   configs = {
#           "fs.azure.account.auth.type"              : "OAuth"
#         , "fs.azure.account.oauth.provider.type"    : "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
#         , "fs.azure.account.oauth2.client.id"       : appRegistrationId
#         , "fs.azure.account.oauth2.client.secret"   : clientSecret
#         , "fs.azure.account.oauth2.client.endpoint" : f"https://login.microsoftonline.com/{directoryId}/oauth2/token"
#         }
  
  dbutils.fs.mount(
    source = storageSource,
    mount_point = fullMountLocationName,
    extra_configs = config
  )
  

def list_mounts() -> None:
  display(dbutils.fs.mounts().toDF) 


def unmount_folder(
  dest_folder_path:str
) -> None:
  dbutils.fs.unmount(dest_folder_path)

  

# COMMAND ----------


