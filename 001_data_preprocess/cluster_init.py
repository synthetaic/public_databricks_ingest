# Databricks notebook source


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create Init Script for Cluster Startup
# MAGIC 
# MAGIC It is important to use command 3 script.  Creates the inside the databricks file system.  Then you can define this init script in the config of the startup of the cluster.  Using this init script during the startup of the cluster will install the components needed on the workers. 

# COMMAND ----------

dbutils.fs.put("/databricks/scripts/requirements.txt", """
pip>=18.1
wheel
setuptools
python-dotenv==0.19.2
tqdm==4.62.3
azure-storage-blob==12.9.0
torch==1.8.2+cu111
torchvision==0.9.2+cu111
scikit-learn==0.22.1
scipy==1.7.3
faiss-cpu==1.7.1
numpy==1.22.0
matplotlib==3.5.1
pandas==1.3.5
pyarrow==6.0.*
ffmpeg-python==0.2.0
ffprobe-python==1.0.3
opencv-python==4.5.5.62
Pillow==9.0.0
gdal2tiles==0.1.9
pygeotile==1.0.6
https://pretrainedmodels001.blob.core.windows.net/model-libraries/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
https://pretrainedmodels001.blob.core.windows.net/model-libraries/big_transfer-1.0.0-py3-none-any.whl
https://pretrainedmodels001.blob.core.windows.net/model-libraries/alae-1.0.2-py3-none-any.whl
-f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
""", True)

# COMMAND ----------

dbutils.fs.put("/databricks/scripts/cluster_init.sh","""
#!/bin/bash
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get -y upgrade
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install ffmpeg libsm6 libxext6
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install libpcap-dev libpq-dev 
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install gdal-bin gdal-data libgdal-perl libgdal-dev python3-gdal
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
sudo apt-get -y autoremove
/databricks/python3/bin/pip install -U pip
git clone https://synthetaic-org:l3ljr3na44undceru6pok6inmen3ss3olx34xisvive73q4k4taa@dev.azure.com/synthetaic-org/RAIC-V1/_git/raic-v1-ai && cd raic-v1-ai && /databricks/python3/bin/pip install -e . && /databricks/python3/bin/pip install -r raicV1/preprocess/requirements.txt
/databricks/python3/bin/pip install gdal==3.0.4 --global-option=build_ext --global-option='-I/usr/include/gdal/'
""", True)

# COMMAND ----------

dbutils.fs.put("/databricks/scripts/install_pip_libraries.sh","""
#!/bin/bash
/databricks/python3/bin/pip install -r /dbfs/databricks/scripts/requirements.txt
""", True)

# COMMAND ----------


