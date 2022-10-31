# Databricks notebook source
from typing import Dict, List, Tuple
import cv2
import numpy as np
import os
from multiprocessing.pool import ThreadPool as Pool
import glob
import tqdm as tqdm
import math

import pygeotile
from pygeotile.point import Point
from pygeotile.tile import Tile

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

def build_x_anchor(
    x:int, 
    next_zoom_dir:str, 
    cur_zoom_dir:str, 
    ext:str, 
    tile_size:int, 
    channels:int, 
    standard:str
) -> None:
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
        min_cur_y_anchors = min(cur_y_anchors) - 1
        max_cur_y_anchors = max(cur_y_anchors) + 1
        actual_cur_y_anchors = list(range(min_cur_y_anchors, (max_cur_y_anchors+1)))
        nxt_anchor_y = [math.floor(tmp / 2) for tmp in actual_cur_y_anchors]
        nxt_anchor_y = list(set(nxt_anchor_y))
        valid_nxt_anchor_y = [a_y for a_y in nxt_anchor_y if valid_xy_anchor(cur_zoom_dir, x, a_y, ext)]

        s = [merge_anchor_x(cur_zoom_dir, next_zoom_dir, x, ext, tile_size, ancy, channels, standard) for ancy in valid_nxt_anchor_y]


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
        edges = [(min(next_x) + 1), (max(next_x) + 1)]
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
      
        xargs_rdd = sc.parallelize([(anchor_x, next_zoom_dir, cur_zoom_dir, self.ext, self.tile_size, self.channels, self.standard,) for anchor_x in cur_x_anchors])
        xargs_df = xargs_rdd.toDF()
        udf_build_x_anchor = udf(build_x_anchor)
        result_rows = xargs_df.select(udf_build_x_anchor(col("_1"), col("_2"), col("_3"), col("_4"), col("_5"), col("_6"), col("_7"))).alias("result").collect()
      
#         xargs = [(anchor_x, next_zoom_dir, cur_zoom_dir, self.ext, self.tile_size, self.channels, self.standard,) for anchor_x in cur_x_anchors]
#         workers = min(os.cpu_count(), len(xargs))
#         with Pool(processes=workers) as p:
#             r = p.map(work_merge_anchor_x, (xargs))

    def build_all_layer(
        self, 
        start_zoom:str
    ) -> None:
        """Runner function to recursively build all zoom layers from start to end.

        Args:
            start_zoom (str): Start zoom level.
        """
        if int(start_zoom) > int(self.end_zoom):
            self.build_one_layer(self.root_data_dir, start_zoom)
            next_zoom = int(start_zoom) - 1

            self.build_all_layer(next_zoom)
        else:
            return
