import io
import zipfile as zp
import posixpath as psx
import numpy as np
import pandas as pd
import requests as req
from typing import Tuple, Dict

from Config import constants as cnst
import Utils.visual_angle_utils as vis_utils
from DataSetLoaders.BaseDataSetLoader import BaseDataSetLoader
from Config.ScreenMonitor import ScreenMonitor


class IRFDataSetLoader(BaseDataSetLoader):
    """
    Loads the dataset from a replication study of the article:
    Using machine learning to detect events in eye-tracking data. Zemblys et al. (2018).
    See also about the repro study: https://github.com/r-zemblys/irf/blob/master/doc/IRF_replication_report.pdf

    Note: binocular data was recorded but only one pair of (x, y) coordinates is provided.

    This loader is based on a previous implementation, see article:
    Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. Behav Res 55, 1653–1714 (2023)
    See their implementation: https://github.com/r-zemblys/EM-event-detection-evaluation/blob/main/misc/data_parsers/humanFixationClassification.py
    """

    _URL = r'https://github.com/r-zemblys/irf/archive/refs/heads/master.zip'
    _ARTICLES = [
        "Zemblys, Raimondas and Niehorster, Diederick C and Komogortsev, Oleg and Holmqvist, Kenneth. Using machine " +
        "learning to detect events in eye-tracking data. Behavior Research Methods, 50(1), 160–181 (2018)."
    ]
    _NAME: str = "IRF"

    # Values used in the apparatus of the experiment.
    # see https://github.com/r-zemblys/irf/blob/master/etdata/lookAtPoint_EL/db_config.json
    __RATER_NAME = "RZ"
    __STIMULUS_VAL = "moving_dot"  # all subjects were shown the same 13-point moving dot stimulus
    __VIEWER_DISTANCE_CM_VAL = 56.5
    __SCREEN_MONITOR = ScreenMonitor(width=37.5, height=30.2, resolution=(1280, 1024), refresh_rate=60)

    @classmethod
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))

        # Get ET Data:
        prefix = 'irf-master/etdata/lookAtPoint_EL'
        gaze_file_names = [f for f in zip_file.namelist() if (f.startswith(psx.join(prefix, "lookAtPoint_EL_"))
                                                              and f.endswith('.npy'))]
        gaze_dfs = []
        for f in gaze_file_names:
            file = zip_file.open(f)
            gaze_data = pd.DataFrame(np.load(file))

            # convert gaze events from int to GazeEventTypeEnum
            gaze_data['evt'] = gaze_data['evt'].apply(lambda x: BaseDataSetLoader._parse_gaze_event(x, safe=True))

            # extract subject id:
            _, file_name, _ = IRFDataSetLoader._extract_filename_and_extension(f)
            subject_id = file_name.split('_')[-1]  # format: "lookAtPoint_EL_S<subject_num>"
            gaze_data[cnst.SUBJECT_ID] = subject_id
            gaze_dfs.append(gaze_data)
        merged_df = pd.concat(gaze_dfs, ignore_index=True, axis=0)

        # add meta data columns:
        merged_df[cnst.STIMULUS] = cls.__STIMULUS_VAL
        merged_df[cls._VIEWER_DISTANCE_CM_STR] = cls.__VIEWER_DISTANCE_CM_VAL
        merged_df[cls._PIXEL_SIZE_CM_STR] = cls.__SCREEN_MONITOR.pixel_size
        return merged_df

    @classmethod
    def _clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # replace invalid samples with NaN:
        idxs_to_replace = df[~df['status']].index
        df.loc[idxs_to_replace, cnst.X] = np.nan
        df.loc[idxs_to_replace, cnst.Y] = np.nan

        # rename columns: replace `t` with `milliseconds`, `evt` with `rater_name`
        # also, drop the `status` column that indicates whether the data is valid.
        df.rename(columns={"t": cnst.T, "evt": cls.__RATER_NAME, "x": cnst.X, "y": cnst.Y},
                  inplace=True)
        df.drop(columns=["status"], inplace=True)

        # convert to milliseconds:
        df[cnst.T] = df[cnst.T] * cnst.MILLISECONDS_PER_SECOND

        # convert x-y coordinates to pixels (use apparatus values):
        x, y = cls.__convert_coordinates(x=df[cnst.X], y=df[cnst.Y])
        df[cnst.X] = x
        df[cnst.Y] = y

        # add a column for trial number:
        # trials are instances that share the same subject id & stimulus.
        trial_counter = 1
        df[cnst.TRIAL] = np.nan
        for _, trial_df in df.groupby([cnst.SUBJECT_ID]):
            df.loc[trial_df.index, cnst.TRIAL] = trial_counter
            trial_counter += 1
        df[cnst.TRIAL] = df[cnst.TRIAL].astype(int)
        return df

    @classmethod
    def __convert_coordinates(cls, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        pixel_width = cls.__SCREEN_MONITOR.width / cls.__SCREEN_MONITOR.resolution[0]  # in cm
        x = x.apply(lambda deg: vis_utils.visual_angle_to_pixels(deg=deg, d=cls.__VIEWER_DISTANCE_CM_VAL,
                                                                 pixel_size=pixel_width, keep_sign=True))
        x += cls.__SCREEN_MONITOR.resolution[0] // 2  # move x=0 coordinate to the top of the screen

        pixel_height = cls.__SCREEN_MONITOR.height / cls.__SCREEN_MONITOR.resolution[1]  # in cm
        y = y.apply(lambda deg: vis_utils.visual_angle_to_pixels(deg=deg, d=cls.__VIEWER_DISTANCE_CM_VAL,
                                                                 pixel_size=pixel_height, keep_sign=True))
        y += cls.__SCREEN_MONITOR.resolution[1] // 2  # move y=0 coordinate to the top of the screen
        return x, y
