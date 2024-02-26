import io
import requests as req
import zipfile as zp
import posixpath as psx
import pandas as pd
import numpy as np
import arff

from Config import constants as cnst
from Config.ScreenMonitor import ScreenMonitor
import Utils.io_utils as ioutils
from DataSetLoaders.BaseDataSetLoader import BaseDataSetLoader


class GazeComDataSetLoader(BaseDataSetLoader):
    """
    Loads a labelled subset of the dataset presented in the article:
    Michael Dorr, Thomas Martinetz, Karl Gegenfurtner, and Erhardt Barth. Variability of eye movements when viewing
    dynamic natural scenes. Journal of Vision, 10(10):1-17, 2010.

    Labels are from the article:
    Agtzidis, I., Startsev, M., & Dorr, M. (2016a). In the pursuit of (ground) truth: A hand-labelling tool for eye
    movements recorded during dynamic scene viewing. In 2016 IEEE second workshop on eye tracking and visualization
    (ETVIS) (pp. 65–68).

    Note 1: This dataset is extremely large and may take a long time to download. It is recommended to use the
    load_from_disk method to load the dataset from a local directory.
    Note 2: This is only a subset of the full GazeCom Dataset, containing hand-labelled samples. The full dataset with
    documentation can be found in https://www.inb.uni-luebeck.de/index.php?id=515.
    Note 3: binocular data was recorded but only one pair of (x, y) coordinates is provided.

    This loader is based on a previous implementation, see article:
    Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. Behav Res 55, 1653–1714 (2023)
    See their implementation: https://github.com/r-zemblys/EM-event-detection-evaluation/blob/main/misc/data_parsers/humanFixationClassification.py
    """

    _URL = r'https://gin.g-node.org/ioannis.agtzidis/gazecom_annotations/archive/master.zip'
    _ARTICLES = [
        "Agtzidis, I., Startsev, M., & Dorr, M. (2016a). In the pursuit of (ground) truth: A hand-labelling tool for " +
        "eye movements recorded during dynamic scene viewing. In 2016 IEEE second workshop on eye tracking and" +
        "visualization (ETVIS) (pp. 65–68).",
        "Michael Dorr, Thomas Martinetz, Karl Gegenfurtner, and Erhardt Barth. Variability of eye movements when " +
        "viewing dynamic natural scenes. Journal of Vision, 10(10):1-17, 2010.",
        "Startsev, M., Agtzidis, I., & Dorr, M. (2016). Smooth pursuit. http://michaeldorr.de/smoothpursuit/"
    ]

    # Values used in the apparatus of the experiment.
    __FILE_NAME = "gazecom_annotations-master.zip"
    __HANDLABELLER = "HL"
    __VIEWER_DISTANCE_CM_VAL = 56.5
    __SCREEN_MONITOR = ScreenMonitor(width=40, height=22.5, resolution=(1280, 720), refresh_rate=30)
    __EVENT_MAPPING = {
        0: cnst.EVENTS.UNDEFINED,
        1: cnst.EVENTS.FIXATION,
        2: cnst.EVENTS.SACCADE,
        3: cnst.EVENTS.SMOOTH_PURSUIT,
        4: cnst.EVENTS.UNDEFINED  # noise
    }
    __COLUMNS_MAPPING = {"time": cnst.MILLISECONDS, "x": cnst.X, "y": cnst.Y,
                         "handlabeller1": f"{__HANDLABELLER}1", "handlabeller2": f"{__HANDLABELLER}2",
                         "handlabeller_final": f"{__HANDLABELLER}_FINAL"}

    @classmethod
    def load_from_disk(cls, root: str = None) -> pd.DataFrame:
        """
        Loads the dataset from a zip file stored in a local directory.
        :param root: path to the directory containing the zip file.
        :return: DataFrame with the annotated gaze data.
        """
        if not root or not psx.isdir(root):
            raise NotADirectoryError(f"Invalid directory: {root}")
        zip_file = psx.join(root, cls.__FILE_NAME)
        if not psx.isfile(zip_file):
            raise FileNotFoundError(f"File not found: {zip_file}")
        with zp.ZipFile(zip_file, 'r') as zip_ref:
            df = cls.__read_zipfile(zf=zip_ref)
            df = cls._clean_data(df)
            ordered_columns = sorted(df.columns, key=lambda col: cls.column_order().get(col, 10))
            df = df[ordered_columns]  # reorder columns
            return df

    @classmethod
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))
        return cls.__read_zipfile(zf=zip_file)

    @classmethod
    def _clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # replace invalid samples with NaN:
        trackloss = np.all(df[["x", "y"]] == 0, axis=1)
        low_confidence = df["confidence"] < 0.5
        invalid_idxs = np.where(trackloss | low_confidence)[0]
        df.iloc[invalid_idxs, df.columns.get_indexer(["x", "y"])] = np.nan

        # rename columns & drop confidence column:
        df['time'] = df['time'] / cnst.MICROSECONDS_PER_MILLISECOND
        df.drop(columns=['confidence'], inplace=True)
        df.rename(columns=cls.__COLUMNS_MAPPING, inplace=True)

        # add a column for trial number:
        # trials are instances that share the same subject id & stimulus.
        trial_counter = 1
        df[cnst.TRIAL] = np.nan
        for _, trial_df in df.groupby([cnst.SUBJECT_ID, cls._STIMULUS_NAME_STR]):
            df.loc[trial_df.index, cnst.TRIAL] = trial_counter
            trial_counter += 1
        df[cnst.TRIAL] = df[cnst.TRIAL].astype(int)
        return df

    @classmethod
    def __read_zipfile(cls, zf: zp.ZipFile) -> pd.DataFrame:
        """
        Reads the contents of a zip file and returns a DataFrame with the annotated data.
        :param zf: ZipFile object
        :return: DataFrame with annotated gaze data
        """
        # Get Annotated Data
        prefix = psx.join('gazecom_annotations', 'ground_truth')
        annotated_file_names = [f for f in zf.namelist() if (f.endswith('.arff') and prefix in f)]
        gaze_dfs = []
        for f in annotated_file_names:
            file = zf.open(f)
            file_str = file.read().decode('utf-8')
            data = arff.loads(file_str)

            # extract data:
            df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])

            # add metadata columns:
            _path, file_name, _ext = ioutils.split_path(f)
            subj_id = file_name.split('_')[0]  # filename format: <subject_id>_<stimulus>_<name>_<with>_<underscores>.arff
            stimulus = '_'.join(file_name.split('_')[1:])
            df[cnst.SUBJECT_ID] = subj_id
            df[cls._STIMULUS_NAME_STR] = stimulus

            # add to list of dataframes
            gaze_dfs.append(df)

        # merge dataframes
        merged_df = pd.concat(gaze_dfs, ignore_index=True, axis=0)

        # add meta data columns:
        merged_df[cnst.STIMULUS] = "video"
        merged_df[cls._VIEWER_DISTANCE_CM_STR] = cls.__VIEWER_DISTANCE_CM_VAL
        merged_df[cls._PIXEL_SIZE_CM_STR] = cls.__SCREEN_MONITOR.pixel_size
        return merged_df
