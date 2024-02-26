import io
import re
import requests as req
import zipfile as zp
import posixpath as psx
import pandas as pd
import numpy as np
import scipy.io as sio
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

    Note 1: This is only a subset of the full GazeCom Dataset, containing hand-labelled samples. The full dataset with
    documentation can be found in https://www.inb.uni-luebeck.de/index.php?id=515.
    Note 2: binocular data was recorded but only one pair of (x, y) coordinates is provided.

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
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))

        # Get Annotated Data
        prefix = psx.join('gazecom_annotations', 'ground_truth')
        annotated_file_names = [f for f in zip_file.namelist() if (f.startswith(prefix) and f.endswith('.arff'))]
        gaze_dfs = []
        for f in annotated_file_names:
            file = zip_file.open(f)
            file_str = file.read().decode('utf-8')
            data = arff.loads(file_str)

            # extract data:
            df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])

            # add metadata columns:
            _path, file_name, _ext = ioutils.split_path(f)
            subj_id, stimulus = file_name.split('_')  # filename format: <subject_id>_<stimulus name>.arff
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


    # @staticmethod
    # def __extract_event_mapping(description: str) -> dict:
    #     """
    #     Event mappings are lines in the description that are of format "<int> is <event_type>" (e.g. "1 is FIX")
    #     Extracts these lines and returns a dictionary with the mappings.
    #
    #     :param description: string containing the description of the dataset
    #     :return: event_dict: dictionary with the mappings
    #     """
    #     # TODO: check if these mappings are the same for all files (and can be a class attribute)
    #     lines = [l for l in description.split('\n') if re.search(GazeComDataSetLoader.__METADATA_DICT_PATTERN, l)]
    #     event_dict = {}
    #     for l in lines:
    #         substring = re.search(GazeComDataSetLoader.__METADATA_DICT_PATTERN, l).group(0)
    #         key, value = substring.split(' is ')
    #         value = value.lower()
    #
    #         # convert value to GazeEventTypeEnum
    #         if any([e in value for e in ["noise", "unknown", "undefined"]]):
    #             ev = cnst.EVENTS.UNDEFINED
    #         elif any([e in value for e in ["fix", "fixation"]]):
    #             ev = cnst.EVENTS.FIXATION
    #         elif any([e in value for e in ["sac", "saccade"]]):
    #             ev = cnst.EVENTS.SACCADE
    #         elif any([e in value for e in ["pursuit", "smooth pursuit", "sp"]]):
    #             ev = cnst.EVENTS.SMOOTH_PURSUIT
    #         elif any([e in value for e in ["blink"]]):
    #             ev = cnst.EVENTS.BLINK
    #         else:
    #             raise ValueError(f"Unknown event type: {value}")
    #
    #         event_dict[int(key)] = ev
    #         event_dict[str(key)] = ev
    #     return event_dict
    #
    # @staticmethod
    # def __extract_subject_metadata(description: str) -> (ScreenMonitor, float):
    #     """
    #     Extracts the metadata of the subject from the description string. Metadata lines are prefixed with '@METADATA'.
    #     :param description: string containing the description of the dataset
    #     :return: metadata: dictionary with the metadata
    #     """
    #     # TODO: check if these metadata are the same for all files (and can be a class attribute)
    #     prefix = '@METADATA '
    #     lines = [l[len(prefix):] for l in description.split('\n') if l.startswith(prefix)]
    #     metadata = {}
    #     for l in lines:
    #         key, value = l.split(' ')
    #         key = key.lower()
    #         value = float(value)
    #         metadata[key] = value
    #
    #     distance_cm = metadata['distance_mm'] / 10
    #     screen = ScreenMonitor(width=metadata['width_mm'] / 10, height=metadata['height_mm'] / 10,
    #                            resolution=(int(metadata['width_px']), int(metadata['height_px'])),
    #                            refresh_rate=30)  # default refresh rate
    #     return screen, distance_cm

