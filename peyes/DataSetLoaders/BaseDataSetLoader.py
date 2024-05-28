import os
import pandas as pd
import requests as req
from abc import ABC, abstractmethod
from typing import final, List, Dict, Union, Optional

from peyes.Config import constants as cnst
from peyes.Config import experiment_config as cnfg
from peyes.Config.GazeEventTypeEnum import GazeEventTypeEnum


class BaseDataSetLoader(ABC):

    _PIXEL_SIZE_CM_STR = "pixel_size_cm"
    _VIEWER_DISTANCE_CM_STR = "viewer_distance_cm"
    _STIMULUS_NAME_STR = f"{cnst.STIMULUS}_name"

    _URL: str = None
    _ARTICLES: List[str] = None
    _NAME: str = None

    @classmethod
    @final
    def load(cls, directory: Optional[str] = cnfg.DATASETS_DIR, should_save: bool = False) -> pd.DataFrame:
        """
        Loads the dataset from the specified directory. If the dataset is not found, it is downloaded.
        If `should_save` is True, the dataset is saved to the specified directory.

        :return: a DataFrame containing the dataset
        :raises ValueError: if `should_save` is True and `directory` is not specified
        """
        try:
            p = os.path.join(directory, f"{cls.dataset_name()}.pkl")
            dataset = pd.read_pickle(p)
        except FileNotFoundError:
            dataset = cls.download_from_remote()
        if should_save:
            if not directory:
                raise ValueError("Directory must be specified to save the dataset")
            _success = cls.save_pickle(dataset, directory)
        return dataset

    @classmethod
    @final
    def download_from_remote(cls) -> pd.DataFrame:
        """ Downloads the dataset from the internet, parses it and returns a DataFrame with cleaned data """
        response = cls._download_raw_dataset()
        df = cls._parse_response(response)
        df = cls._clean_data(df)
        ordered_columns = sorted(df.columns, key=lambda col: cls.column_order().get(col, 10))
        df = df[ordered_columns]  # reorder columns
        return df

    @classmethod
    @final
    def save_pickle(cls, dataset: pd.DataFrame, directory: str = cnfg.DATASETS_DIR) -> bool:
        filename = f"{cls.dataset_name()}.pkl"
        if not os.path.isdir(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, filename)
        dataset.to_pickle(file_path)
        return os.path.isfile(file_path)

    @classmethod
    @final
    def dataset_name(cls) -> str:
        """ Name of the dataset """
        if not cls._NAME:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_NAME`")
        return cls._NAME

    @classmethod
    @final
    def articles(cls) -> List[str]:
        """ List of articles that are connected to the creation of this dataset """
        if not cls._ARTICLES:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_ARTICLES`")
        return cls._ARTICLES

    @classmethod
    def column_order(cls) -> Dict[str, float]:
        return {cnst.TRIAL: 0.1, cnst.SUBJECT_ID: 0.2, cnst.STIMULUS: 0.3, cls._STIMULUS_NAME_STR: 0.4,
                cnst.T: 1.0, cnst.X: 1.1, cnst.Y: 1.2, cnst.PUPIL: 1.3,
                cnst.LEFT_X: 2.1, cnst.LEFT_Y: 2.2, cnst.LEFT_PUPIL: 2.3,
                cnst.RIGHT_X: 3.1, cnst.RIGHT_Y: 3.2, cnst.RIGHT_PUPIL: 3.3,
                cls._PIXEL_SIZE_CM_STR: 4.2, cls._VIEWER_DISTANCE_CM_STR: 4.3}

    @classmethod
    @abstractmethod
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        """ Parses the downloaded response and returns a DataFrame containing the raw dataset """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """ Cleans the raw dataset and returns a DataFrame containing the cleaned dataset """
        raise NotImplementedError

    @classmethod
    @final
    def _download_raw_dataset(cls):
        if not cls._URL:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_URL`")
        response = req.get(cls._URL)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP status code {response.status_code} received from {cls._URL}")
        return response

    @staticmethod
    @final
    def _extract_filename_and_extension(full_path: str) -> (str, str, str):
        """ Splits a full path into its components: path, filename and extension """
        path, extension = os.path.splitext(full_path)
        path, filename = os.path.split(path)
        return path, filename, extension
