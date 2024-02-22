import pandas as pd
import requests as req
from abc import ABC, abstractmethod
from typing import final, List, Dict, Union

from Config import constants as cnst
from Config.GazeEventTypeEnum import GazeEventTypeEnum


class BaseDataSetLoader(ABC):

    _URL: str = None
    _ARTICLES: List[str] = None

    @classmethod
    @final
    def download(cls) -> pd.DataFrame:
        """ Downloads the dataset, parses it and returns a DataFrame with cleaned data """
        response = cls._download_raw_dataset()
        df = cls._parse_response(response)
        df = cls._clean_data(df)
        ordered_columns = sorted(df.columns, key=lambda col: cls.column_order().get(col, 10))
        df = df[ordered_columns]  # reorder columns
        return df

    @classmethod
    @final
    def save_to_pickle(cls, df: pd.DataFrame, path_file: str = None) -> None:
        if path_file is None:
            path_file = f"{cls.__name__}.pkl"
        if not path_file.endswith(".pkl"):
            path_file += ".pkl"
        df.to_pickle(path_file)

    @classmethod
    @final
    def articles(cls) -> List[str]:
        """ List of articles that are connected to the creation of this dataset """
        if not cls._ARTICLES:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_ARTICLES`")
        return cls._ARTICLES

    @classmethod
    def column_order(cls) -> Dict[str, float]:
        return {cnst.TRIAL: 0.1, cnst.SUBJECT_ID: 0.2, cnst.STIMULUS: 0.3,
                cnst.MILLISECONDS: 1.1, cnst.X: 1.2, cnst.Y: 1.3,
                cnst.LEFT_X: 2.1, cnst.LEFT_Y: 2.2, cnst.LEFT_PUPIL: 2.3,
                cnst.RIGHT_X: 3.1, cnst.RIGHT_Y: 3.2, cnst.RIGHT_PUPIL: 3.3}

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
            raise RuntimeError(f"Failed to download dataset from {cls._URL}")
        return response

    @staticmethod
    @final
    def _parse_gaze_event(ev: Union[GazeEventTypeEnum, int, str, float], safe: bool = True) -> GazeEventTypeEnum:
        """
        Parses a gaze label from the original dataset's type to type GazeEventTypeEnum
        :param ev: the gaze label to parse
        :param safe: if True, returns GazeEventTypeEnum.UNDEFINED when the parsing fails
        :return: the parsed gaze label
        """
        try:
            if type(ev) not in [GazeEventTypeEnum, int, str, float]:
                raise TypeError(f"Incompatible type: {type(ev)}")
            if isinstance(ev, GazeEventTypeEnum):
                return ev
            if isinstance(ev, int):
                return GazeEventTypeEnum(ev)
            if isinstance(ev, str):
                return GazeEventTypeEnum[ev.upper()]
            if isinstance(ev, float):
                if not ev.is_integer():
                    raise ValueError(f"Invalid value: {ev}")
                return GazeEventTypeEnum(int(ev))
        except Exception as err:
            if safe and (isinstance(err, ValueError) or isinstance(err, TypeError)):
                return GazeEventTypeEnum.UNDEFINED
            raise err

