import os
import time
import warnings
from abc import ABC, abstractmethod
from typing import final, Set, Dict

import pandas as pd
import pickle as pkl


class BaseCalculator(ABC):
    __CALCULATOR_STR = "Calculator"

    @classmethod
    def calculate(
            cls,
            file_path: str,
            data: pd.DataFrame,
            metric_names: Set[str],
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            if verbose:
                print(f"Calculating {cls._name()}...")
            try:
                with open(file_path, 'rb') as f:
                    results = pkl.load(f)
                missing_metrics = metric_names - set(results.keys())
                if missing_metrics:
                    new_results = cls._calculate_impl(data, missing_metrics)
                    results.update(new_results)
                    cls._save(results, file_path)
            except FileNotFoundError:
                results = cls._calculate_impl(data, metric_names)
                cls._save(results, file_path)
            end = time.time()
            if verbose:
                print(f"\tCompleted:\t{end - start:.2f}s")
        return results

    @classmethod
    @abstractmethod
    def _calculate_impl(
            cls,
            data: pd.DataFrame,
            metric_names: Set[str],
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    @classmethod
    def _name(cls) -> str:
        class_name = cls.__name__
        return class_name[:class_name.index(cls.__CALCULATOR_STR)]

    @staticmethod
    @final
    def _save(results: dict, file_path: str):
        directory = os.path.dirname(os.path.abspath(file_path))
        if not os.path.exists(file_path):
            os.makedirs(directory, exist_ok=True)
        with open(file_path, "wb") as f:
            pkl.dump(results, f, protocol=-1)

    @classmethod
    def __get_file_path(cls, data_dir: str) -> str:
        file_name = cls._name() + ".pkl"
        return os.path.join(data_dir, file_name)
