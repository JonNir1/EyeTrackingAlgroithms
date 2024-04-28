import numpy as np
import pandas as pd
from typing import Tuple

import Config.experiment_config as cnfg
from GazeEvents.BaseEvent import BaseEvent
import Utils.visual_angle_utils as visang_utils


class FixationEvent(BaseEvent):
    _EVENT_LABEL = cnfg.EVENT_LABELS.FIXATION

    @property
    def center_of_mass(self) -> Tuple[float, float]:
        """ returns the mean coordinates of the fixation on the X,Y axes """
        x_mean = float(np.nanmean(self._x))
        y_mean = float(np.nanmean(self._y))
        return x_mean, y_mean

    @property
    def standard_deviation_px(self) -> Tuple[float, float]:
        """ returns the standard deviation of the fixation (in pixel units) """
        x_std = float(np.nanstd(self._x))
        y_std = float(np.nanstd(self._y))
        return x_std, y_std

    @property
    def standard_deviation_deg(self) -> Tuple[float, float]:
        """ returns the standard deviation of the fixation (in visual degrees units) """
        x_std, y_std = self.standard_deviation_px
        x_deg = visang_utils.pixels_to_visual_angle(num_px=x_std,
                                                    d=self._viewer_distance,
                                                    pixel_size=self._pixel_size,
                                                    use_radians=False)
        y_deg = visang_utils.pixels_to_visual_angle(num_px=y_std,
                                                    d=self._viewer_distance,
                                                    pixel_size=self._pixel_size,
                                                    use_radians=False)
        return x_deg, y_deg

    @property
    def dispersion_px(self) -> float:
        """ returns the maximum distance between any two points in the fixation (in pixels units) """
        points = np.column_stack((self._x, self._y))
        distances = np.linalg.norm(points - points[:, None], axis=-1)
        max_dist = float(np.nanmax(distances))
        return max_dist

    @property
    def dispersion_deg(self) -> float:
        """ returns the maximum distance between any two points in the fixation (in visual degrees units) """
        max_dist_px = self.dispersion_px
        return visang_utils.pixels_to_visual_angle(num_px=max_dist_px,
                                                   d=self._viewer_distance,
                                                   pixel_size=self._pixel_size,
                                                   use_radians=False)

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - center_of_mass: fixation's center of mass (2D pixel coordinates)
            - standard_deviation_px: fixation's standard deviation (in pixel units)
            - dispersion_px: maximum distance between any two points in the fixation (in pixels units)
        """
        series = super().to_series()
        series["center_of_mass"] = self.center_of_mass
        series["standard_deviation_px"] = self.standard_deviation_px
        series["dispersion_px"] = self.dispersion_px
        return series



