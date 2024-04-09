import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent
import Utils.pixel_utils as pixel_utils


class SaccadeEvent(BaseEvent):
    _EVENT_LABEL = cnst.EVENT_LABELS.SACCADE

    @final
    @property
    def azimuth(self) -> float:
        """ returns the azimuth of the saccade in degrees """
        return pixel_utils.calculate_azimuth(p1=self.start_point, p2=self.end_point, use_radians=False)

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - azimuth: saccades's azimuth in degrees
        """
        series = super().to_series()
        series["azimuth"] = self.azimuth
        return series
