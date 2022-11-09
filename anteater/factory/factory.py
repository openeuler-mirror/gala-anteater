from anteater.config import AnteaterConf
from anteater.provider.aom import create_aom_collector
from anteater.provider.base import TimeSeriesProvider
from anteater.provider.prometheus import load_prometheus_client


class DataCollectorFactory:
    """The data collector factory"""

    @staticmethod
    def get_instance(data_source: str, config: AnteaterConf) -> TimeSeriesProvider:
        """Gets data collector based on the data source name"""
        if data_source == 'prometheus':
            return load_prometheus_client(config.prometheus)
        elif data_source == 'aom':
            return create_aom_collector(config.aom)
        raise ValueError("Unknown data source:{}, please check!".format(data_source))
