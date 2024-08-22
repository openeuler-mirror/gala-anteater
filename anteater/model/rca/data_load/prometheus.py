from anteater.model.rca.data_load.base import TimeSeriesProvider


class PrometheusAdapter(TimeSeriesProvider):
    """The PrometheusAdapter client to consume time series data"""

    def __init__(self, server, port):
        """The PrometheusAdapter client initializer"""
        self.query_url = f"http://{server}:{port}/api/v1/query_range"
        super().__init__(self.query_url)

    def get_headers(self):
        """Gets the requests headers of prometheus"""
        return {}


def load_prometheus_client() -> PrometheusAdapter:
    """Load and initialize the prometheus client"""
    # server = "192.168.122.21"
    server = "localhost"
    port = "9090"
    client = PrometheusAdapter(server, port)

    return client
