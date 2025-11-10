def build_metric_loader(config_path: str, metricinfo_json=None):
    """使用 AnteaterConf YAML 配置构建 MetricLoader"""
    from anteater.config import AnteaterConf
    from anteater.core.info import MetricInfo
    from anteater.source.metric_loader import MetricLoader
    import os

    cfg = AnteaterConf()
    cfg.load_from_yaml(os.path.dirname(config_path))
    minfo = MetricInfo(**metricinfo_json) if metricinfo_json else MetricInfo()
    return MetricLoader(metricinfo=minfo, config=cfg)
