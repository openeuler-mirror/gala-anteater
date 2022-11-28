# 配置文件介绍
gala-anteater启动必须的参数通过`gala-anteater.yaml`配置，主要包括：全局参数配置、依赖中间件配置、以及相关启动选项。除此之外，还可以配置`Logger`相关参数，以及配置`gala-anteater`异常检测各个子模块相关的参数。

全部配置文件归档在[config](https://gitee.com/openeuler/gala-anteater/tree/master/config)目录。

## 配置文件目录结构
配置文件目录结构如下，主要分为三类：启动参数配置、日志参数配置、异常检测子模块参数配置。
```
gala-anteater               # gala-anteater 主目录
└─ config                   # 配置文件目录
   ├─ gala-anteater.yaml    # 启动参数配置
   ├─ log.settings.ini      # 日志参数配置
   └─ module/               # 异常检测子模块参数配置
      ├─ app_sli_rtt.json            # app-sli-rtt 参数配置
      ├─ proc_io_latency.json        # proc-io-latency 参数配置
      ├─ sys_io_latency.json         # sys-io-latency 参数配置
      ├─ sys_tcp_establish.json      # sys-tcp-establish 参数配置
      ├─ sys_tcp_transmission_latency.json          # sys-tcp-transmission-latency 参数配置
      └─ sys_tcp_transmission_throughput.json       # sys-tcp-transmission-throughput 参数配置
```

## 启动参数配置
在文件`gala-anteater.yaml`中，配置`gala-anteater`启动时所需的参数。该配置项中，主要包含：
- Global: 配置启动时的全局变量
  - data_source: 时序数据的来源，目前支持`"prometheus"`（Prometheus）和`"aom"`（AOM）两种数据来源；
  - sys_level: 是否支持`系统级`异常检测，可选：`true`、`false`。

- Kafka: 配置中间件Kafka所需的参数
  - server: Kafak对应的`server ip`，如："10.xxx.xxx.xxx"；
  - port: Kafka对应的`server port`，如："9092"；
  - model_topic: gala-anteater异常数据数据上报至Kafka，所需的`Topic`；
  - meta_topic: gala-anteater所依赖的`metadata`数据，在Kafka中所对应的`Topic`。

- Prometheus: 配置中间件Prometheus所需的参数
  - server: Prometheus对应的`server ip`, 如："10.xxx.xxx.xxx"；
  - port: Prometheus对应的`server port`, 如："9090"；
  - setps: 数据读取的时间间隔（单位：秒），如：5表示每5秒钟采集一个数据点。

- AOM: 配置中间件AOM(华为云对接指标数据库)所需的参数
  - base_url: AOM服务器地址；
  - project_id: AOM项目ID；
  - auth_type: AOM服务器鉴权类型，支持token、appcode两种方式；
  - auth_info: AOM服务器鉴权配置信息：
    - iam_server: iam服务器；
    - iam_domain: iam域；
    - iam_user_name: iam用户名；
    - iam_password: iam用户密码；
    - ssl_verify: 是否开启SSL证书验证，默认为0表示关闭。

- Schedule: 配置`gala-anteater`运行周期，目前仅支持以**固定时间间隔**运行
  - duration: 启动时间间隔（单位：分钟）：例如：1表示每分钟运行一次。


## 日志参数配置
在文件`log.settings.ini`中，配置日志所需的参数。该配置文件采用`Python`标准的`logging`配置方式，支持设置日志的格式、日志的文件类型、单个文件的大小，文件流的去向等。

- 对于日志文件的配置方式，请参考[Logging HOWTO](https://docs.python.org/3/howto/logging.html)。

## 异常检测子模块参数配置

在文件夹`module/`中，包含异常检测各个子模块的参数配置。异常检测包括很多应用级和系统级的异常检测任务，每个任务被认为是异常检测的一个子模块，它包含异常检测的主指标（KPI）、特征量（Features）、以及模型相关的特征量。

以`app_sli_rtt.json`为例，其中主要的参数为：

- name: 异常检测任务的名称;
- job_type: 任务对应的类型，当前支持两种类型的任务：`app`和`sys`;
- root_cause_number: 当前任务跟因推荐的最大个数；
- KPI: 任务对应的`kpi`指标
  - metric: 当前`kpi`对应的`metric name`;
  - kpi_type: kpi类型，如：sli、tcp、process等，没有固定的类型，主要依据模型的具体实现方式；
  - entity_name: 当前metric对应`metadata`文件（gala-gopher上报的数据）中对应的`entity_name`；
  - enable: 是否使用当前指标进行异常检测：`true`或者`false`;
  - description: 当前kpi发生异常的描述信息；
  - params: 对当前kpi做异常检测所需的参数信息，具体内容取决于所使用的模型。
- Features: 当前异常检测任务对应的特征量，主要用于跟因定位
  - metric: 当前特征量对应的`metric name`；
  - description: 当前特征发生异常对应的异常信息模板；
  - atrend: 当前metric发生异常的条件（趋势信息），包括：`rise/fall/default`。其中`rise`表示上升才能导致异常，`fall`表示下降才能导致异常，`default`表示上升和下降均可以导致异常。
- OnlineModel：在线学习使用的参数信息，当前使用VAE-Based的深度学习模型，进行在线学习
  - name: 在线学习名称;
  - enable: 是否启动在线学习: `true`或者`false`;
  - params: 模型中使用的参数信息，参数的具体信息依赖于具体实施的模型细节。