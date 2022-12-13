# 配置文件介绍
gala-anteater启动必须的参数，主要通过`gala-anteater.yaml`配置文件，主要包括：全局参数配置、依赖中间件配置、以及相关启动选项。除此之外，还可以配置`Logger`相关参数，以及配置`gala-anteater`异常检测各个子模块相关的参数。

全部配置文件归档在[config](https://gitee.com/openeuler/gala-anteater/tree/master/config)目录。

## 配置文件目录结构
启动配置文件目录结构如下，主要分为两类：`启动参数配置`和`日志参数配置`。
```
gala-anteater               # gala-anteater 主目录
└─ config                   # 配置文件目录
   ├─ gala-anteater.yaml    # 启动参数配置
   ├─ log.settings.ini      # 日志参数配置
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
