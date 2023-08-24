# 开发者文档

## gala-anteater介绍

gala-anteater是一款基于AI的操作系统异常检测平台。主要涵盖时序数据预处理、异常点发现、以及异常上报等功能。gala-anteater基于模型线下预训练、线上模型的在线/迁移学习，进行模型更新，能够很好地适应于多场景、多维、多模态数据故障诊断。


## 目录结构
gala-anteater的主要目录结构如下：
```
gala-anteater
├─ anteater/      # 项目目录
│  ├─ core/       # 数据结构模块
│  ├─ factory/    # 工厂模式方法
│  ├─ main.py     # main方法，执行入口
│  ├─ model/      # AI 模型集合，内置多种AI模型
│  ├─ module/     # 异常检测场景
│  ├─ provider/   # 数据提供接口
│  ├─ source/     # 数据整合接口
│  ├─ template/   # 异常结果上报模板
│  ├─ utils/      # 常用函数或方法
├─ config/        # 配置文件
└─ tests/         # 单元测试
```

其中，各个子目录的详细介绍介绍，如下：

1. `anteater/main.py`: 执行的主入口，调用AnomalyDetection模块，并周期性执行AnomalyDetection中的run方法。
2. `anteater/core/`: 数据结构类，主要用于存储各组件输入、输出的数据类型。
3. `anteater/factory/`：工厂模式方法，用于创建具体的对象。
4. `anteater/model/`: 存放项目中使用到的各种算法，并被`anteater/module/`下面的各个模型调用。
5. `anteater/module/`: 为不同场景下的异常检测模块，执行具体场景下的异常检测算法，并输出检测结果。
6. `anteater/provider/`: 数据提供的接口，因为数据存储在不同的Storage中，如：Kafka, Prometheus, ArangoDB等。
7. `anteater/source/`: 数据整合接口，读取数据时，会在这里进行预处理，并生成模型可用的数据格式。
8. `anteater/template/`: 数据上报格式模板，最终接口生成json格式文件，模板提前定义好。
9. `anteater/utils`: 常用的函数或方法。
10. `config/`: 配置文件，如log、server ip, server port，模型对应的config等。
11. `tests/`: 单元测试


## 添加新的异常检测任务

### 基本概念
gala-aneater中，一个异常检测任务主要包含两部分：
* KPI: 主要观测指标，由一个或多个指标组成。其主要用于异常检测模型的输入，利用单指标、多指标模型，进行异常检测，并输出异常结果。
* Feature：特征量，由一个或多个指标组成。其主要作用为：1) 作为异常检测特征的补充，结合KPI一起做异常检测；2) 作为根因推理的输入，并输出最终的根因。

gala-anteater对于一个异常检测任务，能够很好地支持如下两类异常检测模型：
* 单指标异常检测：针对单一指标，进行异常检测任务，最终将单个或多个指标的检测结果，进行合并。
* 多指标异常检测：将多个指标，统一作为异常检测任务的输入，最终得到异常检测结果。

### 示例
下面以`disk read/write await time`异常检测任务（对应的[PR](https://gitee.com/openeuler/gala-anteater/pulls/30)）为例，进行介绍。

#### 1. 确定KPI和Feature
对于特定的任务，可以通过专家经验，或者特征选择技术，选择合适的KPI和Feature。对于`Disk`场景，可以选择如下KPI和Feature:

* KPI:
  * gala_gopher_disk_r_await
  * gala_gopher_disk_w_await
* Feature:
  * gala_gopher_disk_rspeed_kB
  * gala_gopher_disk_wspeed_kB
  * gala_gopher_disk_rareq
  * gala_gopher_disk_wareq

#### 2. 添加异常检测模块
确定异常检测任务的KPI及Feature之后，可以添加具体的异常检测模块。

* 首先，需要在`anteater/module/`文件夹下，通过继承基类E2EDetector，新建该任务对应的`xxxE2EDetector`类，并实现具体的接口。具体接口，请参考基类E2EDetector，其中：
   * config_file：类变量，表示该任务对应的配置文件名称（模型对应的配置文件统一存放在`config/module/`文件夹下）。
   * detectors：成员变量，表示该任务使用到的异常检测模型（异常检测模型代码，统一存放在`anteater/model/`文件夹中）。

* 其次，实施该任务对应的异常检测模型。通过继承Detector，新建异常检测模型，并实现基类的接口。该模块包括：KPI和Feature时序数据采集、执行异常检测模型、根因推理，等。目前已经实现如下两种主要的异常检测模型：
   * N-Sigma模型：基于均值方差的N-Sigma异常检测模型，较好的适应数据量较少、单指标、无标签异常检测任务。
   * VAE模型：基于深度神经网络的Variational Autoencoder模型，能够有效提升异常检测模型的准确率，并且对于多指标、周期性数据，有较好的检测能力。

* 最后，新增配置文件，该任务对于的全部配置文件，统一存放在`config/module/`文件夹中，关于具体配置文件含义，可以参考下一节中的内容。

#### 3. 注册异常检测任务
完成异常检测任务模型模块代码及配置文件之后，需要注册该异常检测任务。此时仅仅需要将该任务对应的`xxxE2EDetector`类，注册到`anteater/main.py`中即可。

### 异常检测任务参数配置指导

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

对于`启动参数配置`、`日志参数配置`，请参考[这里](https://gitee.com/openeuler/gala-anteater/blob/master/docs/conf_introduction.md)。

在文件夹`module/`中，包含异常检测各个子模块的参数配置。异常检测包括很多应用级和系统级的异常检测任务，每个任务被认为是异常检测的一个子模块，它包含异常检测的主指标（KPI）、特征量（Features）、以及模型相关的特征量。

以`app_sli_rtt.json`为例，其中主要的参数为：

- name: 异常检测任务的名称;
- root_cause_num: 当前任务跟因推荐的最大个数；
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
- model_config：在线学习使用的参数信息，当前使用VAE-Based的深度学习模型，进行在线学习
  - name: 在线学习名称;
  - enable: 是否启动在线学习: `true`或者`false`;
  - params: 模型中使用的参数信息，参数的具体信息依赖于具体实施的模型细节。