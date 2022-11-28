# gala-anteater 介绍
gala-anteater是一款基于AI的操作系统异常检测平台。主要涵盖时序数据预处理、异常点发现、以及异常上报等功能。
基于线下预训练、线上模型的增量学习与模型更新，能够很好地适应于多维多模态数据故障诊断。

## 1. 安装gala-anteater
支持的python版本：3.7+

### 1.1 方法一：Docker镜像安装（适用于普通用户）
#### 1.1.1 Docker镜像制作
请在工程`./gala-anteater`目录下，执行下面的命令，将`gala-anteater`工程文件打包成Docker镜像。
```
docker build -f Dockerfile -t gala-anteater:1.0.0 .
```
注：根据环境网络情况，可能需要修改`Dockfile`文件中的`pip`源地址

#### 1.1.2 Docker镜像运行
执行下面的命令，运行Docker镜像。首次运行会将配置文件`gala-anteater.yaml`文件映射到宿主机`\etc\gala-anteater\config`文件中，
请配置`gala-anteater.yaml`里面的参数，配置方式，请参考[配置文件介绍](https://gitee.com/openeuler/gala-anteater/blob/master/docs/conf_introduction.md)。
```
docker run -v /etc/gala-anteater:/etc/gala-anteater -it gala-anteater:1.0.0
```

#### 1.1.3 日志查看
日志文件路径：`/etc/gala-anteater/logs/`

#### 1.1.4 运行结果查看
如果检测到异常，检测结果输出到`Kafka`中，默认`Topic`为：`gala_anteater_hybrid_model`，也可以在`gala-anteater.yaml`中修改配置

### 1.2 方法二：从本仓库源码安装运行（适用于开发者）
#### 1.2.1 下载源码
```
 git clone https://gitee.com/openeuler/gala-anteater.git
```

#### 1.2.2 安装python依赖包  
工程`./gala-anteater`目录下执行下面命令：
```bash
python3 setup.py install
```

#### 1.2.3 参数配置
配置参数会被映射到`\etc\gala-anteater\config`文件中，需要首先设置相应的参数，配置方式，请参考[配置文件介绍](https://gitee.com/openeuler/gala-anteater/blob/master/docs/conf_introduction.md)。


注：在配置文件中，最重要的是完成配置文件中中间件的配置，如其中`Kafka server/port`、`Prometheus server/port`。

#### 1.2.4 程序运行
```
gala-anteater
```

#### 1.2.4 日志查看
日志文件路径：`/var/gala-anteater/logs/`

#### 1.2.5 运行结果查看

如果检测到异常，检测结果输出到`Kafka`中，默认`Topic`为：`gala_anteater_hybrid_model`，也可以在`gala-anteater.yaml`中修改配置。

## 2. 快速使用指南

### 2.1 启动gala-anteater服务

按照1中的方式启动服务，命令如下：

```shell
docker run -v /etc/gala-anteater:/etc/gala-anteater -it gala-anteater:1.0.0
```

或者直接通过下列命令，运行程序（需要将工程目录设置为Python工作目录）。

```shell
python ./anteater/main.py
```

启动结果，可查看运行日志。

### 2.2 异常检测结果信息查看
gala-anteater输出异常检测结果到`Kafka`，可使用`Kafka`命令查看异常检测结果，具体命令如下：

```bash
./bin/kafka-console-consumer.sh --topic gala_anteater_hybrid_model --from-beginning --bootstrap-server localhost:9092
```

## 3. 异常检测结果API文档
### 3.1 API说明

异常检测结果默认输出到`Kafka`中，也可存储到`ArangoDB`中，供第三方运维系统查询、集成。数据格式遵循`OpenTelemetry V1`规范。

本文档介绍异常检测格式，`Kafka、Arangodb`的API参考其官方文档。

### 3.2 输出数据

#### 3.2.1 输出数据格式

| 参数 |  参数含义  | 描述 |
|:---:|:------:|---|
| Timestamp |  时间戳   | 异常事件上报时间戳(datetime.now(timezone.utc).timestamp()) |
| Attributes |  属性值   | 主要包括实体ID:entity_id<br>* entity_id命名规则：<machine_id>_<table_name>_<keys> |
| Resource |   资源   | 异常检测模型输出的信息，主要包括：<br>* metric_id: 异常检测的主指标<br>* labels: 异常metric标签信息（例如：{"machine_id": "xxx", "tgid": "1234", "conn_fd": "xx"}）<br>* cause_metrics: 推荐的 Top N 根因信息 <br>  |
| SeverityText | 异常事件类型 | INFO WARN ERROR FATAL |
| SeverityNumber | 异常事件编号 | 9, 13, 178, 21 ... |
| Body | 异常事件信息 | 字符串，对当前异常事件的描述信息 |

#### 3.2.2 输出数据示例

```json
{
    "Timestamp": 1669343170074,
    "Attributes": {
        "entity_id": "7c2fbaf8-4528-4aaf-90c1-5c4c46b06ebe_sli_2187425_16859_POSTGRE_0",
        "event_id": "1669343170074_7c2fbaf8-4528-4aaf-90c1-5c4c46b06ebe_sli_2187425_16859_POSTGRE_0",
        "event_type": "app",
        "event_source": "gala-anteater"
    },
    "Resource": {
        "metric": "gala_gopher_sli_tps",
        "labels": {
            "app": "POSTGRE",
            "datname": "tpccdb",
            "ins_id": "16859",
            "instance": "10.xxx.xxx.xxx:18001",
            "job": "prometheus-10.xxx.xxx.xxx:8001",
            "machine_id": "7c2fbaf8-4528-4aaf-90c1-5c4c46b06ebe",
            "method": "0",
            "server_ip": "172.xxx.xxx.xxx",
            "server_port": "5432",
            "tgid": "2187425"
        },
        "score":0.36,
        "cause_metrics": [
            {
                "metric": "gala_gopher_net_tcp_retrans_segs",
                "labels": {
                    "instance": "10.xxx.xxx.xxx:18001",
                    "job": "prometheus-10.xxx.xxx.xxx:8001",
                    "machine_id": "7c2fbaf8-4528-4aaf-90c1-5c4c46b06ebe",
                    "origin": "/proc/dev/snmp"
                },
                "score": 16.982238591106373,
                "description": "TCP重传的分片数异常"
            },
            {
                "metric": "gala_gopher_cpu_user_total_second",
                "labels": {
                    "cpu": "6",
                    "instance": "10.xxx.xxx.xxx:18001",
                    "job": "prometheus-10.xxx.xxx.xxx:8001",
                    "machine_id": "7c2fbaf8-4528-4aaf-90c1-5c4c46b06ebe"
                },
                "score": 6.116480503130946,
                "description": "用户态cpu占用时间（不包括nice）异常"
            },
            {
                "metric": "gala_gopher_disk_w_await",
                "labels": {
                    "disk_name": "sda",
                    "instance": "10.xxx.xxx.xxx:18001",
                    "job": "prometheus-10.xxx.xxx.xxx:8001",
                    "machine_id": "7c2fbaf8-4528-4aaf-90c1-5c4c46b06ebe"
                },
                "score": 5.958243288864987,
                "description": "写响应时间异常"
            }
        ],
        "description": "sli tps 异常"
    },
    "SeverityText": "WARN",
    "SeverityNumber": 13,
    "Body": "Fri Nov 25 10:26:10 2022 WARN, APP may be impacting sli performance issues.",
    "event_id": "1669343170074_7c2fbaf8-4528-4aaf-90c1-5c4c46b06ebe_sli_2187425_16859_POSTGRE_0"
}
```

