# gala-anteater
## 介绍
gala-anteater 是一款基于 AI 的操作系统灰度故障的异常检测平台，其集成了多种异常检测算法，针对不同场景和应用，实现实时地系统级故障发现、以及故障点的上报。

anteater 基于系统历史数据，进行自动化模型预训练、线上模型的增量学习和模型更新，能够很好地适应多场景、多指标型数据，实现分钟级模型推理能力。

## 支持的异常检测场景汇总
当前 anteater 支持 3 大故障类别，13 种不同子场景的异常检测。

| 类别              | 诊断场景                               | KPI                                                                                                                                                                                      | 故障注入方式                                                           |
|-----------------|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| 应用级             | 应用时延（RTT）                          | gala_gopher_sli_rtt_nsec                                                                                                                                                                 | chaosblade: network loss/delay, disk fill/burn, cpu              |
|                 | 应用吞吐量（TPS）                         | gala_gopher_sli_tps                                                                                                                                                                      | chaosblade: network loss/delay, disk fill/burn, cpu              |
| 系统级             | TCP建链性能                            | gala_gopher_tcp_link_syn_srtt                                                                                                                                                            | chaosblade: network delay                                        |
|                 | TCP传输性能                            | gala_gopher_tcp_link_srtt                                                                                                                                                                | chaosblade: network loss                                         |
|                 | 系统I/O性能                            | gala_gopher_block_latency_req_max                                                                                                                                                        | chaosblade: disk burn                                            |
|                 | 进程I/O性能                            | gala_gopher_proc_bio_latency<br/>gala_gopher_proc_less_4k_io_read<br/>gala_gopher_proc_less_4k_io_write<br/>gala_gopher_proc_greater_4k_io_read<br/>gala_gopher_proc_greater_4k_io_write | chaosblade: disk burn                                            |
|                 | 磁盘吞吐量                              | gala_gopher_disk_r_await<br/>gala_gopher_disk_w_await                                                                                                                                    | chaosblade: disk full                                            |
|                 | 网卡发送丢包                             | gala_gopher_nic_tc_sent_drop                                                                                                                                                             | chaosblade: network loss                                         |
| [JVM OutOfMemory](docs/jvm_oom_introduction.md) | Heapspace                          | gala_gopher_jvm_mem_bytes_used<br/>gala_gopher_jvm_mem_pool_bytes_used                                                                                                                   | java code: JavaOOMHttpServer |
|                 | GC Overhead                        | gala_gopher_jvm_mem_bytes_used<br/>gala_gopher_jvm_mem_pool_bytes_used                                                                                                                   | java code: JavaOOMHttpServer |
|                 | Metaspace                          | gala_gopher_jvm_class_current_loaded                                                                                                                                                     | java code: JavaOOMHttpServer |
|                 | Unable to create new native thread | gala_gopher_jvm_threads_current                                                                                                                                                          | java code: JavaOOMHttpServer |
|                 | Direct buffer memory               | gala_gopher_jvm_buffer_pool_used_bytes                                                                                                                                                   | java code: JavaOOMHttpServer |

## 安装部署

### 前置条件
* 支持的python版本：3.7+；
* anteater 依赖于 gopher 采集的数据，请先完成 gopher 的安装部署；
* anteater 直接从 Prometheus 中获取时序指标型数据，需要完成 Prometheus 的安装部署；
* anteater 依赖于 gopher 上报的 meta 数据（上报至 Kafka），因为需要确保 Kafka 安装部署完成。

### 方法一：Docker镜像安装（适用于普通用户）
#### Docker镜像制作
请在工程`./gala-anteater`目录下，执行下面的命令，将`gala-anteater`工程文件打包成Docker镜像。
```
docker build -f Dockerfile -t gala-anteater:1.0.0 .
```
注：根据环境网络情况，可能需要修改`Dockfile`文件中的`pip`源地址

#### Docker镜像运行
执行下面的命令，运行Docker镜像。首次运行会将配置文件`gala-anteater.yaml`文件映射到宿主机`/etc/gala-anteater/config`文件中，
请配置`gala-anteater.yaml`里面的参数，配置方式，请参考[配置文件介绍](https://gitee.com/openeuler/gala-anteater/blob/master/docs/conf_introduction.md)。
```
docker run -v /etc/gala-anteater:/etc/gala-anteater -it gala-anteater:1.0.0
```

### 方法二：从本仓库源码安装运行（适用于开发者）
#### 下载源码
```
 git clone https://gitee.com/openeuler/gala-anteater.git
```

#### 安装
工程`./gala-anteater`目录下执行下面命令：
```bash
python3 setup.py install
```

#### 参数配置
配置参数会被映射到`/etc/gala-anteater/config`文件中，需要首先设置相应的参数，配置方式，请参考[配置文件介绍](https://gitee.com/openeuler/gala-anteater/blob/master/docs/conf_introduction.md)。


注：在配置文件中，最重要的是完成配置文件中中间件的配置，如其中`Kafka server/port`、`Prometheus server/port`。

#### 运行
```
systemctl start gala-anteater
```

### 日志
日志文件默认路径：`/var/gala-anteater/logs/`，也可以根据配置文件`log.settings.ini`去修改日志文件的路径。

### 异常上报
gala-anteater输出异常检测结果到`Kafka`，如果检测到异常，检测结果输出到`Kafka`中，默认`Topic`为：`gala_anteater_hybrid_model`，也可以在`gala-anteater.yaml`中修改配置。查看异常检测结果，具体命令如下：

```bash
./bin/kafka-console-consumer.sh --topic gala_anteater_hybrid_model --from-beginning --bootstrap-server localhost:9092
```

## 异常检测结果

异常检测结果默认输出到`Kafka`中，也可存储到`ArangoDB`中，供第三方运维系统查询、集成。数据格式遵循`OpenTelemetry V1`规范，具体方式请参考 [Kafka to ArangoDB](docs/kafka_to_arangodb.md)。如下介绍异常检测输出格式

### 输出数据

#### 输出数据格式

| 参数 |  参数含义  | 描述 |
|:---:|:------:|---|
| Timestamp |  时间戳   | 异常事件上报时间戳(datetime.now(timezone.utc).timestamp()) |
| Attributes |  属性值   | 主要包括实体ID:entity_id<br>* entity_id命名规则：<machine_id>_<table_name>_<keys> |
| Resource |   资源   | 异常检测模型输出的信息，主要包括：<br>* metric_id: 异常检测的主指标<br>* labels: 异常metric标签信息（例如：{"machine_id": "xxx", "tgid": "1234", "conn_fd": "xx"}）<br>* cause_metrics: 推荐的 Top N 根因信息 <br>  |
| SeverityText | 异常事件类型 | INFO WARN ERROR FATAL |
| SeverityNumber | 异常事件编号 | 9, 13, 178, 21 ... |
| Body | 异常事件信息 | 字符串，对当前异常事件的描述信息 |

#### 输出数据示例

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

## 项目路线图

### 异常检测能力

| 特性                                              | 发布时间 | 发布版本                             |
| ------------------------------------------------- | -------- | ------------------------------------ |
| 单维时序数据异常检测（Redis / PG应用性能劣化）       | 22.12    | openEuler 22.03 SP1                  |
| 多维时序数据异常检测（TCP建链 / 传输 / 系统IO ）        | 22.12    | openEuler 22.03 SP1                  |
| 多维阈值异常检测（JAVA OOM类异常）                 | 23.09    | openEuler 22.03 SP1, openEuler 23.09 |
| 异常检测准确率提升（训练集压缩感知离群点过滤技术 + 多指标重构技术 + 异常度动态阈值技术）  | 23.09    | openEuler 22.03 SP1, openEuler 23.09 |
| 异常检测泛化能力提升（平稳 / 非平稳背景流自适应技术） | 23.09    | openEuler 22.03 SP1, openEuler 23.09 |
| 异常检测泛化能力提升（在线学习 + 增量学习技术）      | 23.09    | openEuler 22.03 SP1, openEuler 23.09 |
| 白名单应用性能劣化异常检测                        | 24.03    | openEuler 24.03                      |


### 根因定位能力

| 特性                                                         | 发布时间 | 发布版本                             |
| ------------------------------------------------------------ | -------- | ------------------------------------ |
| 基于专家规则的应用性能劣化根因定位（虚拟化、分布式存储场景网络IO / 磁盘IO类故障） | 22.12    | openEuler 22.03 SP1                  |
| 基于因果图构建、因果传播分析的根因定位（根因传播推导技术 + 根因路径溯源技术）   | 23.09    | openEuler 22.03 SP1, openEuler 23.09 |
| 根因定位准确率提升（PC算法因果图 + 专家经验）                | 24.03    | openEuler 24.03                      |
| 资源类异常通用根因定位（基于图谱的多变量时间序列）             | 24.03    | openEuler 24.03                      |
| 多模态应用性能劣化根因定位（基于Metric、Logging、Tracing）   | 24.09    | openEuler 24.09                      |

