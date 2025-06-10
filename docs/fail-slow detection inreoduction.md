# 慢节点检测特性使用手册

gala-anteater是一款基于AI的操作系统异常检测平台。主要提供时序数据预处理、异常点发现、异常上报等功能。基于线下预训练、线上模型的增量学习与模型更新，能够很好地适用于多维多模态数据故障诊断。

本文主要介绍如何部署和使用gala-anteater服务，检测训练集群中的慢节点/慢卡。

## 安装

挂载repo源：

```basic
[everything]
name=everything
baseurl=http://121.36.84.172/dailybuild/EBS-openEuler-22.03-LTS-SP4/EBS-openEuler-22.03-LTS-SP4/everything/$basearch/
enabled=1
gpgcheck=0
priority=1

[EPOL]
name=EPOL
baseurl=http://repo.openeuler.org/EBS-openEuler-22.03-LTS-SP4/EPOL/main/$basearch/
enabled=1
gpgcheck=0
priority=1

```

安装gala-anteater：

```bash
yum install gala-anteater
```

## 配置

>![](./figures/icon-note.gif)**说明：**
>
>gala-anteater采用配置的config文件设置参数启动，配置文件位置: /etc/gala-anteater/config/gala-anteater.yaml。

##### 配置文件默认参数

```yaml
Global:
  data_source: "prometheus"

Arangodb:
  url: "http://localhost:8529"
  db_name: "spider"

Kafka:
  server: "192.168.122.100"
  port: "9092"
  model_topic: "gala_anteater_hybrid_model"
  rca_topic: "gala_cause_inference"
  meta_topic: "gala_gopher_metadata"
  group_id: "gala_anteater_kafka"
  # auth_type: plaintext/sasl_plaintext, please set "" for no auth
  auth_type: ""
  username: ""
  password: ""

Prometheus:
  server: "localhost"
  port: "9090"
  steps: "5"

Aom:
  base_url: ""
  project_id: ""
  auth_type: "token"
  auth_info:
    iam_server: ""
    iam_domain: ""
    iam_user_name: ""
    iam_password: ""
    ssl_verify: 0

Schedule:
  duration: 1
  
Suppression:
  interval: 10
```

| 参数        | 含义                                                         | 默认值                       |
| ----------- | ------------------------------------------------------------ | ---------------------------- |
| Global      | 全局配置                                                     | 字典类型                     |
| data_source | 设置数据来源                                                 | “prometheus”                 |
| Arangodb    | Arangodb图数据库配置信息                                     | 字典类型                     |
| url         | 图数据库Arangodb的ip地址                                     | "http://localhost:8529"      |
| db_name     | 图数据库名                                                   | "spider"                     |
| Kafka       | kafka配置信息                                                | 字典类型                     |
| server      | Kafka Server的ip地址，根据安装节点ip配置                     | “192.168.122.100”            |
| port        | Kafka Server的port，如：9092                                 | ”9092“                       |
| model_topic | 故障检测结果上报topic                                        | "gala_anteater_hybrid_model" |
| rca_topic   | 根因定位结果上报topic                                        | "gala_cause_inference"       |
| meta_topic  | gopher采集指标数据topic                                      | "gala_gopher_metadata"       |
| group_id    | kafka设置组名                                                | "gala_anteater_kafka"        |
| Prometheus  | 数据源prometheus配置信息                                     | 字典类型                     |
| server      | Prometheus Server的ip地址，根据安装节点ip配置                | "localhost"                  |
| port        | Prometheus Server的port，如：9090                            | "9090"                       |
| steps       | 指标采样间隔                                                 | ”5“                          |
| Schedule    | 循环调度配置信息                                             | 字典类型                     |
| duration    | 异常检测模型执行频率（单位：分），每x分钟，检测一次          | 1                            |
| Suppression | 告警抑制配置信息                                             | 字典类型                     |
| interval    | 告警抑制间隔(单位: 分)，表示距离上一次告警x分钟内相同告警过滤 | 10                           |

## 启动

执行如下命令启动gala-anteater

```
systemctl start gala-anteater
```

>![](./figures/icon-note.gif)**说明：**
>
>gala-anteater支持启动一个进程实例，启动多个会导致内存占用过大，日志混乱。

### 查询gala-anteater服务慢节点检测执行状态

若日志显示如下内容，说明慢节点正常运行，启动日志也会保存到当前运行目录下`logs/anteater.log`文件中。

```log
2024-12-02 16:25:20,727 - INFO - anteater - Groups-0, metric: npu_chip_info_hbm_used_memory, start detection.
2024-12-02 16:25:20,735 - INFO - anteater - Metric-npu_chip_info_hbm_used_memory single group has data 8. ranks: [0, 1, 2, 3, 4, 5, 6, 7]
2024-12-02 16:25:20,739 - INFO - anteater - work on npu_chip_info_hbm_used_memory, slow_node_detection start.
2024-12-02 16:25:21,128 - INFO - anteater - time_node_compare result: [].
2024-12-02 16:25:21,137 - INFO - anteater - dnscan labels: [-1  0  0  0 -1  0 -1 -1]
2024-12-02 16:25:21,139 - INFO - anteater - dnscan labels: [-1  0  0  0 -1  0 -1 -1]
2024-12-02 16:25:21,141 - INFO - anteater - dnscan labels: [-1  0  0  0 -1  0 -1 -1]
2024-12-02 16:25:21,142 - INFO - anteater - space_nodes_compare result: [].
2024-12-02 16:25:21,142 - INFO - anteater - Time and space aggregated result: [].
2024-12-02 16:25:21,144 - INFO - anteater - work on npu_chip_info_hbm_used_memory, slow_node_detection end.

2024-12-02 16:25:21,144 - INFO - anteater - Groups-0, metric: npu_chip_info_aicore_current_freq, start detection.
2024-12-02 16:25:21,153 - INFO - anteater - Metric-npu_chip_info_aicore_current_freq single group has data 8. ranks: [0, 1, 2, 3, 4, 5, 6, 7]
2024-12-02 16:25:21,157 - INFO - anteater - work on npu_chip_info_aicore_current_freq, slow_node_detection start.
2024-12-02 16:25:21,584 - INFO - anteater - time_node_compare result: [].
2024-12-02 16:25:21,592 - INFO - anteater - dnscan labels: [0 0 0 0 0 0 0 0]
2024-12-02 16:25:21,594 - INFO - anteater - dnscan labels: [0 0 0 0 0 0 0 0]
2024-12-02 16:25:21,597 - INFO - anteater - dnscan labels: [0 0 0 0 0 0 0 0]
2024-12-02 16:25:21,598 - INFO - anteater - space_nodes_compare result: [].
2024-12-02 16:25:21,598 - INFO - anteater - Time and space aggregated result: [].
2024-12-02 16:25:21,598 - INFO - anteater - work on npu_chip_info_aicore_current_freq, slow_node_detection end.

2024-12-02 16:25:21,598 - INFO - anteater - Groups-0, metric: npu_chip_roce_tx_err_pkt_num, start detection.
2024-12-02 16:25:21,607 - INFO - anteater - Metric-npu_chip_roce_tx_err_pkt_num single group has data 8. ranks: [0, 1, 2, 3, 4, 5, 6, 7]
2024-12-02 16:25:21,611 - INFO - anteater - work on npu_chip_roce_tx_err_pkt_num, slow_node_detection start.
2024-12-02 16:25:22,040 - INFO - anteater - time_node_compare result: [].
2024-12-02 16:25:22,040 - INFO - anteater - Skip space nodes compare.
2024-12-02 16:25:22,040 - INFO - anteater - Time and space aggregated result: [].
2024-12-02 16:25:22,040 - INFO - anteater - work on npu_chip_roce_tx_err_pkt_num, slow_node_detection end.

2024-12-02 16:25:22,041 - INFO - anteater - accomplishment: 1/9
2024-12-02 16:25:22,041 - INFO - anteater - accomplishment: 2/9
2024-12-02 16:25:22,041 - INFO - anteater - accomplishment: 3/9
2024-12-02 16:25:22,041 - INFO - anteater - accomplishment: 4/9
2024-12-02 16:25:22,042 - INFO - anteater - accomplishment: 5/9
2024-12-02 16:25:22,042 - INFO - anteater - accomplishment: 6/9
2024-12-02 16:25:22,042 - INFO - anteater - accomplishment: 7/9
2024-12-02 16:25:22,042 - INFO - anteater - accomplishment: 8/9
2024-12-02 16:25:22,042 - INFO - anteater - accomplishment: 9/9
2024-12-02 16:25:22,043 - INFO - anteater - SlowNodeDetector._execute costs 1.83 seconds!
2024-12-02 16:25:22,043 - INFO - anteater - END!
```

## 异常检测输出数据

gala-anteater如果检测到异常点，会将结果输出至kafka的model_topic，输出数据格式如下：

```json
{
    "Timestamp": 1730732076935, 
    "Attributes": {
        "resultCode": 201, 
        "compute": false, 
        "network": false, 
        "storage": true, 
        "abnormalDetail": [{
            "objectId": "-1", 
            "serverIp": "96.13.19.31", 
            "deviceInfo": "96.13.19.31:8888*-1", 
            "kpiId": "gala_gopher_disk_wspeed_kB", 
            "methodType": "TIME", 
            "kpiData": [], 
            "relaIds": [], 
            "omittedDevices": []
        }], 
        "normalDetail": [], 
        "errorMsg": ""
    }, 
    "SeverityText": "WARN", 
    "SeverityNumber": 13, 
    "is_anomaly": true
}
```

## 输出字段说明

| 输出字段       | 单位   | 含义                                                  |
| -------------- | ------ | ----------------------------------------------------- |
| Timestamp      | ms     | 检测到故障上报的时刻                                  |
| resultCode     | int    | 故障码，201表示故障，200表示无故障                    |
| compute        | bool   | 故障类型是否为计算类型                                |
| network        | bool   | 故障类型是否为网络类型                                |
| storage        | bool   | 故障类型是否为存储类型                                |
| abnormalDetail | list   | 表示故障的细节                                        |
| objectId       | int    | 故障对象id，-1表示节点故障，0-7表示具体的故障卡号     |
| serverIp       | string | 故障对象ip                                            |
| deviceInfo     | string | 详细的故障信息                                        |
| kpiId          | string | 检测到故障的算法类型，“TIME", "SPACE"                 |
| kpiData        | list   | 故障时序数据，需开关打开，默认关闭                    |
| relaIds        | list   | 故障卡关联的正常卡，表示在”SPACE“算法下对比的正常卡号 |
| omittedDevices | list   | 忽略显示的卡号                                        |
| normalDetail   | list   | 正常卡的时序数据                                      |
| errorMsg       | string | 错误信息                                              |
| SeverityText   | string | 错误类型，表示”WARN“, "ERROR"                         |
| SeverityNumber | int    | 错误等级                                              |
| is_anomaly     | bool   | 表示是否故障                                          |