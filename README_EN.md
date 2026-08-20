# gala-anteater

## Introduction

gala-anteater is an AI-based exception detection platform for OS gray faults. It integrates multiple exception detection algorithms to detect system-level faults and report fault points in real time for different scenarios and applications.

Based on historical system data, gala-anteater performs automatic model pre-training, incremental learning of online models, and model update. It can adapt to multi-scenario and multi-metric data and implement minute-level model inference.

## Supported Exception Detection Scenarios

Currently, gala-anteater supports exception detection in 13 sub-scenarios of three fault categories.

| Category             | Diagnosis Scenario                              | KPI                                                                                                                                                                                     | Fault Injection Mode                                                          |
|-----------------|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| Application level            | Application delay (RTT)                         | gala_gopher_sli_rtt_nsec                                                                                                                                                                 | chaosblade: network loss/delay, disk fill/burn, cpu              |
|                 | Application throughput (TPS)                        | gala_gopher_sli_tps                                                                                                                                                                      | chaosblade: network loss/delay, disk fill/burn, cpu              |
| System level            | TCP connection setup performance                           | gala_gopher_tcp_link_syn_srtt                                                                                                                                                            | chaosblade: network delay                                        |
|                 | TCP transmission performance                           | gala_gopher_tcp_link_srtt                                                                                                                                                                | chaosblade: network loss                                         |
|                 | System I/O performance                           | gala_gopher_block_latency_req_max                                                                                                                                                        | chaosblade: disk burn                                            |
|                 | Process I/O performance                           | gala_gopher_proc_bio_latency<br>gala_gopher_proc_less_4k_io_read<br>gala_gopher_proc_less_4k_io_write<br>gala_gopher_proc_greater_4k_io_read<br>gala_gopher_proc_greater_4k_io_write | chaosblade: disk burn                                            |
|                 | Drive throughput                             | gala_gopher_disk_r_await<br>gala_gopher_disk_w_await                                                                                                                                    | chaosblade: disk full                                            |
|                 | NIC TX packet loss                            | gala_gopher_nic_tc_sent_drop                                                                                                                                                             | chaosblade: network loss                                         |
| [JVM OutOfMemory](docs/jvm_oom_introduction.md) | Heapspace                          | gala_gopher_jvm_mem_bytes_used<br>gala_gopher_jvm_mem_pool_bytes_used                                                                                                                   | java code: JavaOOMHttpServer |
|                 | GC Overhead                        | gala_gopher_jvm_mem_bytes_used<br>gala_gopher_jvm_mem_pool_bytes_used                                                                                                                   | java code: JavaOOMHttpServer |
|                 | Metaspace                          | gala_gopher_jvm_class_current_loaded                                                                                                                                                     | java code: JavaOOMHttpServer |
|                 | Unable to create new native thread | gala_gopher_jvm_threads_current                                                                                                                                                          | java code: JavaOOMHttpServer |
|                 | Direct buffer memory               | gala_gopher_jvm_buffer_pool_used_bytes                                                                                                                                                   | java code: JavaOOMHttpServer |

## Installation and Deployment

### Prerequisites

* Supported Python version: 3.7+
* gala-anteater depends on the data collected by gala-gopher. Install and deploy gala-gopher first.
* gala-anteater directly obtains time series metric data from Prometheus. Prometheus needs to be installed and deployed.
* gala-anteater depends on the meta data reported by gala-gopher (to Kafka). Therefore, ensure that Kafka has been installed and deployed.

### Method 1: Installing Using a Docker Image (for Common Users)

#### Creating a Docker Image

Run the following command in the `./gala-anteater` directory of the project to pack the `gala-anteater` project file into a Docker image:

```bash
docker build -f Dockerfile -t gala-anteater:1.1.0 .
```

Note: You may need to change the `pip` source address in the `Dockfile` file based on the network conditions.

#### Running the Docker Image

Run the following command to run the Docker image: When the Docker image is run for the first time, the configuration file `gala-anteater.yaml` is mapped to the `/etc/gala-anteater/config` file on the host machine.
Configure the parameters in the `gala-anteater.yaml` file. For details about the configuration method, see [Configuration File Introduction](https://atomgit.com/openeuler/gala-anteater/blob/master/docs/conf_introduction.md).

```bash
docker run -v /etc/gala-anteater:/etc/gala-anteater -it gala-anteater:1.1.0
```

### Method 2: Installing and Running from the Source Code in This Repository (for Developers)

#### Downloading the Source Code

```bash
 git clone https://atomgit.com/openeuler/gala-anteater.git
```

#### Installation

Run the following command in the `./gala-anteater` project directory:

```bash
python3 setup.py install
```

#### Parameter Configuration

The configuration parameters will be mapped to the `/etc/gala-anteater/config` file. You need to set the corresponding parameters first. For details about the configuration method, see [Configuration File Introduction](https://atomgit.com/openeuler/gala-anteater/blob/master/docs/conf_introduction.md).

Note: In the configuration file, the most important thing is to configure the middleware in the configuration file, such as `Kafka server/port` and `Prometheus server/port`.

#### Execution

```bash
systemctl start gala-anteater
```

### Log

The default log file path is `/var/gala-anteater/logs/`. You can also change the log file path in the `log.settings.ini` configuration file.

### Exception Reporting

gala-anteater outputs the exception detection result to `Kafka`. If an exception is detected, the detection result is output to `Kafka`. The default `Topic` is `gala_anteater_hybrid_model`. You can also modify the configuration in `gala-anteater.yaml`. Run the following command to view the exception detection result:

```bash
./bin/kafka-console-consumer.sh --topic gala_anteater_hybrid_model --from-beginning --bootstrap-server localhost:9092
```

## Exception Detection Result

By default, the exception detection result is output to `Kafka`. It can also be stored in `ArangoDB` for third-party O&M systems to query and integrate. The data format complies with the `OpenTelemetry V1` specifications. For details, see [Kafka to ArangoDB](docs/kafka_to_arangodb.md). The following describes the output format of exception detection.

### Output Data

#### Output Data Format

| Parameter|  Meaning | Description|
|:---:|:------:|---|
| Timestamp|  Timestamp  | Timestamp when an exception event is reported.|
| Attributes|  Attribute value  | It mainly includes:<br>1. **entity_id**: naming rule, \<machine_id\>_\<table_name\>_\<keys\><br>2. **entity_id**: event ID, \<timestamp\>_\<entity_id\><br>3. **event_type**: main event types, APP/SYS/JVM<br>4. **event_source**: event source<br>5. **keywords(optional)**: event keywords, which are used for quick search|
| Resource |   Resource  | The output information of the exception detection model mainly includes:<br>1. **metric**: main metric for exception detection<br>2. **labels**: exception metric label information (for example, Host/PID/COMM/IP)<br>3. **score**: exception score of an event<br>4. **root_causes (optional)**: recommended top *N* root causes<br> |
| SeverityText | Exception event type| INFO, WARN, ERROR, FATAL |
| SeverityNumber | Exception event number| 9, 13, 178, 21...|
| Body | Exception event information| Description of the current exception event (string type)<br>Format: \<timestamp\> - \<header\> - \<description\> - \<details\>|

#### Output Data Example

Example 1:

```json
{
    "Timestamp": 1669343170074,
    "Attributes": {
        "entity_id": "7c2fbaf8-xxx-xxx-xxx-xxx_sli_xxx_16859_POSTGRE_0",
        "event_id": "1669343170074_7c2fbaf8-xxx-xxx-xxx-xxx_sli_2187425_16859_POSTGRE_0",
        "event_type": "app",
        "event_source": "gala-anteater",
        "keywords": [
            "sli",
            "tcp"
            ]
    },
    "Resource": {
        "metric": "gala_gopher_sli_tps",
        "labels": {
            "Host": "110f3138-xxx-xxx-xxx-xxxx-xxx",
 "PID": "1188486",
 "COMM": "xxx-server",
 "IP": "xx.xxx.xxx.xxx"
        },
        "score":0.36,
        "root_causes": [
            {
                "metric": "gala_gopher_net_tcp_retrans_segs",
                "labels": {
                    "instance": "xxx.xxx.xxx.xxx:x",
                    "job": "prometheus-xxx.xxx.xxx.xxx:x",
                    "machine_id": "7c2fbaf8-xxx-xxx-xxx-xxx",
                    "origin": "/proc/dev/snmp"
                },
                "score": 16.9
            },
            {
                "metric": "gala_gopher_cpu_user_total_second",
                "labels": {
                    "cpu": "6",
                    "instance": "10.xxx.xxx.xxx:18001",
                    "job": "prometheus-10.xxx.xxx.xxx:8001",
                    "machine_id": "7c2fbaf8-xxx-xxx-xxx-xxx"
                },
                "score": 6.1
            }
        ]
    },
    "SeverityText": "WARN",
    "SeverityNumber": 13,
    "Body": "2023-xx-xx xx:xx:xx - System Failure - xxx protocol request RTT"
}
```

Example 2:

```json
{
    "Timestamp": 1693385669409,
    "Attributes": {
        "entity_id": "110f3138-xxx-xxx-xxx-xxx-xxx.xxx.xxx.xxx_jvm_xxx",
        "event_id": "1693385669409_110f3138-xxx-xxx-xxx-xxx_jvm_xxx",
        "event_type": "jvm",
        "event_source": "gala-anteater",
        "keywords": [
            "jvm"
            ]
    },
    "Resource": {
        "metric": "gala_gopher_jvm_mem_pool_bytes_used","labels": {
            "PID": "xxx",
            "COMM": "java"
        },
        "score": 0.1,
        "root_causes": []
    },
    "SeverityText": "WARN",
    "SeverityNumber": 13,
    "Body": "2023-08-30 08:54:29 - JVM OutOfMemory - Number of used bytes in the specified JVM memory pool - {'PS Old Gen Usage': 0.99}"
}
```

## Project Roadmap

### Exception Detection Capabilities

| Feature                                             | Release Time| Release Version                            |
| ------------------------------------------------- | -------- | ------------------------------------ |
| Single-dimensional time series data exception detection (Redis/PostgreSQL application performance deterioration)      | 2022-12   | openEuler 22.03 SP1                  |
| Multi-dimensional time series data exception detection (TCP link setup, transmission, and system I/O)       | 2022-12   | openEuler 22.03 SP1                  |
| Multi-dimensional threshold exception detection (Java OOM exceptions)                | 2023-09   | openEuler 22.03 SP1, openEuler 23.09 |
| Improved exception detection accuracy (compressed sensing outlier filtering technology for training sets + multi-metric reconstruction technology + exception degree dynamic threshold technology) | 2023-09   | openEuler 22.03 SP1, openEuler 23.09 |
| Improved generalization capability of exception detection (adaptive technology for stable/unstable background flows)| 2023-09   | openEuler 22.03 SP1, openEuler 23.09 |
| Improved generalization capability of exception detection (online learning + incremental learning)     | 2023-09   | openEuler 22.03 SP1, openEuler 23.09 |
| Performance deterioration exception detection for whitelisted applications                       | 2024-03   | openEuler 24.03                      |

### Root Cause Locating Capabilities

| Feature                                                        | Release Time| Release Version                            |
| ------------------------------------------------------------ | -------- | ------------------------------------ |
| Root cause locating for application performance deterioration based on expert rules (network/disk I/O faults in virtualized storage and SDS scenarios)| 2022-12   | openEuler 22.03 SP1                  |
| Root cause locating based on causality diagram creation and causality propagation analysis (root cause propagation and derivation technology + root cause path tracing technology)  | 2023-09   | openEuler 22.03 SP1, openEuler 23.09 |
| Improved root cause locating accuracy (causality diagram of the PC algorithm + expert experience)               | 2024-03   | openEuler 24.03                      |
| General root cause locating for resource exceptions (graph-based multi-variable time series)            | 2024-03   | openEuler 24.03                      |
| Root cause locating for multi-modal application performance deterioration (based on metrics, logging, and tracing)  | 2024-09   | openEuler 24.09                      |
