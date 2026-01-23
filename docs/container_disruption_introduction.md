# 容器干扰检测诊断
## 创建systemd slice（资源组）
后续运行的测试容器需要跑在该资源组下运行，限制资源2个cpu核  
```
sudo tee /etc/systemd/system/stress.slice <<EOF
[Unit]
Description=Stress Containers Resource Slice
Before=slices.target

[Slice]
CPUAccounting=yes
MemoryAccounting=yes
CPUQuota=200%        # 相当于 2.0 个 CPU 核
MemoryMax=2G         # 最大内存 2GB
EOF
```
重新加载systemd
```
systemctl daemon-reload
systemctl set-property stress.slice CPUQuota=200%
```
## 构造测试容器
Dockerfile如下：  
```
FROM ubuntu:20.04

# 更新包列表并安装 stress-ng
RUN apt-get update && \
    apt-get install -y stress-ng && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```
将此Dockerfile放到当前目录下，使用此Dockerfile构建镜像：
```
docker build -t stress-container .
```
使用构建的Docker镜像创建四个容器，并让四个容器处于同一个资源组限制下（后续需判断四个容器运行是否继承资源组的限制，如不满足可能需要修改Docker Cgroup Driver为systemd）：  
```
docker run -d --name stress-container1 \
  --cgroup-parent=stress.slice \
  stress-container \
  stress-ng --cpu 1 --cpu-load 40 -t 0

docker run -d --name stress-container2 \
  --cgroup-parent=stress.slice \
  stress-container \
  stress-ng --cpu 1 --cpu-load 40 -t 0

docker run -d --name stress-container3 \
  --cgroup-parent=stress.slice \
  stress-container \
  stress-ng --cpu 1 --cpu-load 40 -t 0

docker run -d --name stress-container4 \
  --cgroup-parent=stress.slice \
  stress-container \
  stress-ng --cpu 1 --cpu-load 40 -t 0
```
正常运行情况下，四个容器占用cpu资源为4*0.4=1.6，小于资源组的2个cpu核限制，互相之间不存在干扰，后续会往其中一个容器中加更多的负载，从而超出资源组的限制，此时会触发容器干扰检测，并返回异常信息

## gala-gopher 容器干扰探针数据采集
探针示例, container_id需要根据实际容器id填入：  
```
curl -X PUT http://localhost:9999/sli \
-d 'json={
    "cmd": {
        "probe": [
            "cpu",
            "mem",
            "io",
            "node",
            "container",
            "histogram"
        ]
    },
    "snoopers": {
        "container_id": [
            "xxxxxxxxxxxx",
            "xxxxxxxxxxxx",
            "xxxxxxxxxxxx",
            "xxxxxxxxxxxx"
        ]
    },
    "params": {
        "report_period": 5
    },
    "state": "running"
}'
```

## 向指定容器中模拟注入故障
```
docker exec -it stress-container1 bash
stress-ng --cpu 2 --cpu-load 100 -t 300
```

## 异常上报
```
# 先跳转到kafka的安装路径
./bin/kafka-console-consumer.sh --bootstrap-server [ip]:9092 --topic gala_anteater_hybrid_model --from-beginning
```

## 异常检测结果示例
```
{"Timestamp": 1767864105381, "Attributes": {"entity_id": "fc214d56-657e-2c2d-7b3e-ace4653752d0-192.168.140.132_sli_0_0_0_0", "event_id": "1767864105381_fc214d56-657e-2c2d-7b3e-ace4653752d0-192.168.140.132_sli_0_0_0_0", "event_type": "simple", "event_source": "spot", "keywords": ["app"]}, "Resource": {"metric": "gala_gopher_sli_container_cpu_rundelay", "labels": {"ContainerID": "e9ca900cbc2d", "ContainerName": "stress-container"}, "score": "0.800", "root_causes": [{"metric": "gala_gopher_sli_container_cpu_rundelay", "labels": {"container_id": "aff3bc28a7ee", "container_image": "stress-container", "container_name": "/stress-container4", "instance": "192.168.140.132:8888", "job": "gala-gopher", "machine_id": "fc214d56-657e-2c2d-7b3e-ace4653752d0-192.168.140.132", "cpu_num": 0}, "score": "0.964"}, {"metric": "gala_gopher_sli_container_cpu_rundelay", "labels": {"container_id": "2cc2b8cc678b", "container_image": "stress-container", "container_name": "/stress-container1", "instance": "192.168.140.132:8888", "job": "gala-gopher", "machine_id": "fc214d56-657e-2c2d-7b3e-ace4653752d0-192.168.140.132", "cpu_num": 0}, "score": "0.811"}], "description": "cpu \u8c03\u5ea6\u65f6\u5ef6"}, "SeverityText": "WARN", "SeverityNumber": 13, "Body": "2026-01-08 17:21:45 - app metric anomaly - cpu \u8c03\u5ea6\u65f6\u5ef6 - {'event_source': 'spot', 'info': {'abnormal_start': 1767864090, 'abnormal_end': 1767864105, 'container_name': '/stress-container2', 'machine_id': 'fc214d56-657e-2c2d-7b3e-ace4653752d0-192.168.140.132', 'gala_gopher_sli_container_cpu_busy': 0.0, 'appkey': '', 'cpu_num': 0}}", "is_anomaly": true}
```