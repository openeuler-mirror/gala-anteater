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
使用构建的Docker镜像创建三个容器，并让三个容器处于同一个资源组限制下（后续需判断三个容器运行是否继承资源组的限制，如不满足可能需要修改Docker Cgroup Driver为systemd）：  
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

## 查看gala-anteater检测的异常信息
```
# 先跳转到kafka的安装路径
./bin/kafka-console-consumer.sh --bootstrap-server [ip]:9092 --topic gala_anteater_hybrid_model --from-beginning
```