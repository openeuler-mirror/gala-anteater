# gala-anteater 支持的故障诊断场景

## 测试准则

* 场景：Redis
* 打流：redis-benchmark -h xxx.xxx.xxx.xxx -t set -p 
* 当前 anteater 主要基于历史数据进行无监督训练，然后进行异常检测；
* 训练数据集应保证**无故障数据**，数据量不小于 14h；
* 推理阶段，按分钟级别进行推理，最小推理时间 1 分钟；
* 针对不同的场景，构建不同的模型，如应用级检测主要为“深度神经网络模型”、系统级为 “Rule-based” + “n-sigma” 模型，JVM OOM 主要为：Tree-based 模型。

## anteater 支持的故障诊断场景汇总
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
| JVM OutOfMemory | Heapspace                          | gala_gopher_jvm_mem_bytes_used<br/>gala_gopher_jvm_mem_pool_bytes_used                                                                                                                   | java code: JavaOOMHttpServer，参考：[link](jvm_oom_introduction.md)。 |
|                 | GC Overhead                        | gala_gopher_jvm_mem_bytes_used<br/>gala_gopher_jvm_mem_pool_bytes_used                                                                                                                   | java code: JavaOOMHttpServer，参考：[link](jvm_oom_introduction.md)。 |
|                 | Metaspace                          | gala_gopher_jvm_class_current_loaded                                                                                                                                                     | java code: JavaOOMHttpServer，参考：[link](jvm_oom_introduction.md)。 |
|                 | Unable to create new native thread | gala_gopher_jvm_threads_current                                                                                                                                                          | java code: JavaOOMHttpServer，参考：[link](jvm_oom_introduction.md)。 |
|                 | Direct buffer memory               | gala_gopher_jvm_buffer_pool_used_bytes                                                                                                                                                   | java code: JavaOOMHttpServer，参考：[link](jvm_oom_introduction.md)。 |