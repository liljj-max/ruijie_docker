# 智慧零售赛题 · 参赛 Client（MMK2）

面向 DG-202606「面向智慧零售的自主服务机器人研发与应用」的参赛 Client 代码，
运行在官方 `supermarket_sorting:client` 镜像内，代码挂载到 `/workspace/baseline`。

## 目录结构

```
competition_client/      参赛主程序（本仓库核心）
  competition_client.py    主状态机：任务→扫描→导航→抓取→配送→循环
  task_parser.py           解析 /supermarket_sorting/task（TRANSIENT_LOCAL）
  mmk2_adapter.py          MMK2 底盘/头/升降/双臂/夹爪控制封装 + 安全
  perception.py            统一感知：9 类 YOLO + ArUco + 世界坐标
  detector.py              9 类检测后端（weights/products9.pt）
  shelf_scanner.py         货架库存 kind→货位 映射
  aruco_detect.py          官方 ArUco 检测器（DICT_4X4_50）
scripts/
  run_competition.sh       统一入口：启动感知 + 主程序
  task_probe.py            P1 冒烟测试（任务 + 控制）
  angtest.py               底盘速度响应标定
tools/
  gen_dataset_9cls.py      9 类数据集生成（3DGS 仿真内运行）
  train_products9.py       YOLOv8s 训练脚本
kinematics/                MMK2 FK/IK（来自官方 Baseline）
common/                    step_func 等工具
models/mmk2_head_fk.xml    头部相机 FK 模型
weights/products9.pt       9 类检测权重（mAP50≈0.99）
智慧零售_下一步执行规划.md   执行规划技术文档
```

## 运行（在官方 Client 容器内）

```bash
cd /workspace/baseline
ROS_DOMAIN_ID=99 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp bash scripts/run_competition.sh
```

## 环境要求

- 官方镜像 `supermarket_sorting:server` / `:client`（`ROS_DOMAIN_ID=99`、`rmw_cyclonedds_cpp`）
- Client 镜像自带 ultralytics 8.0.196（不支持 YOLO-World，故 9 类为自训 YOLOv8s）

## 9 类模型

由 `tools/gen_dataset_9cls.py`（3DGS 自动标注，1200 图/1700 框）生成数据集，
`tools/train_products9.py` 训练得到 `weights/products9.pt`（22MB）。

## 当前状态

- 已完成：任务解析、MMK2 控制、ArUco、9 类检测、统一感知、货架库存、主状态机骨架、超时恢复、限速与接近减速。
- 进行中：货位关联准确性（几何关联存在误关联）、DEPLOY 锁定失败容错。
- 详见 `智慧零售_下一步执行规划.md`。
