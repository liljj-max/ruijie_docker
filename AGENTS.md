# AGENTS.md — 本仓库协作约定

## 版本控制
- 每次做出**实质性、可用的改动**后，及时 `git commit` 并 `git push` 到
  `https://github.com/liljj-max/ruijie_docker.git`（分支 `main`）。
- 提交信息用中文简述“做了什么 + 为什么”，一次提交聚焦一件事。
- 不要把大文件/镜像 tar/预训练下载提交进来（见 `.gitignore`）。

## 代码约定
- 参赛代码位于 `competition_client/`，在官方 `supermarket_sorting:client` 镜像内运行，
  挂载到 `/workspace/baseline`。
- 统一入口：`scripts/run_competition.sh`。
- 运行环境：`ROS_DOMAIN_ID=99`、`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`。
- 安全红线：任何异常/超时都必须发零速；机械臂未收回禁止动底盘；一次只处理一件。

## 已知约束
- Client 镜像 ultralytics 8.0.196 **不支持 YOLO-World**，9 类识别用自训 YOLOv8s
  (`weights/products9.pt`)。
- 仿真底盘对 `/cmd_vel` 的实际响应约为指令的 1/8，`mmk2_adapter.py` 用发布增益补偿。
- 货位关联仍需提高准确性（当前几何关联存在误关联）。
