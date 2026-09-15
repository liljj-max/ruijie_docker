# 智慧零售赛题 · 会话交接文档（Handoff）

> 用途：新开会话时，把本文件发给助手即可快速衔接。最后更新：2026-09-15。

## 0. 一句话现状
已在官方双镜像上跑通“扫描→抓取→配送→放置”闭环；当前把**货架区/取货区改为写死原语**（去规划器、去避障/脱困），**仅动态障碍区用 A*/DWB**；正修“首轮先跑右二货架”和“ALIGN 卡死”。

## 1. 机器与环境（已部署）
- 主机：x86_64，i9-14900HX / 32GB / **RTX 4060 8GB** / Ubuntu 22.04；磁盘可用 ~358G；swap 已扩到 **16G**。
- Docker CE **29.6.2**（华为云源）；`nvidia-container-toolkit` **1.20.0**；`daemon.json` 已配 nvidia runtime。
- 官方镜像（摘要与 PDF 一致）：
  - `supermarket_sorting:server` = `sha256:eb0b58a600b85910c2e5392852e9268ca6a5aaca02385ab8a0f69981d833c4b2`
  - `supermarket_sorting:client` = `sha256:dbe0bfd2c75e34af430e2e79f07c0ab9528d843e6fc6b8ce067216fd28168f0e`
- 镜像 tar：`/home/makabaka/ruijie_docker/*.tar`；官方参考导出：`/home/makabaka/ruijie_docker/official_reference/`

## 2. 代码仓库
- 本地：`/home/makabaka/ruijie_code`；远端：`https://github.com/liljj-max/ruijie_docker`（`main`，gh 已登录）。
- 最新提交 `e3ccb4a`；**有一批改动未提交**（见第 5 节），下次先 `git status`。
- 官方 Baseline（MMK2 FK/IK）：`/home/makabaka/supermarket_sorting_baseline`（`cyathea152-bit/supermarket_sorting_baseline`）。
- 约定见 `AGENTS.md`：每次实质改动后 commit + push。

## 3. 官方接口/规则关键事实
### 话题
- 收：`/supermarket_sorting/task`(String, **RELIABLE+TRANSIENT_LOCAL+depth1**)、`/slamware_ros_sdk_server_node/scan`(360点/12Hz/frame=laser/0.02–12m)、`/slamware_ros_sdk_server_node/odom`、`/tf`、`/joint_states`、头部/左右腕 RGB、头部对齐深度(**毫米**)。
- 发：`/cmd_vel`(仅 linear.x/angular.z)、`/spine_|head_|left_arm_|right_arm_forward_position_controller/commands`。
### 任务消息（每局一次）
```json
{"schema_version":1,"run_prefix":"run_xxx","count":5,"targets":[{"id":"item_run_xxx_01","kind":"kele"}, ...]}
```
只有匿名 `id`+`kind`，**无位置**；新 `run_prefix` 需清空库存与进度。
### 关节顺序
`slide_joint, head_yaw_joint, head_pitch_joint, left_arm_joint1..6, left_arm_eef_gripper_joint, right_arm_joint1..6, right_arm_eef_gripper_joint`（夹爪 1.0 开 / 0.08 闭）。
### 限位（实测）
`head_yaw ∈ [-0.5,0.5]`、`head_pitch ∈ [-1.18,0.16]`、`slide ∈ [-0.04,0.87]`。
### 底盘
- **两轮差速**：轮半径 0.0838、轮距参数 0.189；`v_l=(v-ωL)/r, v_r=(v+ωL)/r`。
- **实测响应约指令的 1/8 且有滞后/饱和**；`/odom` 是**仿真真值** → 闭环准、开环偏。
### 机械臂外伸（初始姿态碰撞网格）
侧向 **±0.435**、前向 **+0.369**、z 到 ~1.6（肘部 z≈1.2）。配送台面 ~0.77m（低于手臂）；货架/隔板/墙 ~1.3–1.5m。
### 随机障碍投放范围（世界系）
`x∈[-2.47,0.50]`、`y∈[-3.72,1.70]`（走廊）。观察线 `y=2.40` 在其北、起步走廊 `x≈1.8` 在其东 → **写死路径安全**。
### 评分（官方 V2.0）
任务完成分 80（交付 60=12/件×5 + 效率 20）+ 技术评价 20（含 ArUco 利用）；另有扣除分。**自动裁判只看仿真物理结果**；ArUco 仅人工技术分。本地可用镜像内 `referee.py` 自测。

## 4. 代码架构（`competition_client/`）
| 文件 | 职责 |
| --- | --- |
| `task_parser.py` | 订阅 task（TRANSIENT_LOCAL），解析 run_prefix/count/targets |
| `mmk2_adapter.py` | 19 维控制封装、发布 5 控制话题、odom/关节回读、平滑限幅；`MAX_LIN=0.08, MAX_ANG=0.18, BASE_GAIN_LIN=8, BASE_GAIN_ANG=4` |
| `perception.py` | 9 类 YOLO + 深度→世界(MMK2FK) + ArUco + 近区过滤（`fwd 0.15–1.40, |lat|≤0.35, z 0.30–1.50`），发布 `/competition/product_detections`、`/competition/aruco_detections`、`/competition/result_image` |
| `detector.py` | 9 类检测后端，权重 `weights/products9.pt`（mAP50≈0.99，22MB） |
| `shelf_scanner.py` | 多帧库存 kind→候选货位，支持 reserve/release/consume |
| `planner.py` | 全局栅格 + 实时 TF 外参的 360°雷达射线建图 + **按高度分档膨胀(高 0.45/低 0.35)** + 代价加权 A* |
| `local_planner.py` | DWB（v,ω 采样 + 弧长推进/切向对齐/离障/只前进/防抖，`safety=0.05`） |
| `competition_client.py` | 主状态机 + 写死原语 + 目标选择 |
| `scripts/run_competition.sh` | 入口：perception + competition_client |
| `tools/` | `gen_dataset_9cls.py` / `train_products9.py` |

### 状态机
`WAIT_TASK, STOW, SCAN, ALIGN, DEPLOY, WAIT_ARM, CREEP, BRAKE, CLOSE, LIFT, RETREAT, RETURN, NAV_TABLE, PLACE, NEXT, NAV_RETURN, DONE, ERROR`
- **SCAN**：只访问未完成货架；头部实测稳定后执行 `yaw[-0.3,0,0.3]×pitch[-0.3,-0.7,-1.1]` 网格。仅稳定扫描窗口写库存，发现任一 pending kind 后立即预留并抓取；中断货架不会记完成。
- **ALIGN**：两段——先 `_drive_to(lane)` 到位，再 `_turn_to(GRASP_YAW)` 对准 + 0.3s 稳定 → DEPLOY。
- **DEPLOY/CREEP/CLOSE/LIFT/RETREAT**：官方抓取流程。
- **RETURN**：取到后**写死**折线到 `OBSTACLE_ENTRY=(-0.50,2.475)`。
- **NAV_TABLE/NAV_RETURN**：随机障碍走廊双向使用实时 LaserScan + A* + DWB；货架区、出发区和抓取退让继续使用 odom 闭环固定原语。
- **PLACE**：降 slide + 开爪。
- **NEXT**：成功货位标记 consumed；优先复用已扫描库存，没有目标才返回继续扫描未知货架。

### 关键常量
`YELLOW_MID_Y=2.475`、`GRASP_YAW=π/2−11°`、`APPROACH_DX=0.068`、`SHELF_X={A:-1.735,B:-0.850,C:0.035,D:0.920,E:1.805}`、`COLUMN_DX={C1:-0.22,C2:0,C3:0.22}`、`SCAN_Y=2.40`、`EXPLORE_SPEED=0.06`、`EXPLORE_TURN_MAX=0.15`、`SCAN_SLIDE=0.11`、`HEAD_PITCH=-0.6`、`SLIDE_GRASP=0.11`、`RETREAT_SPEED=0.12`、`PLACE_LOWER_SLIDE=0.17`、`OBSTACLE_ENTRY=(-0.50,2.475)`、`TABLE_APPROACH=(-1.88,-2.80)`。

## 5. 当前实现
1. 任务 ID 只保留为订单身份，不解析后缀，也不推导货架/层/列。
2. YOLO 决定商品 kind；ArUco 只映射固定货位。同一 RGB 帧完成关联，并拒绝几何不一致、过远或歧义 marker。
3. 库存按货位融合多帧类别和世界坐标，抓取前 reserve，失败 release，配送成功 consume。
4. LaserScan 超过 0.5 秒、odom/JointState 超过 0.75 秒或 scan 找不到 0.1 秒内对应 odom 时立即停车。
5. 规划路径被新障碍占用时立即重规划；删除未经验证的直接倒车脱困。

## 6. 已知限制
1. 官方裁判按匿名 body ID 计分，但相机只提供商品类别、ArUco 只提供货位。若任务仅指定多个同类商品中的某一个实体，在不通过 ID 推导随机货位的约束下不可观测；默认全场任务或按类别选择全部该类商品不受影响。
2. 固定场景联调确认移动/转向期间不再污染库存；尚未完成一次随机五订单端到端回归。
3. 当前固定场景未看到 ArUco 检测，需继续验证 marker 可见角度和尺寸参数。
4. IK 和机械臂到位时间仍需在完整抓取循环中调参。

## 7. 下一步计划
- 随机场景 `SEED=11 TASKS=all` 验证双向走廊 A*/DWB、动态重规划和断流停车。
- 完成至少一次“扫描中断→抓取配送→复用库存/续扫原货架”的多订单回归。
- 验证 ArUco 可见率后再调整抓取微调和机械臂分组速度。

## 8. 常用命令
### 启动 Server（固定 Baseline）
```bash
sudo docker rm -f supermarket_sorting_server
sudo docker run -d --gpus all --network host --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=:1 -e ROS_DOMAIN_ID=99 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw -e SUPERMARKET_HEADLESS=0 -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 -e SUPERMARKET_USE_GS=1 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e SUPERMARKET_FIXED_BASELINE=1 -e SUPERMARKET_RANDOMIZE=0 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=0 -e SUPERMARKET_TASKS=product_032 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v supermarket_sorting_cache:/root/.cache \
  supermarket_sorting:server \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```
> 随机场景：把 `RANDOMIZE=1 RANDOMIZE_OBSTACLES=1 TASKS=all`（可加 `SEED=11`）。
### 启动 Client
```bash
sudo docker rm -f supermarket_sorting_client
sudo docker run -dit --gpus all --network host --ipc host \
  --name supermarket_sorting_client \
  -e ROS_DOMAIN_ID=99 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /home/makabaka/ruijie_code:/workspace/baseline:rw \
  supermarket_sorting:client
```
### 运行 / 看日志 / 停车
```bash
# 运行
sudo docker exec -d supermarket_sorting_client bash -lc \
 'source /opt/ros/humble/setup.bash && cd /workspace/baseline && ROS_DOMAIN_ID=99 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp bash scripts/run_competition.sh > /tmp/comp.log 2>&1'
# 看
sudo docker exec supermarket_sorting_client tail -50 /tmp/comp.log
sudo docker exec supermarket_sorting_client bash -lc "grep -aoE 'phase=[a-z_>-]+' /tmp/comp.log | uniq -c"
sudo docker exec supermarket_sorting_client bash -lc "grep -aE 'locked |stuck|timeout|IK failed|safety buffer' /tmp/comp.log | tail"
# 停车（测试前务必杀干净，避免多实例抢 /cmd_vel）
sudo docker exec supermarket_sorting_client bash -lc "pkill -9 -f '[c]ompetition_client'; pkill -9 -f '[r]un_competition'"
# 语法自检
sudo docker exec supermarket_sorting_client bash -lc 'source /opt/ros/humble/setup.bash && cd /workspace/baseline && python3 -c "import competition_client.competition_client; print(\"IMPORT_OK\")"'
```

## 9. 关键路径
- 仓库：`/home/makabaka/ruijie_code`；官方 Baseline：`/home/makabaka/supermarket_sorting_baseline`
- 镜像/参考：`/home/makabaka/ruijie_docker/`；数据集：`/home/makabaka/competition_dataset/`
- 资料：`/home/makabaka/文档/xwechat_files/wxid_ipka03tcmvxu22_e3be/msg/file/2026-09/`

## 10. 未决问题
1. 主办方是否保证任务按 kind 覆盖该类全部实体；否则匿名 body 子集任务无法仅靠现有视觉区分。
2. 提交形式是否允许自建镜像，还是严格官方 Client 仅挂载代码。
