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
| `shelf_scanner.py` | 库存 kind→候选货位 |
| `planner.py` | 全局栅格 + 360°雷达射线建图(footprint 过滤) + **按高度分档膨胀(高 0.45/低 0.25)** + 代价加权 A* + `_clear_path` |
| `local_planner.py` | DWB（v,ω 采样 + 弧长推进/切向对齐/离障/只前进/防抖，`safety=0.05`） |
| `competition_client.py` | 主状态机 + 写死原语 + 目标选择 |
| `scripts/run_competition.sh` | 入口：perception + competition_client |
| `tools/` | `gen_dataset_9cls.py` / `train_products9.py` |

### 状态机
`WAIT_TASK, SCAN, ALIGN, NAV_SHELF, DEPLOY, CREEP, CLOSE, LIFT, RETREAT, RETURN, NAV_TABLE, PLACE, NEXT, DONE, ERROR`
- **SCAN**：写死原语 `_build_scan_plan`：`goto(1.805,2.40)→scan→[左转90→西行0.885→右转90→scan]×4`（E→D→C→B→A）；头部 `yaw[-0.3,0,0.3]×pitch[-0.3,-0.7,-1.1]` 网格 + 自适应节拍；命中即停。
- **ALIGN**：两段——先 `_drive_to(lane)` 到位，再 `_turn_to(GRASP_YAW)` 对准 + 0.3s 稳定 → DEPLOY。
- **DEPLOY/CREEP/CLOSE/LIFT/RETREAT**：官方抓取流程。
- **RETURN**：取到后**写死**折线到 `OBSTACLE_ENTRY=(-0.50,2.475)`。
- **NAV_TABLE**：`_navigate`(A*+DWB) 到 `TABLE_APPROACH=(-1.88,-2.80)`（**唯一用规划器的段**）。
- **PLACE**：降 slide + 开爪。

### 关键常量
`YELLOW_MID_Y=2.475`、`GRASP_YAW=π/2−11°`、`APPROACH_DX=0.068`、`SHELF_X={A:-1.735,B:-0.850,C:0.035,D:0.920,E:1.805}`、`COLUMN_DX={C1:-0.22,C2:0,C3:0.22}`、`SCAN_Y=2.40`、`SHELF_STEP=0.885`、`EXPLORE_X0=1.805`、`EXPLORE_SPEED=0.06`、`EXPLORE_TURN_MAX=0.15`、`SCAN_SLIDE=0.11`、`HEAD_PITCH=-0.6`、`SLIDE_GRASP=0.11`、`CREEP_SPEED=0.08`、`RETREAT_SPEED=0.12`、`PLACE_LOWER_SLIDE=0.17`、`OBSTACLE_ENTRY=(-0.50,2.475)`、`TABLE_APPROACH=(-1.88,-2.80)`。

## 5. 最近改动（**部分未提交**）
已提交：DWB、两层膨胀、脱困背离障碍、任务解析、ArUco、9 类模型等（最新 `e3ccb4a`）。
**未提交本地改动**：
1. `perception.py`：`NEAR_LAT_MAX 0.80→0.35`（减少邻架斜视角误检）。
2. `competition_client.py`：
   - `EXPLORE_X0 1.70→1.805`（首扫对准货架 E 中心）。
   - `_select_target` 改**当前货架优先**（按 `|shelf_x−base_x|` 降权）。
   - **ALIGN 拆两段**（`pos→yaw` + 0.3s 稳定）。
   - `_turn_to` 增益 0.8→0.5。
   - 新增 `RETURN` + `OBSTACLE_ENTRY`；`ALIGN/RETURN` 加入 `phase_timeouts`。
   - 探索原语去掉 `front_clear`（取货区不避障）。

## 6. 已知问题（均已改，**待验证**）
1. 首轮先跑“右二”货架 D：横向过滤太宽 + 全局置信度选择 → 已改。
2. ALIGN 卡死/乱转：`drive_to and turn_to` 同拍互扰 → 已改两段式。
3. 取到后切回规划器又脱困：`RETREAT→NAV_TABLE` 直接进 A*/DWB → 已加 `RETURN`。
4. 起步偏右、右臂贴东墙 → `EXPLORE_X0` 已移 1.805。
5. IK 偶发失败（停偏/朝向）→ 先到位再对准，**后续用 ArUco 微调**。
6. 未验证：随机场景（45 商品 + 5 障碍 + 多订单）。

## 7. 下一步计划
已确认：
- **A** 验证第 5/6 节修复（固定场景单跑）。
- **B** 抓取前 **ArUco 微调**（读 `/competition/aruco_detections`，ID→货架/层/列固定），对准列中心与朝向再 DEPLOY。
- **C** 闭环原语加“死区 + 到位稳定”（转向 2–3°、直行 3–5cm，每段停 0.3s）。

可选：
- **D** 黑箱自动调参（类 RL）：基准“转 90° + 直行 1m”，cost=误差+超调+时间+振荡；随机/网格→CMA-ES/贝叶斯，**只调客户端参数**（Server 轮速 PID 不可改）。
- **E** 随机场景回归：`SUPERMARKET_RANDOMIZE=1 RANDOMIZE_OBSTACLES=1 SEED=11 TASKS=all`。

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
1. 自动裁判口径：抓到**任意一件该 kind** 即算，还是**必须指定匿名 body**？
2. 提交形式：是否允许自建镜像，还是严格官方 Client 仅挂载代码？
3. 随机场景回程（配送台→货架区）是否需避障（会穿走廊）。
