# 交接文档：新会话立即上手（导航/取货区修复中）

> 本文件是给“下一个对话”的即时交接。项目总体背景见 `HANDOFF.md`；
> 本文件只记录**当前正在做的事、当前 bug、以及立即要跑的调试命令**。

## 0. 一句话现状
固定 Baseline 场景的“扫描→抓取→退让→走廊→配送”已能端到端跑通（曾成功到 `DONE`），
但**走廊（障碍区）里会撞上固定障碍 box_05 并卡死**。用户认为是“全局代价地图没生效/雷达没扫到”，
实测**雷达数据正常**（见 §4），问题在规划器对动态障碍的使用。**当前正在排查这一点。**

## 1. 环境（重要）

### 容器
- `supermarket_sorting_server`（仿真，需 GPU + 显示）
- `supermarket_sorting_client`（我们的代码，挂载 `/home/makabaka/ruijie_code` → `/workspace/baseline`）

### 宿主机 NVIDIA（本会话踩过的坑，已解决）
- 驱动被自动升级过，导致**已加载内核模块版本 ≠ 用户态**，新容器起不来：
  - 报错 `Failed to initialize NVML: Driver/library version mismatch`、
    `failed to fulfil mount request: open /run/nvidia-persistenced/socket`、
    `open /usr/lib/.../libEGL_nvidia.so.580.173.02: no such file`。
- **用户已重启宿主机解决**（重启后内核模块与用户态一致）。若再次出现，唯一可靠办法是**重启主机**。
- 容器需要 X 授权：`DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority xhost +local:`

### 启动 Server（固定 Baseline，验证抓取+固定障碍）
```bash
docker rm -f supermarket_sorting_server
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority xhost +local:
docker run -d --gpus all --network host --ipc host --name supermarket_sorting_server \
  -e DISPLAY=:0 -e ROS_DOMAIN_ID=99 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw -e SUPERMARKET_HEADLESS=0 -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 -e SUPERMARKET_USE_GS=1 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e SUPERMARKET_FIXED_BASELINE=1 -e SUPERMARKET_RANDOMIZE=0 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=0 -e SUPERMARKET_TASKS=product_032 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v supermarket_sorting_cache:/root/.cache \
  supermarket_sorting:server \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```
- 随机障碍：把 `SUPERMARKET_RANDOMIZE_OBSTACLES=0` 改成 `1`（可加 `SUPERMARKET_OBSTACLE_SEED=11`）。
- 随机商品：`SUPERMARKET_FIXED_BASELINE=0 SUPERMARKET_RANDOMIZE=1`（可加 `SUPERMARKET_SEED=7`）。
- 固定 Baseline 的目标恒为 `product_032`（kele），位于 **D/L2/C2**（世界 `[0.92,3.243,0.924]`）。

### 启动 Client / 看日志 / 停车
```bash
# 跑（约 5-6 分钟一轮）
docker exec -d -w /workspace/baseline supermarket_sorting_client bash -lc \
 'source /opt/ros/humble/setup.bash && ROS_DOMAIN_ID=99 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp COMP_DEBUG=1 bash scripts/run_competition.sh > /tmp/comp.log 2>&1'
# 看
docker exec supermarket_sorting_client bash -lc "grep -aE 'phase=|\[close\]|\[fine\]|\[nav\]|\[nav-abort\]|reversing|escaping|timeout|stuck|planned' /tmp/comp.log | tail -40"
docker exec supermarket_sorting_client bash -lc "grep -aoE 'phase=[a-z_>-]+' /tmp/comp.log | uniq -c"
# 停车（务必杀干净，避免抢 /cmd_vel）
docker exec supermarket_sorting_client bash -lc "pkill -9 -f '[c]ompetition_client'; pkill -9 -f '[r]un_competition'; pkill -9 -f '[p]erception'"
```

## 2. 当前未提交改动（本会话刚做的“方案 1 回退”）
`git status`：`competition_client.py`、`mmk2_adapter.py`、`tests/test_planner.py` 已改；
`planner.py`、`local_planner.py` 已 `git checkout a19bba0 --`（回退到用户认可的避障版本）。
**尚未 commit。** 新会话应先决定是否 commit（建议 commit，信息：“导航回退到 a19bba0 口径 + 取货区降速去摆”）。

具体：
- `planner.py`（已回退）：`INFLATE_LOW=0.25`、`MIN_RANGE=0.25`、`LASER_OFFSET=(0.1137,0.0)`、
  `update_scan(ranges,angle_min,angle_inc,rx,ry,ryaw)`（**无** sensor_pose/range 参数）、无 `BLOCK_MARGIN`。
- `local_planner.py`（已回退）：`v_samples=[0,0.03,0.06,0.09,0.12]`、`w_samples=±0.25`、倒车 `[-0.10,-0.15]`、`safety=0.05`。
- `competition_client.py`：
  - `_navigate` 回退为 a19bba0 结构（每 tick 用**当前底盘位姿** `update_scan`；A* 重规划周期 1.5 s；
    缓冲 `margin<0.02` 触发 `_command_escape`；无 `_stall_recovery`/`_planned_path_invalid`/scan-odom 对齐）。
  - 保留：`scan` 陈旧（>1.5 s）即停；`_nav_abort` 诊断；`_command_escape` 脱困。
  - 取货区：`EXPLORE_SPEED=0.10`、`EXPLORE_TURN_MAX=0.15`（**关键：之前 0.4 导致 RETURN 左右摆**）。
  - 保留：手眼 ArUco 微调、按层 slide、CREEP 过中心、CLOSE 位置判据、RETREAT 直线后退。
  - 仍存在**死代码**：`_planned_path_invalid`、`_stall_recovery`、`_escape_step`、`_laser_pose_in_base`（未调用，可删）。
- `mmk2_adapter.py`：`MAX_LIN,MAX_ANG=0.12,0.18`；`MAX_LIN_ACC,MAX_ANG_ACC=0.2,0.8`（之前 0.4/0.8+0.8 会翻车）。
- `tests/test_planner.py`：改为匹配回退后的 `update_scan`（固定 `LASER_OFFSET` + `MIN_RANGE`）。

## 3. 立即要解决的 bug：走廊撞 box_05 卡死
**现象**（固定 Baseline，`/tmp/comp.log`）：
```
phase=nav->table base=(-0.78,0.30) yaw=-0.89 cmd=(0.12,0.08) meas_v=(0.002,-0.008) front_clear=False
planned 3 pts -> [-1.88 -2.8]        # 反复规划近乎直线路径
stuck (no progress toward [-1.88 -2.8 ], d=3.29)
```
- 机器人撞上 `box_05`（世界 `[-1.07,-0.197]`，约 0.58 m 远）后原地卡住。
- `front_clear=False` → 前向雷达看到 0.55 m 内有东西 → **雷达没坏**。

**已排除**：雷达数据正常（§4）。**怀疑点**（按可能性排序）：
1. **`update_scan` 的 `inside` 自过滤把 box_05 最近回波丢掉**：
   `FOOT_HALF_X/Y=0.35`，box_05 表面最近约 0.215 m，投影到激光系 `ex_b≈0.238, ey_b≈-0.175`
   → 落在 ±0.35 的 footprint 框内 → 被当自回波 `valid &= ~inside` 丢弃。只剩 box 远端被标，
   可能不足以形成阻挡，于是 A* 仍规划穿障。
2. **`update_scan` 用“当前底盘位姿”而不是“scan 时间戳位姿”**：机器人边转边走时，回波会被投影到
   错误的世界位置（尤其近距离），使 box 落点偏移。
3. **`INFLATE_LOW=0.25` 对 box_05 不够**：box 半宽 0.30，`d_low≈0.275` 时 `margin≈0.025>0`，
   机器人“贴着”box 也算可行；DWB `safety=0.05` 又要求 `margin>0.05`，两者不一致 → 规划器给的路 DWB 走不了，
   最后靠惯性撞上。
4. **DWB 的 `_simulate` 只看 `planner.margin`**：若动态层没标到 box，DWB 也认为前方可走。

**建议排查步骤**：
- 在 `_navigate` 的 `[nav]` 日志里临时加：`dynamic_cells=int(self.planner.dynamic.sum())`、
  `margin_at_base=float(self.planner.margin[ci,cj])`、`blocked=bool(self.planner.blocked[ci,cj])`。
- 用 `ros2 topic echo /slamware_ros_sdk_server_node/scan --once` 或 §4 脚本确认回波；
  再在客户端容器里直接构造 `GridPlanner`，喂一帧真实 scan + 真实位姿，打印 box_05 附近的
  `dynamic/margin/blocked`，确认 box 是否被标。
- 修复方向（待定，勿盲改）：
  a. 缩小 `FOOT_HALF_X/Y`（如 0.25）或对“最近回波”不做自过滤；
  b. `update_scan` 改用 scan 时间戳位姿（见 `a19bba0..dfc363c` 的 `pose_at`/`mapped_scan_seq` 写法，已被本会话回退掉，
     可参考 `git show dfc363c:competition_client/competition_client.py`）；
  c. 让 `INFLATE_LOW` 与 DWB `safety` 一致（例如 LOW=0.30 且 `blocked=margin<=0.05`），但注意
     `INFLATE_LOW=0.35` 会封死固定布局约 0.59 m 的通道（`git show dfc363c` 里踩过）。

## 4. 雷达数据核对脚本（已验证：数据正常）
```bash
docker exec -w /workspace/baseline supermarket_sorting_client bash -lc \
 'source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=99 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && python3 - <<PY
import rclpy, math, time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
rclpy.init(); n=Node("s")
st={"scan":None,"xy":None,"yaw":None}
def sc(m): st["scan"]=m
def od(m):
    q=m.pose.pose.orientation
    st["xy"]=(m.pose.pose.position.x,m.pose.pose.position.y)
    st["yaw"]=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
n.create_subscription(LaserScan,"/slamware_ros_sdk_server_node/scan",sc,10)
n.create_subscription(Odometry,"/slamware_ros_sdk_server_node/odom",od,10)
t0=time.time()
while time.time()-t0<3: rclpy.spin_once(n,timeout_sec=0.05)
m=st["scan"]; r=np.array(m.ranges,dtype=float)
valid=np.isfinite(r)&(r>m.range_min)&(r<m.range_max)
i=int(np.argmin(np.where(valid,r,1e9)))
print("frame",m.header.frame_id,"min",round(float(r[i]),3),"at_deg",round(math.degrees(m.angle_min+m.angle_increment*i),1))
print("robot",[round(v,3) for v in st["xy"]],"yaw",round(st["yaw"],2),"finite",int(valid.sum()),"/",r.size)
PY'
```
上次实测：`frame laser, min 0.215 @ -54.7°, robot (-0.781,0.299) yaw -0.9, finite 360/360`。

## 5. 关键常量（当前值）
| 位置 | 值 |
|---|---|
| `competition_client.EXPLORE_SPEED` | 0.10 |
| `competition_client.EXPLORE_TURN_MAX` | 0.15 |
| `competition_client.RETREAT_SPEED` | 0.12 |
| `competition_client.SLIDE_GRASP_BY_LEVEL` | `{L1:0.45, L2:0.11, L3:0.30}` |
| `competition_client.MIN_DEPLOY_FWD` | 0.58 |
| `competition_client.CREEP_STOP_GAP` | 0.035（CREEP 用 `stop_gap=-CREEP_STOP_GAP` 过中心） |
| `mmk2_adapter.GRIP_OPEN/GRIP_CLOSE` | 1.0 / **0.02** |
| `mmk2_adapter.MAX_LIN/MAX_ANG` | 0.12 / 0.18 |
| `mmk2_adapter.MAX_LIN_ACC/MAX_ANG_ACC` | 0.2 / 0.8 |
| `mmk2_adapter.BASE_GAIN_LIN/ANG` | 8.0 / 4.0（仿真响应约 1/8 线、~0.23 角，增益补偿） |
| `planner.INFLATE_HIGH/LOW` | 0.45 / 0.25 |
| `planner.MIN_RANGE` / `LASER_OFFSET` | 0.25 / `(0.1137, 0.0)` |
| `planner.FOOT_HALF_X/Y` | 0.35 / 0.35 |
| `local_planner` | `v≤0.12, w≤±0.25, safety=0.05` |

官方上限（用户给）：速度 0.45 m/s、1.2 rad/s；加速度 0.8 m/s²、5.0 rad/s²。

## 6. 关键代码位置（`competition_client/competition_client.py`）
- `_navigate`：走廊 A*/DWB 跟随（**当前 bug 所在**）。
- `_command_escape` / `_escape_dir`：背离障碍脱困（a19bba0 的脱困方式，用户认可“卡住才退”）。
- `_drive_to` / `_turn_to` / `_drive_dist`：取货区**硬编码直线/转向原语**（**禁止加激光/避障**）。
- `_tick_scan` / `_build_scan_plan` / `_do_scan`：增量扫描。
- `_select_target` / `_approach_lane` / `_lock_from_products` / `_fine_adjust_target`：目标选择、手眼微调。
- 相位：`WAIT_TASK, STOW, SCAN, ALIGN, DEPLOY, WAIT_ARM, CREEP, BRAKE, CLOSE, LIFT, RETREAT, RETURN, NAV_TABLE, PLACE, NEXT, NAV_RETURN, DONE, ERROR`。

## 7. 已知限制 / 待办
1. **走廊撞障卡死**（本文件 §3，最高优先级）。
2. 模拟器 `LaserScan` 实际约 **4–5 Hz**（不是 12 Hz），且客户端 tick 偶占执行器；scan 时龄阈值 1.5 s。
3. 取货区 RETURN 之前**左右摆**（已通过 `EXPLORE_TURN_MAX 0.4→0.15` 修复，实测平滑）。
4. 加速过大曾**翻车**（`MAX_LIN=0.40/accel 0.8`）；现为保守值。
5. **匿名 body 限制**：裁判按具体 body 计分；模拟器任务只给 `{id,kind}`。固定 Baseline 会把多余同类可乐搬到别的货架 L2，
   故按 kind 选可能选到非目标同类。真实赛题订单带 `location_id`（官方 PDF 第 32-33/42-61 行），届时可直接用。
6. L1 层 IK 不可达已用按层 slide 修复。
7. 未做：随机五订单端到端回归；YOLO 重训（现模型 mAP50≈0.99，收益小）；打包交付（见 §8）。

## 8. 交付打包（用户要求：**一个 tar，只放代码+自包含镜像**，部署文档单独交，传百度网盘）
- 官方 `supermarket_sorting:client` 内 `/workspace/baseline` 为空，代码是运行时挂载。
- 计划：`Dockerfile.delivery`（`FROM supermarket_sorting:client` + `COPY . /workspace/baseline` + 自动跑 `scripts/run_competition.sh`）
  → `docker build` → `docker save`（约 11 GB）→ 与代码一起打进一个 tar。**尚未开始。**
- 规则要求另交：部署说明文档、任务执行视频、作品介绍 PPT。

## 9. 提交历史（近期）
```
a3d8e5b 更新 HANDOFF：记录手眼 ArUco 微调、按层 slide、抓取/走廊修复与固定场景端到端跑通
3f3981d 走廊导航增加堵转检测与倒车脱困，扩展诊断
a377da6 接入手眼相机做 ArUco 微调，修复抓取与走廊导航卡死
dfc363c 合规化任务处理与增量扫描，走廊导航接入实时传感器   ← 引入 INFLATE_LOW=0.35/scan-odom 对齐（部分已回退）
a19bba0 改进抓取时序与目标锁定，提高识别和闭爪稳定性          ← 用户认可的“避障不错”版本
```
远端：`https://github.com/liljj-max/ruijie_docker`（`main`）。仓库约定见 `AGENTS.md`（实质改动后 commit+push）。

## 10. 合规红线（务必遵守）
- **不得**从任务 ID 字符串推导货位，**不得**硬编码商品→货位。已删除 `slot_from_target_id`/`LAYOUT_ORDER`/`prefer_slots`。
- 允许的固定结构：货架几何 `SHELF_X/COLUMN_DX/LEVEL_Z`、ArUco↔货位固定映射（`aruco_id_to_slot`）。
- **取货区（SCAN/ALIGN/RETURN）必须硬编码直线/转向，不得使用规划器/避障/激光恢复。**
