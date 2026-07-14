# L0 · 面试地基（Foundation）

系统问一道面试题 → 用户对着摄像头回答 → 输出统计量（转录 + 凝视 + 语速）→ **不存任何音视频**。

## 架构
- **浏览器（纯 HTML/JS）**：取摄像头/麦克风、浏览器内 MediaPipe 算凝视代理、录音。**原始视频永不离开浏览器。**
- **服务器（FastAPI）**：本地 faster-whisper 转录、算 WPM、合并凝视、汇总落库 JSON。**音频转完即丢，不落盘。**

数据流：点「开始」→ 服务器出题 → 浏览器 TTS 念题 →（念完自动录音）→ 回答期间浏览器逐帧算凝视 → 点「结束」→ 音频经 HTTP 上传转录 → 汇总统计量落库 → 显示。

## 环境要求
- **Python 3.11 或 3.12**（faster-whisper 依赖 CTranslate2，对最新 Python 支持滞后，勿用 3.13）
- ffmpeg、cloudflared（经 Homebrew 安装）

## 跑起来
```bash
# 1. 建虚拟环境并装依赖（首次）
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install ffmpeg cloudflared

# 2. 本地启动
./run.sh                      # http://localhost:8000

# 3. 公网访问（另开一个终端）
./tunnel.sh                   # 拿到 https://xxx.trycloudflare.com 公网链接
```
> 浏览器摄像头/麦克风要求 HTTPS 或 localhost。cloudflared 给的是 https 链接，满足要求，别人点链接即可授权、无需安装任何东西。

首次转录会自动下载 whisper 模型（`small.en`，约 240MB）。

## 逐步验证
按 `docs/superpowers/plans/2026-07-12-L0-foundation.md` 的 Step 1→7，每步都有「跑通标志」。

## 消融开关（server/config.py）
- `RAG_ENABLED`：L0 恒 `False`（占位，L1 才接 RAG）。
- `MULTIMODAL_ENABLED`：`True`=统计量含**凝视 + 语速**；`False`=**只有转录（纯文本）**。对应 RQ2b。
- ⚠ 这是**消融开关**，与 L2 的 **HUD 显示开关**是两回事，勿混。

## 凝视阈值（server/config.py — 单一真源）
- `GAZE_ON_CAMERA_DEG`（默认 15 度）：头部朝向偏离镜头中心小于此角度即判定「看镜头」。是**代理**，非精确视线追踪。
- **只在 `server/config.py` 改一处**；前端页面加载时经 `GET /config` 拉取，不再硬编码（改阈值无需动前端）。
- **待调参数**：页面上「校准阈值」的两个按钮（直视镜头 / 看别处）各采集 5 秒，输出 `|yaw|/|pitch|` 的 p50/p90/max，据此定值。论文方法章节需交代取值与调法。

## RAG 检索（L1 Step 1，server/rag.py）
给一段回答，从知识库检索相关知识片段（还不接 LLM）。受 `config.RAG_ENABLED` 控制——关掉时 `retrieve()` 返回空列表（RQ2a 消融的基础）。

- **知识库**：`server/knowledge/` 下 `questions.json` + `general.json`（中文原文，给 LLM）与 `questions_en.json` + `general_en.json`（英文镜像，仅用于检索）。切块：每条 pitfall/hint/字段单独成块并带 `[Question: ...]` 上下文，约 46 块。
- **embedding**：本地 `all-MiniLM-L6-v2`（零成本、离线、不占 Gemini 额度），FAISS 内积检索（向量归一化 → 余弦相似度）。
- **索引产物**：`index.faiss` + `chunks.json` 由知识库生成，已 gitignore；首次 `retrieve()` 或索引缺失时自动 `build_index()` 重建。**改了知识库要手动重建**：删除这两个文件，或调用 `rag.build_index()`。
- **两处相对开发说明的偏离（有数据支撑）**：
  1. **英文镜像做索引**：知识库是中文、被试回答是英文，小型 embedding 模型跨语言细粒度判别弱（目标 pitfall 排 #6）。改用英文镜像算向量、中文原文返回给 LLM 后，目标命中 top-1。中英两份须结构对齐（题数、每题 pitfall/hint 条数一致），`_build_chunks()` 有断言防漂移；**改中文库时英文镜像要同步改**。
  2. **查询只用「回答」**：题目上下文已在每块的 `[Question:]` 前缀里；实测把题目也拼进 query 会引入泛化词干扰，把精确 pitfall 挤出 top-k，而跨题隔离并不因此变差（验证 3 仍 3/3 命中正确题）。若将来知识库变大或题目相近，需重估此策略。
- **依赖**：`sentence-transformers`、`faiss-cpu`（注意不是 `faiss`）。首次会下载 embedding 模型（~80MB）。

## 数据安全（伦理承诺）
- **只存统计量**：`results/session_<id>.json` 里仅有转录文本、凝视占比、平均 WPM、flags 与时间戳。
- **绝不存音视频**：视频不出浏览器；音频服务器转完即丢，不落盘（ffmpeg 解码用的临时文件在函数返回前立即删除）。每轮落库后 `assert_no_media()` 自动自检 `results/` 目录，发现任何音视频扩展名即抛错。
- **`results/` 是被试的研究数据**，已加入 `.gitignore`，**绝不能进 git 仓库**。
- **存储安全**：`results/` 仅存于运行主机本地。正式用户研究前应放到受控目录（可加访问控制/加密，`session_id` 可改为匿名标识）。对齐伦理清单「数据安全存储」一条。
- **自查红线**（L0 完成标志之一）：
  ```bash
  find . \( -name "*.webm" -o -name "*.wav" -o -name "*.mp4" -o -name "*.m4a" -o -name "*.ogg" \) -not -path "*/.venv/*"
  # 应无输出
  ```

## 测试
```bash
.venv/bin/python -m pytest
```
纯 Python 逻辑（转录接口、WPM、凝视聚合与平滑、汇总/落库/红线断言）由 pytest 覆盖；摄像头/凝视/录音/部署等浏览器与硬件相关部分按「逐步验证」手动确认。

## L0 之后
L0 = 可远程访问、伦理合规、能产出行为统计量的 Web 系统。之后进 L1：加 LLM + RAG，生成纯文本反馈，搭裁判 + 专家校准 + RAG 消融。
