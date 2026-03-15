# NewVideoTranslate

**English YouTube videos → Chinese dubbed videos, 100% local models on Google Colab GPU. No cloud API needed.**

**英文 YouTube 视频 → 中文配音视频，全部使用本地模型在 Google Colab GPU 上运行，无需任何云端 API。**

## Features / 功能

- **Transcription / 转录**: WhisperX large-v3 with speaker diarization
- **Translation / 翻译**: Qwen2.5-7B-Instruct (duration-aware length control)
- **TTS / 语音合成**: Fish Speech S2 s2-pro (4.8B params, voice cloning)
- **Video Synthesis / 视频合成**: FFmpeg with burned-in Chinese subtitles
- **Bilibili Upload / B站上传**: Dual-part upload (P1: Chinese dubbed, P2: original + subtitles)
- **Checkpoint Resume / 断点续传**: Automatic resume after Colab disconnection

## Architecture / 架构

```
YouTube URL
  → Local: yt-dlp download (with metadata + thumbnail)
  → Google Drive: upload video
  → Colab GPU (via SSH over ngrok):
      Step 1-2: FFmpeg audio extraction + Demucs vocal separation
      Step 3:   WhisperX transcription + speaker diarization
      Step 4:   Qwen2.5-7B translation (with length control)
      Step 5:   Fish Speech S2 TTS (with voice cloning)
      Step 6:   FFmpeg video synthesis (P1 dubbed + P2 subtitled)
  → Local: Bilibili upload (dual-part + cover)
```

## Prerequisites / 前置条件

### Google Colab Pro (Required / 必需)

A **Colab Pro** (or higher) subscription is required for GPU access. Free-tier GPUs are insufficient.

**需要 Colab Pro 或更高级别的订阅**才能获得 GPU 访问权限。免费版 GPU 不够用。

- **Recommended / 推荐**: A100 GPU (~50 min for a 25-min video)
- **Minimum / 最低**: L4 GPU (~2 hours for a 25-min video)

### ngrok Account with TCP Tunnels (Required / 必需)

A free ngrok account does NOT support TCP tunnels. You must:

**免费 ngrok 账户不支持 TCP 隧道**，你必须：

1. Sign up at [ngrok.com](https://ngrok.com)
2. **Add a credit/debit card** to your account (required for TCP tunnels, even on the free plan)
3. Copy your auth token from the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)

> **Note / 注意**: ngrok requires a verified payment method to enable TCP tunnel functionality. This is ngrok's policy to prevent abuse. You will NOT be charged on the free plan, but a card must be on file.
>
> ngrok 要求绑定信用卡/借记卡才能启用 TCP 隧道功能。这是 ngrok 防止滥用的策略。免费计划不会产生费用，但必须绑定卡片。

### Other Requirements / 其他要求

- **macOS** with Google Drive for Desktop installed (for local file sync)
- **HuggingFace account** (optional, for speaker diarization — accept [pyannote license](https://huggingface.co/pyannote/speaker-diarization-community-1))
- **yt-dlp** installed locally (`brew install yt-dlp`)
- **ffmpeg** installed locally (`brew install ffmpeg`)
- **Python 3.10+** with `biliup` (`pip install biliup`) for Bilibili upload

## Setup / 配置

### 1. Prepare Google Drive / 准备 Google Drive

Create this folder structure in your Google Drive:

在 Google Drive 中创建如下目录结构：

```
My Drive/
└── video-translate/
    ├── .env                    # Credentials (see below)
    ├── ref_voice.wav           # Your voice reference for TTS (10-30s Chinese speech)
    ├── bilibili_cookies.json   # Optional: Bilibili upload cookies
    └── processing/             # Video processing folders (auto-created)
```

### 2. Create `.env` file / 创建 `.env` 文件

Create `video-translate/.env` in your Google Drive with:

在 Google Drive 中创建 `video-translate/.env`：

```env
# Required / 必需
NGROK_TOKEN=your_ngrok_auth_token_here
SSH_PASSWORD=your_ssh_password_here

# Optional / 可选
HF_TOKEN=your_huggingface_token_here
```

### 3. Upload the Colab Notebook / 上传 Colab Notebook

Upload `NewVideoTranslate.ipynb` to Google Colab.

将 `NewVideoTranslate.ipynb` 上传到 Google Colab。

### 4. Record Voice Reference / 录制声音参考

Record 10-30 seconds of yourself reading Chinese text. Save as `ref_voice.wav` (44.1kHz) in the `video-translate/` folder on Drive. This ensures consistent voice across all TTS segments.

录制 10-30 秒的中文朗读音频，保存为 `ref_voice.wav`（44.1kHz）到 Drive 的 `video-translate/` 目录。这确保所有 TTS 片段使用一致的声音。

### 5. Bilibili Upload (Optional) / B站上传（可选）

Create `video-translate/bilibili_cookies.json` on Drive:

在 Drive 创建 `video-translate/bilibili_cookies.json`：

```json
{
  "SESSDATA": "your_sessdata_here",
  "bili_jct": "your_bili_jct_here"
}
```

Get these values from browser DevTools (F12) → Application → Cookies → bilibili.com.

从浏览器开发者工具 (F12) → Application → Cookies → bilibili.com 获取这些值。

## Usage / 使用方法

### Step 1: Start Colab / 启动 Colab

1. Open the notebook in Colab
2. Switch to A100 GPU runtime (Runtime → Change runtime type)
3. Run Cell 1 (Mount Drive) — authorize when prompted
4. Run Cell 2 (SSH Tunnel) — copy the SSH host/port
5. Run Cell 3 (Keepalive) — keep this running

### Step 2: Run Pipeline via SSH / 通过 SSH 运行流水线

```bash
# Download video locally
yt-dlp -f "bestvideo[height<=1080]+bestaudio/best[height<=1080]" \
  --merge-output-format mp4 --write-info-json --write-thumbnail \
  -o "/tmp/download.%(ext)s" "YOUTUBE_URL"

# Upload to Drive
PROJ="~/Library/CloudStorage/GoogleDrive-.../video-translate/processing/Channel/2025-01-01 Title"
mkdir -p "$PROJ"
cp /tmp/download.mp4 "$PROJ/"
cp /tmp/download.info.json "$PROJ/" 2>/dev/null

# SSH into Colab and install dependencies (see SKILL.md for full commands)
sshpass -p "$SSH_PASSWORD" ssh root@host -p port

# Upload and run scripts
# Step 1-2: Audio extraction + Demucs
# Step 3-4: WhisperX + Qwen translation (run_step3_4.py)
# Step 5-6: Fish Speech TTS + Video synthesis (run_step5_6.py)
```

### Step 3: Upload to Bilibili / 上传到B站

```bash
python3 scripts/upload_bilibili.py "/path/to/video/folder"
```

## File Structure / 文件结构

```
NewVideoTranslate/
├── README.md                         # This file
├── NewVideoTranslate.ipynb           # Colab notebook (3 cells)
├── .env.example                      # Credential template
└── scripts/
    ├── run_step3_4.py                # WhisperX transcription + Qwen translation
    ├── run_step5_6.py                # Fish Speech TTS + FFmpeg video synthesis
    └── upload_bilibili.py            # Local Bilibili upload (dual-part + cover)
```

## Model Stack / 模型栈

| Step | Model | VRAM | Speed (A100) |
|------|-------|------|-------------|
| Vocal Separation | Demucs htdemucs_ft | ~2 GB | Fast |
| Transcription | WhisperX large-v3 | ~4-6 GB | Fast |
| Translation | Qwen2.5-7B-Instruct (bf16) | ~14 GB | ~50 tok/s |
| TTS | Fish Speech S2 s2-pro (4.8B) | ~22 GB | ~4.8 tok/s |

## Key Optimizations / 关键优化

- **Translation length control / 翻译长度控制**: Estimates max Chinese characters based on segment duration (4.8 chars/sec), retries with stricter prompts if too long
- **Audio speed limiting / 音频变速限制**: `min_speed=0.7` (max 1.43x speedup) prevents unnatural speech
- **Gap borrowing / 间隙借用**: Audio can extend into silence gaps between segments (up to 0.3s)
- **TTS text preprocessing / TTS 文本预处理**: Converts proper nouns to Chinese phonetics (ChatGPT → 恰特吉皮提, Claude → 克劳德, etc.)
- **Voice cloning / 声音克隆**: User's own voice reference ensures consistent output across all segments

## Troubleshooting / 故障排除

| Problem | Cause | Solution |
|---------|-------|----------|
| TTS outputs noise | `semantic_begin_id == 0` | Script auto-fixes; ensure latest version |
| Chinese subtitles show as boxes | Missing CJK font | `apt-get install fonts-noto-cjk && fc-cache -f` |
| Audio too fast | Translation too long for segment | Increase `min_speed` or enable translation length control |
| SSH disconnected | Colab timeout | Re-run Cell 2; checkpoint resume is automatic |
| ngrok TCP refused | No card on file | Add credit/debit card at ngrok.com |
| Speaker diarization 403 | HF license not accepted | Accept [pyannote license](https://huggingface.co/pyannote/speaker-diarization-community-1) |

## License / 许可

MIT

## Acknowledgments / 致谢

- [Fish Speech](https://github.com/fishaudio/fish-speech) — TTS model
- [WhisperX](https://github.com/m-bain/whisperX) — Speech recognition
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) — Translation model
- [biliup](https://github.com/biliup/biliup) — Bilibili upload
- [YouDub-webui](https://github.com/liuzhao1225/YouDub-webui) — Inspiration for TTS text preprocessing
