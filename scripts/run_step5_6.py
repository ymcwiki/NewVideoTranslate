#!/usr/bin/env python3
"""Run Steps 5-6: TTS (Fish Speech S2 or IndexTTS2) + Video synthesis."""
import os, sys, json, re, gc, tempfile, shutil, subprocess, time
from datetime import datetime
import numpy as np
import torch, librosa, soundfile as sf

os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

FOLDER = sys.argv[1]
TTS_ENGINE = sys.argv[2] if len(sys.argv) > 2 else 'fish'
print(f'TTS engine: {TTS_ENGINE}', flush=True)

# ============================================================
# TTS Text Preprocessing: proper nouns + special chars
# ============================================================
TERM_MAP = {
    'ChatGPT': '恰特吉皮提',
    'GPT-4': '吉皮提四',
    'GPT-5': '吉皮提五',
    'GPT': '吉皮提',
    'Claude': '克劳德',
    'OpenAI': '欧朋爱',
    'AI': '人工智能',
    'API': '接口',
    'GitHub': '吉特哈布',
    'MCP': '模型上下文协议',
    'GPU': '显卡',
    'VRAM': '显存',
    'Opus': '欧帕斯',
    'Sonnet': '索内特',
    'Haiku': '俳句',
    'XTTS': '语音合成',
    'TTS': '语音合成',
    'LLM': '大语言模型',
}

def preprocess_tts_text(text):
    """预处理翻译文本，让 TTS 发音更自然"""
    # 1. 专有名词替换（按长度降序，避免部分匹配）
    for en, zh in sorted(TERM_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(en, zh)
    # 2. 特殊字符转中文
    text = text.replace('°', '度')
    text = text.replace('²', '的平方')
    text = text.replace('—', '，')
    text = text.replace('…', '。')
    # 3. 清理多余空格（中文不需要）
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[,，]{2,}', '，', text)
    return text.strip()
DRIVE_ROOT = '/content/drive/MyDrive/video-translate'
ENV_PATH = f'{DRIVE_ROOT}/.env'
FISH_DIR = '/content/fish-speech'
CKPT_DIR = '/content/checkpoints/s2-pro'
SPEED_UP, BG_VOLUME, TTS_VOLUME = 1.05, 0.3, 1.0
TTS_SR = 44100  # Fish Speech S2 native output rate

# Load HF_TOKEN
HF_TOKEN = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith('HF_TOKEN'):
                HF_TOKEN = line.split('=')[1].strip().strip("'\"")
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN

GPU_NAME = torch.cuda.get_device_name(0)
print(f'GPU: {GPU_NAME}', flush=True)
print(f'Folder: {os.path.basename(FOLDER)}', flush=True)

def clear_vram():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

def adjust_audio_length(audio, sr, desired_length, max_length=None, min_speed=0.7, max_speed=1.1):
    """Adjust audio duration to fit desired_length using ffmpeg atempo."""
    cur = len(audio) / sr
    if cur <= 0: return audio, cur
    sf_val = max(min(desired_length / cur, max_speed), min_speed)
    actual = cur * sf_val
    tempo = 1.0 / sf_val
    if abs(tempo - 1.0) < 0.01:
        adj = audio
    else:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            tin = tmp.name
        tout = tin.replace('.wav', '_a.wav')
        try:
            fs = []
            t = tempo
            while t > 2.0: fs.append('atempo=2.0'); t /= 2.0
            while t < 0.5: fs.append('atempo=0.5'); t *= 2.0
            fs.append(f'atempo={t:.4f}')
            r = subprocess.run(['ffmpeg','-y','-i',tin,'-filter:a',','.join(fs),'-ar',str(sr),tout],
                             capture_output=True, timeout=30)
            adj = sf.read(tout)[0].astype(np.float32) if r.returncode == 0 and os.path.exists(tout) else audio
        except Exception:
            adj = audio
        finally:
            for p in [tin, tout]:
                if os.path.exists(p): os.remove(p)
    if max_length and actual > max_length: actual = max_length
    return adj[:int(actual * sr)], actual

# ============================================================
# Step 5: TTS — Fish Speech S2 or IndexTTS2
# ============================================================
translation_path = os.path.join(FOLDER, 'translation.json')
vocals_path = os.path.join(FOLDER, 'audio_vocals.wav')
step5_done = os.path.join(FOLDER, '.step5_done')
wavs_dir = os.path.join(FOLDER, 'wavs')

if not os.path.exists(step5_done):
    print('\n' + '='*60 + f'\nStep 5: TTS ({TTS_ENGINE})\n' + '='*60, flush=True)
    os.makedirs(wavs_dir, exist_ok=True)
    with open(translation_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    clear_vram()

    if TTS_ENGINE == 'fish':
        print('  Using Fish Speech S2 engine', flush=True)

        # Phase 1: Load models directly in this process (NO subprocess)
        print('  Loading Fish Speech models directly...', flush=True)
        print(f'  Checkpoint: {CKPT_DIR}', flush=True)

        import torchaudio
        from fish_speech.models.text2semantic.inference import (
            init_model, generate_long,
        )
        from fish_speech.models.dac.inference import load_model as load_decoder_model

        precision = torch.bfloat16

        # Load LLM (DualARTransformer)
        print('  Loading LLM...', flush=True)
        model, decode_one_token = init_model(CKPT_DIR, 'cuda', precision)

        # CRITICAL FIX: inject semantic token IDs if not already set by from_pretrained
        # New s2-pro models auto-inject via tokenizer_config.json; old ones need special_tokens.json
        if model.config.semantic_begin_id == 0:
            _st_path = os.path.join(CKPT_DIR, 'special_tokens.json')
            if os.path.exists(_st_path):
                with open(_st_path) as _f:
                    _special = json.load(_f)
                _sem_ids = [v for k, v in _special.items() if k.startswith('<|semantic:')]
                if _sem_ids:
                    model.config.semantic_begin_id = min(_sem_ids)
                    model.config.semantic_end_id = max(_sem_ids)
            else:
                print('  WARNING: semantic_begin_id=0 and no special_tokens.json found!', flush=True)
        print(f'  Semantic IDs: begin={model.config.semantic_begin_id}, end={model.config.semantic_end_id}', flush=True)

        with torch.device('cuda'):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True
        print('  LLM loaded.', flush=True)

        # Load DAC codec (decoder/encoder)
        print('  Loading codec...', flush=True)
        codec_path = os.path.join(CKPT_DIR, 'codec.pth')
        if not os.path.exists(codec_path):
            for candidate in ['firefly_gan_vq', 'codec']:
                p = os.path.join(CKPT_DIR, candidate)
                if os.path.exists(p):
                    codec_path = p
                    break
        decoder = load_decoder_model(
            config_name='modded_dac_vq',
            checkpoint_path=codec_path,
            device='cuda'
        )
        TTS_SR = decoder.sample_rate
        print(f'  Codec loaded. Sample rate: {TTS_SR}', flush=True)

        free, total = torch.cuda.mem_get_info()
        print(f'  VRAM: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total', flush=True)

        # Phase 2: Encode user's reference voice (stored on Drive, shared across all videos)
        ref_voice_path = os.path.join(DRIVE_ROOT, 'ref_voice.wav')
        ref_codes = None
        if os.path.exists(ref_voice_path):
            wav, sr_orig = torchaudio.load(ref_voice_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = torchaudio.functional.resample(wav, sr_orig, TTS_SR)
            model_dtype = next(decoder.parameters()).dtype
            audios = wav[None].to('cuda', dtype=model_dtype)
            audio_lengths = torch.tensor([wav.shape[1]], device='cuda', dtype=torch.long)
            with torch.inference_mode():
                indices, feat_lens = decoder.encode(audios, audio_lengths)
                ref_codes = indices[0, :, :feat_lens[0]].cpu()
            print(f'  Reference voice: {wav.shape[1]/TTS_SR:.1f}s, {ref_codes.shape[1]} frames', flush=True)
        else:
            print(f'  WARNING: ref_voice.wav not found at {ref_voice_path}', flush=True)
            print(f'  TTS will generate without voice reference (inconsistent voices!)', flush=True)

        # Phase 3: Generate TTS for each segment via direct inference
        failed = 0
        for i, seg in enumerate(segments):
            wp = os.path.join(wavs_dir, f'{i:04d}.wav')
            if os.path.exists(wp) and os.path.getsize(wp) > 100:
                continue

            text = seg.get('translation', '').strip()
            if not text or text.startswith('['):
                sf.write(wp, np.zeros(int(0.1 * TTS_SR), dtype=np.float32), TTS_SR)
                continue

            # Fish Speech expects speaker-tagged text
            tagged_text = f"<|speaker:0|>{preprocess_tts_text(text)}"

            success = False
            for retry in range(3):
                try:
                    codes_list = []
                    for response in generate_long(
                        model=model,
                        device='cuda',
                        decode_one_token=decode_one_token,
                        text=tagged_text,
                        max_new_tokens=2048,
                        top_p=0.8,
                        repetition_penalty=1.5,
                        temperature=0.7,
                        chunk_length=200,
                        prompt_text=[''] if ref_codes is not None else None,
                        prompt_tokens=[ref_codes] if ref_codes is not None else None,
                    ):
                        if response.action == 'sample':
                            codes_list.append(response.codes)
                        elif response.action == 'next':
                            break

                    if codes_list:
                        merged_codes = torch.cat(codes_list, dim=1).to('cuda')
                        with torch.inference_mode():
                            audio_tensor = decoder.from_indices(merged_codes[None])[0].squeeze()
                        audio_np = audio_tensor.float().cpu().numpy()
                        if len(audio_np) > 100:
                            sf.write(wp, audio_np, TTS_SR)
                            success = True
                            break

                    if retry < 2:
                        time.sleep(1)
                except Exception as e:
                    if retry < 2:
                        time.sleep(1)
                    else:
                        print(f'  Warning [{i}]: {str(e)[:150]}', flush=True)

            if not success:
                failed += 1
                sf.write(wp, np.zeros(int(0.5 * TTS_SR), dtype=np.float32), TTS_SR)

            if (i + 1) % 10 == 0:
                print(f'  [{i+1}/{len(segments)}] {text[:40]}', flush=True)

        # Phase 4: Free VRAM
        print('  Freeing models...', flush=True)
        del model, decode_one_token, decoder
        clear_vram()
        free, total = torch.cuda.mem_get_info()
        print(f'  VRAM: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total', flush=True)

    elif TTS_ENGINE == 'indextts':
        print('  Using IndexTTS2 engine', flush=True)
        INDEXTTS_DIR = '/content/index-tts'
        INDEXTTS_CKPT = os.path.join(INDEXTTS_DIR, 'checkpoints')

        sys.path.insert(0, INDEXTTS_DIR)
        from indextts.infer_v2 import IndexTTS2

        tts = IndexTTS2(
            cfg_path=os.path.join(INDEXTTS_CKPT, 'config.yaml'),
            model_dir=INDEXTTS_CKPT,
            use_fp16=True,
            device='cuda',
        )
        TTS_SR = 22050  # IndexTTS2 native output rate

        ref_voice_path = os.path.join(DRIVE_ROOT, 'ref_voice.wav')
        if not os.path.exists(ref_voice_path):
            print(f'  WARNING: ref_voice.wav not found at {ref_voice_path}', flush=True)

        free, total = torch.cuda.mem_get_info()
        print(f'  VRAM: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total', flush=True)

        failed = 0
        for i, seg in enumerate(segments):
            wp = os.path.join(wavs_dir, f'{i:04d}.wav')
            if os.path.exists(wp) and os.path.getsize(wp) > 100:
                continue

            text = seg.get('translation', '').strip()
            if not text or text.startswith('['):
                sf.write(wp, np.zeros(int(0.1 * TTS_SR), dtype=np.float32), TTS_SR)
                continue

            processed_text = preprocess_tts_text(text)
            success = False
            for retry in range(3):
                try:
                    tts.infer(
                        spk_audio_prompt=ref_voice_path,
                        text=processed_text,
                        output_path=wp,
                        verbose=False,
                    )
                    if os.path.exists(wp) and os.path.getsize(wp) > 100:
                        success = True
                        break
                except Exception as e:
                    if retry < 2:
                        time.sleep(1)
                    else:
                        print(f'  Warning [{i}]: {str(e)[:150]}', flush=True)

            if not success:
                failed += 1
                sf.write(wp, np.zeros(int(0.5 * TTS_SR), dtype=np.float32), TTS_SR)

            if (i + 1) % 10 == 0:
                print(f'  [{i+1}/{len(segments)}] {text[:40]}', flush=True)

        # Free VRAM
        print('  Freeing models...', flush=True)
        del tts
        clear_vram()

    open(step5_done, 'w').write(f'{TTS_ENGINE}:{datetime.now().isoformat()}')
    valid = len(segments) - failed
    print(f'  TTS done: {valid}/{len(segments)} segments ({failed} failed)', flush=True)
else:
    print('Step 5: already done')

# ============================================================
# Step 6: Video synthesis
# ============================================================
step6_done = os.path.join(FOLDER, '.all_done')
if not os.path.exists(step6_done):
    print('\n' + '='*60 + '\nStep 6: Video synthesis\n' + '='*60)
    video_path = os.path.join(FOLDER, 'download.mp4')
    bg_path = os.path.join(FOLDER, 'audio_background.wav')
    output_video = os.path.join(FOLDER, 'video_chinese.mp4')
    srt_path = os.path.join(FOLDER, 'subtitles.srt')

    with open(translation_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    sr = TTS_SR
    print('  Stitching audio...')
    full_wav = np.zeros((0,), dtype=np.float32)
    for i, seg in enumerate(segments):
        wp = os.path.join(wavs_dir, f'{i:04d}.wav')
        if not os.path.exists(wp):
            continue
        try:
            au, _ = librosa.load(wp, sr=sr)
        except Exception:
            continue
        start, end = seg['start'], seg['end']
        last_end = len(full_wav) / sr
        if start > last_end:
            full_wav = np.concatenate((full_wav, np.zeros(int((start - last_end) * sr), dtype=np.float32)))
        actual_start = len(full_wav) / sr
        seg['start'] = actual_start
        # Allow audio to extend into gap before next segment (borrow up to 0.3s)
        if i < len(segments) - 1:
            next_start = segments[i+1]['start']
            strict_ml = max(0, next_start - actual_start)
            # Calculate gap between this segment's original end and next start
            next_gap = next_start - end if next_start > end else 0
            gap_buffer = min(0.3, next_gap * 0.5)
            ml = strict_ml + gap_buffer
        else:
            ml = None
        dl = end - start
        if dl > 0:
            au, al = adjust_audio_length(au, sr, dl, max_length=ml)
        else:
            al = len(au) / sr
            if ml and al > ml:
                au = au[:int(ml * sr)]
                al = ml
        full_wav = np.concatenate((full_wav, au))
        seg['end'] = actual_start + al

    # Volume matching
    if os.path.exists(vocals_path):
        vw, _ = librosa.load(vocals_path, sr=sr)
        mv, mt = np.max(np.abs(vw)), np.max(np.abs(full_wav))
        if mt > 0:
            full_wav = full_wav / mt * mv

    tts_path = os.path.join(FOLDER, 'audio_tts.wav')
    sf.write(tts_path, full_wav, sr)

    # Subtitles
    print('  Generating subtitles...')
    with open(srt_path, 'w', encoding='utf-8') as f:
        idx = 1
        for seg in segments:
            t = seg.get('translation', '').strip()
            if not t or t.startswith('['):
                continue
            s = seg['start'] / SPEED_UP
            e = seg['end'] / SPEED_UP
            sh, sm, ss = int(s // 3600), int(s % 3600 // 60), s % 60
            eh, em, es = int(e // 3600), int(e % 3600 // 60), e % 60
            # Split subtitles: max 15 chars/line to avoid player auto-wrap orphans
            MAX_LINE = 15
            lines = []
            if len(t) > MAX_LINE:
                # Split at punctuation first
                parts = re.split(r'([，。！？、；：])', t)
                cur_line = ''
                for part in parts:
                    if cur_line and len(cur_line + part) > MAX_LINE:
                        lines.append(cur_line)
                        cur_line = part
                    else:
                        cur_line += part
                if cur_line:
                    lines.append(cur_line)
                # Force split lines still too long
                final_lines = []
                for line in lines:
                    while len(line) > MAX_LINE + 3:
                        # Find a natural break point (avoid orphan chars)
                        bp = MAX_LINE
                        for k in range(MAX_LINE, max(MAX_LINE - 5, 0), -1):
                            if line[k] in '，。！？、；：的了是在有和':
                                bp = k + 1
                                break
                        final_lines.append(line[:bp])
                        line = line[bp:]
                    if line:
                        final_lines.append(line)
                lines = final_lines[:3]  # Max 3 lines
            else:
                lines = [t]

            f.write(f"{idx}\n")
            f.write(f"{sh:02d}:{sm:02d}:{int(ss):02d},{int((ss % 1) * 1000):03d} --> ")
            f.write(f"{eh:02d}:{em:02d}:{int(es):02d},{int((es % 1) * 1000):03d}\n")
            f.write('\n'.join(lines) + '\n\n')
            idx += 1

    # Mix audio
    final_audio = os.path.join(FOLDER, 'audio_final.wav')
    if os.path.exists(bg_path):
        print('  Mixing background audio...')
        bw, _ = librosa.load(bg_path, sr=sr)
        if len(full_wav) > len(bw):
            bw = np.pad(bw, (0, len(full_wav) - len(bw)))
        elif len(bw) > len(full_wav):
            full_wav = np.pad(full_wav, (0, len(bw) - len(full_wav)))
        comb = full_wav * TTS_VOLUME + bw * BG_VOLUME
        mx = np.max(np.abs(comb))
        if mx > 1.0:
            comb = comb / mx * 0.95
        sf.write(final_audio, comb, sr)
    else:
        shutil.copy(tts_path, final_audio)

    # Probe source video bitrate to match quality
    print('  Encoding video...')
    src_bitrate = None
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path],
            capture_output=True, text=True, timeout=10)
        src_bitrate = int(json.loads(probe.stdout).get('format', {}).get('bit_rate', 0))
    except Exception:
        pass
    # Use CRF 18 (visually near-lossless) with medium preset; cap bitrate to source
    encode_args = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18']
    if src_bitrate and src_bitrate > 0:
        # Cap at source bitrate to avoid bloating file size
        maxrate = f'{int(src_bitrate / 1000)}k'
        bufsize = f'{int(src_bitrate / 500)}k'
        encode_args += ['-maxrate', maxrate, '-bufsize', bufsize]
        print(f'  Source bitrate: {src_bitrate/1e6:.1f} Mbps, capping output')

    srt_esc = srt_path.replace(':', '\\:').replace("'", "\\'")
    fc = f"[0:v]setpts=PTS/{SPEED_UP},subtitles='{srt_esc}':force_style='FontName=Noto Sans CJK SC,FontSize=22'[v];[1:a]atempo={SPEED_UP}[a]"
    r = subprocess.run([
        'ffmpeg', '-y', '-i', video_path, '-i', final_audio,
        '-filter_complex', fc, '-map', '[v]', '-map', '[a]',
        *encode_args,
        '-c:a', 'aac', '-b:a', '192k', output_video
    ], capture_output=True, text=True)

    if os.path.exists(output_video) and os.path.getsize(output_video) > 1000:
        sz = os.path.getsize(output_video) / 1e6
        print(f'  P1 Done! {sz:.1f}MB')

        # Generate P2: original video + Chinese subtitles (for Bilibili dual-P upload)
        p2_video = os.path.join(FOLDER, 'original_with_subs.mp4')
        if not os.path.exists(p2_video):
            print('  Encoding P2 (original + subtitles)...')
            # Generate SRT with original timestamps (not TTS-adjusted)
            p2_srt = os.path.join(FOLDER, 'subtitles_original.srt')
            with open(translation_path, 'r', encoding='utf-8') as f:
                orig_segs = json.load(f)
            with open(p2_srt, 'w', encoding='utf-8') as f:
                idx = 1
                for seg in orig_segs:
                    t = seg.get('translation', '').strip()
                    if not t or t.startswith('['): continue
                    s, e = seg.get('original_start', seg['start']), seg.get('original_end', seg['end'])
                    sh, sm, ss = int(s//3600), int(s%3600//60), s%60
                    eh, em, es = int(e//3600), int(e%3600//60), e%60
                    lines = [t] if len(t) <= 15 else [t[i:i+15] for i in range(0, len(t), 15)][:3]
                    f.write(f"{idx}\n{sh:02d}:{sm:02d}:{int(ss):02d},{int((ss%1)*1000):03d} --> {eh:02d}:{em:02d}:{int(es):02d},{int((es%1)*1000):03d}\n")
                    f.write('\n'.join(lines) + '\n\n')
                    idx += 1
            p2_srt_esc = p2_srt.replace(':', '\\:').replace("'", "\\'")
            p2_fc = f"subtitles='{p2_srt_esc}':force_style='FontName=Noto Sans CJK SC,FontSize=22'"
            r2 = subprocess.run([
                'ffmpeg', '-y', '-i', video_path, '-vf', p2_fc,
                '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                *((['-maxrate', maxrate, '-bufsize', bufsize] if src_bitrate and src_bitrate > 0 else [])),
                p2_video
            ], capture_output=True, text=True)
            if os.path.exists(p2_video) and os.path.getsize(p2_video) > 1000:
                print(f'  P2 Done! {os.path.getsize(p2_video)/1e6:.1f}MB')
            else:
                print(f'  P2 failed: {r2.stderr[:200] if r2.stderr else "unknown"}')

        open(step6_done, 'w').write(datetime.now().isoformat())
        print(f'  Output: {output_video}')
    else:
        print(f'  Encoding failed!')
        if r.stderr:
            print(f'  stderr: {r.stderr[:500]}')
else:
    print('Step 6: already done')

print('\nAll done!')
