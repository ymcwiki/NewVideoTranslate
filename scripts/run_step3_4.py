#!/usr/bin/env python3
"""Run Steps 3-4: WhisperX transcription + Qwen2.5-7B translation."""
import os, sys, json, re, gc, shutil
from datetime import datetime
import numpy as np
import torch, torchaudio, librosa, soundfile as sf

os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

FOLDER = sys.argv[1]
DRIVE_ROOT = '/content/drive/MyDrive/video-translate'
ENV_PATH = f'{DRIVE_ROOT}/.env'

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
print(f'GPU: {GPU_NAME}')
print(f'Folder: {os.path.basename(FOLDER)}')

def unload_models(*models):
    for m in models:
        if m is not None:
            try: m.cpu()
            except: pass
            del m
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f'  VRAM: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total')

def clear_vram():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# ============================================================
# Step 3: WhisperX transcription + speaker diarization
# ============================================================
vocals_path = os.path.join(FOLDER, 'audio_vocals.wav')
transcript_path = os.path.join(FOLDER, 'transcript.json')
step3_done = os.path.join(FOLDER, '.step3_done')
speaker_folder = os.path.join(FOLDER, 'SPEAKER')

if not os.path.exists(step3_done):
    print('\n' + '='*60 + '\nStep 3: WhisperX transcription\n' + '='*60)
    import whisperx

    # Transcribe
    print('  Loading WhisperX model...')
    model = whisperx.load_model('large-v3', device='cuda', compute_type='int8_float16',
                                language='en')
    audio = whisperx.load_audio(vocals_path)
    print('  Transcribing...')
    result = model.transcribe(audio, batch_size=16)
    segments = result['segments']
    lang = result.get('language', 'en')
    print(f'  Language: {lang}, Segments: {len(segments)}')

    # Align
    print('  Aligning...')
    align_model, align_meta = whisperx.load_align_model(language_code=lang, device='cuda')
    result = whisperx.align(segments, align_model, align_meta, audio, device='cuda',
                           return_char_alignments=False)
    segments = result['segments']
    unload_models(align_model)

    # Diarize
    if HF_TOKEN:
        print('  Diarizing...')
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(token=HF_TOKEN, device=torch.device('cuda'))
            diarize_result = diarize_model(audio, min_speakers=2, max_speakers=5)
            result = whisperx.assign_word_speakers(diarize_result, {'segments': segments})
            segments = result['segments']
            unload_models(diarize_model)
            speakers = set(s.get('speaker', 'SPEAKER_00') for s in segments)
            print(f'  Speakers detected: {sorted(speakers)}')
        except Exception as e:
            print(f'  WARNING: Diarization failed ({str(e)[:100]}), skipping')
            print('  NOTE: Accept license at https://huggingface.co/pyannote/speaker-diarization-community-1')
            for s in segments:
                s['speaker'] = 'SPEAKER_00'
    else:
        print('  WARNING: No HF_TOKEN, skipping diarization')
        for s in segments:
            s['speaker'] = 'SPEAKER_00'

    unload_models(model)

    # Merge very short segments
    merged = []
    for s in segments:
        text = s.get('text', '').strip()
        if not text:
            continue
        if merged and s['start'] - merged[-1]['end'] < 0.3 and \
           s.get('speaker') == merged[-1].get('speaker') and \
           len(merged[-1].get('text', '')) < 80:
            merged[-1]['text'] += ' ' + text
            merged[-1]['end'] = s['end']
        else:
            merged.append({
                'start': s['start'],
                'end': s['end'],
                'text': text,
                'speaker': s.get('speaker', 'SPEAKER_00')
            })
    segments = merged
    print(f'  Merged to {len(segments)} segments')

    # Extract speaker reference audio
    os.makedirs(speaker_folder, exist_ok=True)
    full_audio, sr_full = librosa.load(vocals_path, sr=24000)
    all_speakers = set(s.get('speaker', 'SPEAKER_00') for s in segments)
    for spk in sorted(all_speakers):
        spk_segs = [s for s in segments if s.get('speaker') == spk]
        # Take up to 10s of the clearest speech
        spk_segs.sort(key=lambda x: x['end'] - x['start'], reverse=True)
        ref_chunks = []
        total_dur = 0
        for seg in spk_segs:
            if total_dur >= 10:
                break
            s_sample = int(seg['start'] * sr_full)
            e_sample = int(seg['end'] * sr_full)
            chunk = full_audio[s_sample:e_sample]
            ref_chunks.append(chunk)
            total_dur += seg['end'] - seg['start']
        if ref_chunks:
            ref_audio = np.concatenate(ref_chunks)[:sr_full * 10]
            spk_path = os.path.join(speaker_folder, f'{spk}.wav')
            sf.write(spk_path, ref_audio, sr_full)
            print(f'  Speaker {spk}: {len(spk_segs)} segs, {total_dur:.1f}s ref audio')

    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    open(step3_done, 'w').write(datetime.now().isoformat())
    print(f'  Transcription done: {len(segments)} segments')
else:
    print('Step 3: already done')

# ============================================================
# Step 4: Qwen2.5-7B-Instruct translation
# ============================================================
translation_path = os.path.join(FOLDER, 'translation.json')
checkpoint_path = os.path.join(FOLDER, 'translation_checkpoint.json')
step4_done = os.path.join(FOLDER, '.step4_done')

if not os.path.exists(step4_done):
    print('\n' + '='*60 + '\nStep 4: Qwen2.5-7B-Instruct translation\n' + '='*60)

    with open(transcript_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    # Load checkpoint if exists
    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            cp = json.load(f)
        if len(cp) == len(segments):
            start_idx = sum(1 for s in cp if 'translation' in s)
            segments = cp
            print(f'  Resuming from checkpoint: {start_idx}/{len(segments)}')

    clear_vram()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f'  Loading {model_id}...')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    free, total = torch.cuda.mem_get_info()
    print(f'  Model loaded. VRAM: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total')

    # Chinese speech rate: ~4 chars/sec. Allow 20% buffer.
    CHARS_PER_SEC = 4.8

    def translate_text(text, max_retries=2, duration=None):
        """Translate English text to Chinese using Qwen.
        duration: original segment duration in seconds, used to estimate max Chinese chars.
        """
        if not text.strip() or text.strip().startswith('['):
            return text

        # Estimate max Chinese chars based on segment duration
        max_chars = int(duration * CHARS_PER_SEC) if duration and duration > 0 else None

        base_prompt = "你是专业翻译。将英文翻译为简体中文。只输出中文译文，不要解释，不要输出原文。保持口语化，自然流畅。"
        if max_chars and max_chars < 80:
            base_prompt += f" 译文控制在{max_chars}字以内，表达要精炼。"

        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": text}
        ]

        result = None
        for attempt in range(max_retries + 1):
            try:
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

                max_new = min(max(len(text) * 2, 50), 512)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        temperature=0.3,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        do_sample=True,
                    )

                gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
                result = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                result = clean_translation(result, text)

                if result and not is_bad_translation(result, text):
                    # Check length: if too long for duration, retry with stricter prompt
                    if max_chars and len(result) > max_chars * 1.3 and attempt < max_retries:
                        shorter_max = int(max_chars * 0.9)
                        messages = [
                            {"role": "system", "content": f"你是专业翻译。将英文翻译为简体中文。译文必须控制在{shorter_max}字以内，用最精炼的表达。只输出译文。"},
                            {"role": "user", "content": text}
                        ]
                        continue
                    return result

                if attempt < max_retries:
                    messages[0]["content"] = "你是专业翻译。将以下英文翻译为简体中文。只输出翻译结果。"
                    continue

            except Exception as e:
                print(f'    Translation error: {str(e)[:100]}')

        return result if result else text

    def clean_translation(trans, original):
        """Remove common model artifacts from translation."""
        if not trans:
            return trans
        # Remove thinking tags
        trans = re.sub(r'<think>.*?</think>', '', trans, flags=re.DOTALL).strip()
        # Remove [Translation], [Image...], <<<...>>> artifacts
        trans = re.sub(r'\[Translation[:\]]*\]?\s*', '', trans)
        trans = re.sub(r'\[Image[^\]]*\]', '', trans)
        trans = re.sub(r'<<<[^>]*>>>', '', trans)
        trans = re.sub(r'<s>|</s>', '', trans)
        # Remove original English if echoed back
        if original in trans:
            trans = trans.replace(original, '').strip()
        # Remove leading/trailing punctuation artifacts
        trans = trans.strip(' \n\t.。，,')
        # If translation starts with English, try to extract Chinese part
        if trans and ord(trans[0]) < 0x4e00:
            # Check if there's Chinese content after some English
            chinese_match = re.search(r'[\u4e00-\u9fff]', trans)
            if chinese_match:
                # Take from first Chinese character
                idx = chinese_match.start()
                candidate = trans[idx:].strip()
                if len(candidate) > 5:
                    trans = candidate
        return trans.strip()

    def is_bad_translation(trans, original):
        """Check if translation is clearly bad."""
        if not trans:
            return True
        # Repetitive pattern: same phrase repeated many times
        if len(trans) > 100:
            words = trans.split()
            if len(words) > 10:
                # Check if any 3-gram repeats more than 3 times
                for i in range(len(words) - 2):
                    trigram = ' '.join(words[i:i+3])
                    if trans.count(trigram) > 3:
                        return True
        # Too long relative to original (Chinese should be similar or shorter)
        if len(trans) > len(original) * 4 and len(original) > 10:
            return True
        # Contains image hallucinations
        if '[Image' in trans or '[image' in trans:
            return True
        return False

    for i in range(start_idx, len(segments)):
        seg = segments[i]
        text = seg.get('text', '').strip()

        if not text or text.startswith('['):
            seg['translation'] = text
        else:
            dur = seg.get('end', 0) - seg.get('start', 0)
            trans = translate_text(text, duration=dur)
            seg['translation'] = trans

        if (i + 1) % 5 == 0:
            print(f'  [{i+1}/{len(segments)}] {text[:30]} → {seg["translation"][:30]}')

        # Checkpoint every 10 segments
        if (i + 1) % 10 == 0:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

    unload_models(model)
    del tokenizer

    # Final save
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    open(step4_done, 'w').write(datetime.now().isoformat())

    # Count results
    good = sum(1 for s in segments if s.get('translation', '').strip() and
               not s['translation'].startswith('['))
    print(f'  Translation done: {good}/{len(segments)} segments translated')
else:
    print('Step 4: already done')

print('\nSteps 3-4 complete!')
