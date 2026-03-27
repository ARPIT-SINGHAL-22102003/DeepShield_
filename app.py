import os
import gradio as gr
import moviepy.editor as mp
from mesonet import MesoNet, predict_image
from lcnn import LCNN, predict_audio
from gradcam_utils import generate_heatmap
from report import generate_report
import torch

# ---- Models load karo ----
mesonet_model = MesoNet()
if os.path.exists('weights/mesonet.pth'):
    mesonet_model.load_state_dict(torch.load('weights/mesonet.pth', map_location='cpu'))
    print("MesoNet weights loaded!")
mesonet_model.eval()

lcnn_model = LCNN()
if os.path.exists('weights/lcnn.pth'):
    lcnn_model.load_state_dict(torch.load('weights/lcnn.pth', map_location='cpu'))
    print("LCNN weights loaded!")
lcnn_model.eval()

# ---- Video se audio nikalo ----
def extract_audio_from_video(video_path):
    clip = mp.VideoFileClip(video_path)
    audio_path = video_path.replace('.mp4', '_audio.wav')
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path

# ---- Video se frame nikalo ----
def extract_frame_from_video(video_path):
    clip = mp.VideoFileClip(video_path)
    frame_time = clip.duration / 2
    frame_path = video_path.replace('.mp4', '_frame.jpg')
    clip.save_frame(frame_path, t=frame_time)
    return frame_path

# ---- Fusion logic ----
def adaptive_fusion(score_v, score_a):
    alpha = score_v / (score_v + (1 - score_a) + 0.001)
    return alpha * score_v + (1 - alpha) * score_a, alpha

# ---- Main detection function ----
def analyze_file(file):
    if file is None:
        return "Koi file upload nahi ki!", None, None

    file_path = file.name
    ext = os.path.splitext(file_path)[1].lower()

    score_v, score_a, heatmap_path, alpha_t = None, None, None, 1.0

    try:
        if ext in ['.mp4', '.mkv', '.avi']:
            frame_path = extract_frame_from_video(file_path)
            audio_path = extract_audio_from_video(file_path)
            score_v = predict_image(frame_path)
            score_a = predict_audio(audio_path)
            final, alpha_t = adaptive_fusion(score_v, score_a)
            heatmap_path = generate_heatmap(frame_path, mesonet_model)

        elif ext in ['.jpg', '.jpeg', '.png']:
            score_v = predict_image(file_path)
            final = score_v
            alpha_t = 1.0
            heatmap_path = generate_heatmap(file_path, mesonet_model)

        elif ext in ['.wav', '.mp3', '.m4a']:
            score_a = predict_audio(file_path)
            final = score_a
            alpha_t = 0.0
            heatmap_path = None

        else:
            return "Unsupported format! Sirf mp4, jpg, png, wav, mp3 chalega!", None, None

        # ---- Verdict ----
        verdict = "DEEPFAKE DETECTED" if final > 0.5 else "LIKELY REAL"
        confidence = final if final > 0.5 else 1 - final

        # ---- Result text ----
        result_text = f"""
## {"DEEPFAKE DETECTED" if final > 0.5 else "LIKELY REAL"}

| Detail | Value |
|--------|-------|
| Confidence | {confidence:.1%} |
| Visual Score | {f'{score_v:.4f}' if score_v is not None else 'N/A'} |
| Audio Score | {f'{score_a:.4f}' if score_a is not None else 'N/A'} |
| Fusion Weight (alpha) | {alpha_t:.4f} |
| Decision | {"FAKE" if final > 0.5 else "REAL"} |
        """

        # ---- PDF report ----
        report_path = generate_report(
            file_path,
            verdict,
            confidence,
            score_v,
            score_a,
            alpha_t,
            heatmap_path
        )

        return result_text, heatmap_path, report_path

    except Exception as e:
        return f"Error aaya: {str(e)}", None, None


# ---- Gradio UI ----
with gr.Blocks(title="DeepShield", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # DeepShield - Deepfake Detection System
    ### AI-Powered Multi-Modal Deepfake Detection
    Upload any image, video or audio file to detect if it is real or fake!
    """)

    gr.Markdown("""
    **Supported Formats:**
    Video: .mp4, .mkv, .avi | Image: .jpg, .png | Audio: .wav, .mp3
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload File Here",
                file_types=['.mp4', '.jpg', '.jpeg',
                           '.png', '.wav', '.mp3',
                           '.mkv', '.avi']
            )
            analyze_btn = gr.Button(
                "Analyze",
                variant="primary",
                size="lg"
            )
            gr.Markdown("""
            **How it works:**
            1. Upload your file
            2. Click Analyze
            3. View results and heatmap
            4. Download forensic report
            """)

        with gr.Column(scale=2):
            result_output = gr.Markdown(label="Detection Result")

            with gr.Row():
                heatmap_output = gr.Image(
                    label="Grad-CAM Heatmap (Red = Fake, Blue = Real)"
                )

            report_output = gr.File(
                label="Download Forensic PDF Report"
            )

    analyze_btn.click(
        fn=analyze_file,
        inputs=file_input,
        outputs=[result_output, heatmap_output, report_output]
    )

    gr.Markdown("""
    ---
    **Disclaimer:** Confidence below 60% indicates borderline cases.
    DeepShield is for forensic assistance only.
    
    **Model Info:**
    - Visual Module: MesoNet (Trained: 91.1% accuracy)
    - Audio Module: LCNN (Spectro-temporal analysis)
    - Fusion: Adaptive alpha weighting
    """)

app.launch()