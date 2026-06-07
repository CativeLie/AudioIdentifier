## Gradio 参考模板
import gradio as gr

def predict_audio(audio_path):
    #1.统一采样率与时长
    #2.转成128*128频谱图
    #3.送入CNN预测
    spec_path = "latest_spectrogram.png"
    return {"dog_bark": 0.82, "siren": 0.12}, spec_path

demo = gr.Interface(
    fn = predict.audio,
    inputs = gr.Audio(type = "filepath", label = "上传音频")
    outputs = [gr.Label(label = "分类结果"),gr.Image(type = "filepath", label = "频谱图")]
)
demo.launch