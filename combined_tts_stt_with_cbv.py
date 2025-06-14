import gradio as gr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import threading
import time
import matplotlib.pyplot as plt
from faster_whisper import WhisperModel
from gtts import gTTS
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


class CivilAISystem:
    def __init__(self):
        # Load Whisper model
        self.whisper_model = WhisperModel("base.en", compute_type="int8", device="cpu")
        
        # Load LangChain components
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorDB = FAISS.load_local("VDb", self.embedding_model, allow_dangerous_deserialization=True)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Audio recording state
        self.recording = False
        self.audio_data = []

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_data.append(indata.copy())

    def record_audio(self, duration=5, samplerate=16000):
        self.recording = True
        self.audio_data = []
        sd.default.samplerate = samplerate
        sd.default.channels = 1

        stream = sd.InputStream(callback=self.audio_callback)
        with stream:
            for _ in range(int(duration * 10)):
                if not self.recording:
                    break
                time.sleep(0.1)

        self.recording = False
        audio_np = np.concatenate(self.audio_data, axis=0)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(temp_wav.name, samplerate, audio_np)

        return temp_wav.name, audio_np

    def plot_waveform(self, audio_np):
        fig, ax = plt.subplots()
        ax.plot(audio_np)
        ax.set_title("Audio Waveform")
        ax.axis("off")

        temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(temp_img.name)
        plt.close(fig)
        return temp_img.name

    def transcribe_audio(self, duration):
        status = "🎙 Listening... Speak now."
        yield status, None, ""

        audio_path, audio_np = self.record_audio(duration)
        yield "🛑 Recording stopped. Processing audio...", None, ""

        waveform_img = self.plot_waveform(audio_np)
        segments, _ = self.whisper_model.transcribe(audio_path, language="en")
        result_text = " ".join([segment.text for segment in segments])

        yield "✅ Transcription complete.", waveform_img, result_text

    def stop_recording(self):
        self.recording = False
        return "🛑 Recording manually stopped. Processing audio..."

    def is_general_conversation(self, query):
        greetings = ["hi", "hello", "how are you", "what's up", "hey", "good morning",
                     "good evening", "who are you", "how do you work", "tell me about yourself"]
        return any(greet in query.lower() for greet in greetings)

    def run_ollama(self, prompt, model="mistral", temperature=0.7):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
        )
        return response.json()["response"]

    def civilai_query_with_docs(self, query):
        if self.is_general_conversation(query):
            answer = (
                "👋 Hi! I'm CivilAI, your assistant for road-related documentation.\n\n"
                "📂 I help you understand forest clearance, NH projects, road widening, DPRs, etc.\n"
                "💬 Try asking things like:\n"
                " - What's the forest clearance status?\n"
                " - Details of NH-458 DPR?\n"
                " - What are the coordinates for section 2?\n"
            )
            return answer, "No specific documents used."

        docs = self.vectorDB.similarity_search(query, k=10)
        if not docs:
            return "❌ No relevant information found.", "No documents matched."

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        top_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)][:3]

        combined_context = "\n\n".join(doc.page_content[:1000] for doc in top_docs)

        doc_info = "\n\n**Documents used:**\n" + "\n\n---\n".join(
            f"**Doc {i+1}** (Source: `{doc.metadata.get('source', 'Unknown')}`):\n"
            f"{doc.page_content[:300]}..."
            for i, doc in enumerate(top_docs)
        )

        prompt = f"""You are a helpful assistant for CivilAI. Use the context below to answer clearly.

Context:
{combined_context}

Question: {query}

Answer:"""

        answer = self.run_ollama(prompt)
        return answer, doc_info

    def respond_with_tts(self, query):
        answer, doc_info = self.civilai_query_with_docs(query)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            tts = gTTS(text=answer, lang='en')
            tts.save(fp.name)
            audio_path = fp.name
        return answer, doc_info, audio_path


system = CivilAISystem()

with gr.Blocks(title="CivilAI with Voice and Text") as demo:
    gr.Markdown("# 🛣️ CivilAI: Text + Voice Interface with TTS")

    with gr.Tab("🎙 Speak"):
        duration_input = gr.Slider(2, 10, value=5, label="Recording Duration (sec)")
        start_btn = gr.Button("Start Recording")
        stop_btn = gr.Button("Stop Recording")

        status_out = gr.Textbox(label="Status", interactive=False)
        waveform_out = gr.Image(type="filepath", label="Waveform")
        transcribed_query = gr.Textbox(label="Recognized Speech", lines=2)
        answer_out = gr.Textbox(label="Answer", interactive=False)
        doc_info_out = gr.Markdown(label="Docs Used")
        audio_out = gr.Audio(label="Answer (TTS)", type="filepath")

        start_btn.click(fn=system.transcribe_audio, inputs=duration_input, outputs=[status_out, waveform_out, transcribed_query])
        transcribed_query.change(fn=system.respond_with_tts, inputs=transcribed_query, outputs=[answer_out, doc_info_out, audio_out])
        stop_btn.click(fn=system.stop_recording, outputs=status_out)

    with gr.Tab("⌨️ Type"):
        user_query = gr.Textbox(label="Ask your question")
        submit_btn = gr.Button("Submit")

        typed_answer = gr.Textbox(label="Answer", interactive=False)
        typed_docs = gr.Markdown(label="Docs Used")
        typed_audio = gr.Audio(label="Answer (TTS)", type="filepath")

        submit_btn.click(fn=system.respond_with_tts, inputs=user_query, outputs=[typed_answer, typed_docs, typed_audio])


demo.launch()
