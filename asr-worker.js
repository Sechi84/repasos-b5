/* Reconocimiento de voz en el propio dispositivo (Whisper, gratuito y privado).
   El modelo se descarga una sola vez desde Hugging Face y queda guardado en el dispositivo. */
const LIB = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const MODEL = 'onnx-community/whisper-base';
let pipePromise = null;

async function getPipe() {
  if (!pipePromise) {
    pipePromise = (async () => {
      const T = await import(LIB);
      T.env.allowLocalModels = false;
      T.env.useBrowserCache = true;
      return T.pipeline('automatic-speech-recognition', MODEL, {
        dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' },
        device: 'wasm',
        progress_callback: p => {
          if (p.status === 'progress' && p.total) {
            self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
          }
        }
      });
    })().catch(e => { pipePromise = null; throw e; });
  }
  return pipePromise;
}

self.onmessage = async ev => {
  const { id, audio, lang } = ev.data || {};
  try {
    self.postMessage({ type: 'status', id, status: 'loading' });
    const asr = await getPipe();
    self.postMessage({ type: 'status', id, status: 'transcribing' });
    const out = await asr(audio, {
      language: lang || 'spanish',
      task: 'transcribe',
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: false
    });
    const text = Array.isArray(out) ? out.map(o => o.text).join(' ') : (out && out.text) || '';
    self.postMessage({ type: 'done', id, text: text.trim() });
  } catch (e) {
    self.postMessage({ type: 'error', id, error: String((e && e.message) || e) });
  }
};
