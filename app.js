import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';

// The open-weight CLIP model runs in the browser through Transformers.js.
// It needs a connection for the first download; subsequent inference is local.
env.allowLocalModels = false;
const plantLabels = [
  'a banana plant (matooke)', 'a coffee plant', 'a mango tree', 'an avocado tree',
  'a cassava plant', 'a maize plant', 'a jackfruit tree', 'a papaya plant',
  'a guava tree', 'a hibiscus flower', 'a bougainvillea plant', 'a neem tree',
  'an aloe vera plant', 'a passionfruit vine', 'a tomato plant', 'a bean plant',
  'a sweet potato vine', 'an amaranth plant', 'a eucalyptus tree', 'grass',
  'a different kind of plant', 'not a plant'
];

const input = document.querySelector('#photoInput');
const preview = document.querySelector('#preview');
const emptyState = document.querySelector('#emptyState');
const changePhoto = document.querySelector('#changePhoto');
const welcome = document.querySelector('#welcomeResult');
const loading = document.querySelector('#loadingResult');
const answer = document.querySelector('#answerResult');
const loadingTitle = document.querySelector('#loadingTitle');
const loadingMessage = document.querySelector('#loadingMessage');
const statusDot = document.querySelector('#statusDot');
let classifier;
let gemma;
let activeUrl;

function setBusy(isBusy) {
  welcome.hidden = true;
  loading.hidden = !isBusy;
  answer.hidden = true;
  statusDot.style.background = isBusy ? '#d6a74d' : '#78aa62';
}

function showWelcome() {
  welcome.hidden = false;
  loading.hidden = true;
  answer.hidden = true;
  statusDot.style.background = '#78aa62';
}

function outdoorNote(label) {
  const name = label.toLowerCase();
  if (name.includes('leaf') || name.includes('plant') || name.includes('tree') || name.includes('flower') || name.includes('mushroom')) {
    return 'Look for a second clue nearby: leaf edges, stem shape, flowers, or the way it grows. A wider view can help you notice its habitat.';
  }
  return 'Try a wider photo in natural light, then compare leaf shape, stem and nearby flowers. The model may be seeing background objects instead of the plant.';
}

function renderPredictions(predictions) {
  const best = predictions[0];
  const alternatives = predictions.slice(1, 3).map(item => item.label).join(' · ');
  const confidence = Math.round(best.score * 100);
  answer.innerHTML = `
    <div class="answer">
      <span class="answer-label">MODEL'S BEST GUESS</span>
      <h2>${best.label}</h2>
      <p class="answer-sub">Closest match from a Kampala-friendly shortlist</p>
      <div class="confidence-row"><span>Relative match</span><span>${confidence}%</span></div>
      <div class="confidence-track"><span style="width:${confidence}%"></span></div>
      ${alternatives ? `<p class="other-guesses">Also considered: ${alternatives}</p>` : ''}
      <div class="outdoor-tip">${outdoorNote(best.label)}</div>
      <div class="gemma-quest">
        <button class="quest-button" id="questButton">✳ Ask Gemma for a tiny field quest</button>
        <p class="gemma-note">Gemma runs locally in your browser. Its model downloads only if you tap.</p>
        <div class="quest-result" id="questResult" hidden></div>
      </div>
      <p class="disclaimer">This is a comparison score across the shortlist, not a calibrated probability. The model can be wrong; it is not a species or safety identifier. Never eat or handle a plant based on this guess.</p>
    </div>`;
  loading.hidden = true;
  welcome.hidden = true;
  answer.hidden = false;
  statusDot.style.background = '#78aa62';
  document.querySelector('#questButton').addEventListener('click', () => makeQuest(best.label));
}

async function makeQuest(guess) {
  const button = document.querySelector('#questButton');
  const result = document.querySelector('#questResult');
  button.disabled = true;
  button.textContent = 'Loading Gemma locally…';
  result.hidden = false;
  result.textContent = 'The Gemma model downloads on first use. This may take a few minutes.';
  try {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser. Try a recent version of Chrome or Edge on a WebGPU-capable device.');
    if (!gemma) {
      gemma = await pipeline('text-generation', 'onnx-community/gemma-3-270m-it-ONNX', {
        device: 'webgpu',
        dtype: 'q4f16',
        progress_callback: progress => {
          if (progress.status === 'progress' && progress.total) {
            result.textContent = `Downloading Gemma: ${Math.round(progress.progress)}%. This one-time download stays in your browser cache.`;
          }
        }
      });
    }
    button.textContent = 'Gemma is making a field quest…';
    const messages = [
      { role: 'system', content: 'You create safe, simple outdoor observation quests. Never suggest touching, tasting, picking, or damaging plants. Do not claim a plant identification is certain. Use one short sentence.' },
      { role: 'user', content: `A local image model guessed the photo might show ${guess}. Make one fun, 3-minute observation quest someone can do outdoors. Treat the guess as uncertain.` }
    ];
    const output = await gemma(messages, { max_new_tokens: 64, do_sample: false });
    result.textContent = output[0].generated_text.at(-1).content;
    button.textContent = '✳ Make another field quest';
  } catch (error) {
    console.error(error);
    result.textContent = error.message || 'Gemma could not load. Try again with a recent WebGPU-capable browser.';
    button.textContent = 'Try Gemma again';
  } finally {
    button.disabled = false;
  }
}

async function identify() {
  setBusy(true);
  loadingTitle.textContent = 'Warming up the model…';
  loadingMessage.textContent = 'The open-weight model is loading into your browser. The first load may take a little while.';
  try {
    if (!classifier) {
      classifier = await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32', {
        progress_callback: (progress) => {
          if (progress.status === 'progress' && progress.total) {
            const percent = Math.round(progress.progress);
            loadingTitle.textContent = 'Getting the model ready…';
            loadingMessage.textContent = `Downloading model files: ${percent}%. This happens once in this browser.`;
          }
        }
      });
    }
    loadingTitle.textContent = 'Looking closely…';
    loadingMessage.textContent = 'The model is comparing visual patterns in your photo.';
    renderPredictions(await classifier(preview, plantLabels));
  } catch (error) {
    console.error(error);
    loading.hidden = true;
    welcome.hidden = false;
    answer.hidden = true;
    statusDot.style.background = '#c27b60';
    welcome.innerHTML = '<div class="sun-icon">↻</div><h2>Couldn’t load<br>the model.</h2><p>Check your connection for the first download, then try again. Your photo stays in this browser.</p><button class="upload-button" id="retryButton">Try again <span>↻</span></button><div class="model-tag" style="margin-top:16px">LOCAL INFERENCE · OPEN WEIGHTS</div>';
    document.querySelector('#retryButton').addEventListener('click', identify);
  }
}

input.addEventListener('change', () => {
  const file = input.files?.[0];
  if (!file) return;
  if (activeUrl) URL.revokeObjectURL(activeUrl);
  activeUrl = URL.createObjectURL(file);
  preview.src = activeUrl;
  preview.hidden = false;
  emptyState.hidden = true;
  changePhoto.hidden = false;
  showWelcome();
  identify();
});

changePhoto.addEventListener('click', () => input.click());
const dropzone = document.querySelector('#dropzone');
dropzone.addEventListener('dragover', event => { event.preventDefault(); dropzone.classList.add('dragging'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
dropzone.addEventListener('drop', event => {
  event.preventDefault();
  dropzone.classList.remove('dragging');
  const file = event.dataTransfer.files?.[0];
  if (file?.type.startsWith('image/')) {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    input.dispatchEvent(new Event('change'));
  }
});
