/* static/js/main3.js */

// Emotion classes
const EMOTIONS = ["depression","anxiety","frustration","calmness"];

/* ---------------- DOM refs ---------------- */
const messagesEl = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const transcriptHint = document.getElementById('transcriptHint');
const liveEmotionFeedback = document.getElementById('liveEmotionFeedback');
const sendBtn = document.getElementById('sendBtn');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const exportBtn = document.getElementById('exportBtn');
const voiceToggle = document.getElementById('voiceToggle');

const micRecordBtn = document.getElementById('micRecordBtn');
const micStatus = document.getElementById('micStatus');

const presenceGlow = document.getElementById('presenceGlow');
const statusText = document.getElementById('statusText');
const dominantEmotionEl = document.getElementById('dominantEmotion');
const dominantConfidence = document.getElementById('dominantConfidence');

const depressionBar = document.getElementById('depression-bar');
const anxietyBar = document.getElementById('anxiety-bar');
const frustrationBar = document.getElementById('frustration-bar');
const calmnessBar = document.getElementById('calmness-bar');
const depressionPct = document.getElementById('depression-percent');
const anxietyPct = document.getElementById('anxiety-percent');
const frustrationPct = document.getElementById('frustration-percent');
const calmnessPct = document.getElementById('calmness-percent');

const analysisStatus = document.getElementById('analysisStatus');
const totalFramesEl = document.getElementById('totalFrames');
const avgAccuracyEl = document.getElementById('avgAccuracy');
const lastResponseTimeEl = document.getElementById('lastResponseTime');
const recentReplies = document.getElementById('recentReplies');

/* ---------------- state ---------------- */
let isDetecting = false;
let pollHandle = null;
let frameCount = 0;
let totalConfidence = 0;

// Live text analysis state
let liveAnalysisTimeout = null;
let isAnalyzing = false;
let currentLiveEmotion = 'calmness';
let lastAnalyzedText = '';
let analysisRequestId = 0;

// TTS support & configuration
const speechEnabled = 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined';
let voiceEnabled = speechEnabled;
let availableVoices = [];
let preferredVoice = null;

// STT support
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition || null;
const sttSupported = !!SpeechRecognition;
let recognition = null;
let isRecording = false;

// voice parameter map per emotion
const EMOTION_VOICE_MAP = {
  depression:  { rate: 0.82, pitch: 0.75, volume: 0.9 },
  anxiety:     { rate: 0.9,  pitch: 0.92, volume: 0.95 },
  frustration: { rate: 1.05, pitch: 1.02, volume: 0.98 },
  calmness:    { rate: 0.98, pitch: 1.05, volume: 1.0 }
};

/* ---------------- utilities ---------------- */
const el = (tag, cls='') => { const e=document.createElement(tag); if(cls) e.className = cls; return e; };
const sleep = (ms) => new Promise(res => setTimeout(res, ms));
const clamp = (v,min=0,max=1)=> Math.max(min, Math.min(max, v));

/* ---------------- LIVE TEXT ANALYSIS SYSTEM ---------------- */

// Enhanced live emotion analysis while typing
function analyzeLiveText() {
  const text = userInput.value.trim();
  
  // Skip if text hasn't changed significantly or too short
  if (!text || text === lastAnalyzedText || text.length < 5) {
    return;
  }
  
  // Skip if already analyzing same text
  if (isAnalyzing && text === lastAnalyzedText) {
    return;
  }
  
  lastAnalyzedText = text;
  isAnalyzing = true;
  analysisRequestId++;
  const currentRequestId = analysisRequestId;
  
  // Update UI to show analyzing state
  updateAnalysisUI('analyzing', 'Analyzing...');
  userInput.classList.add('analyzing');
  
  // Add shimmer effect to bars
  addShimmerEffect();
  
  const analysisStartTime = performance.now();
  
  fetch('/analyze_text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })
  .then(response => response.json())
  .then(data => {
    // Check if this is still the current request
    if (currentRequestId !== analysisRequestId) {
      return; 
    }
    
    if (data && data.success && data.probabilities) {
      const analysisTime = Math.round(performance.now() - analysisStartTime);
      
      // Update emotion bars with smooth animation
      updateBarsAnimated(data.probabilities);
      
      // Determine dominant emotion - EMOTIONS is now accessible here
      const maxIdx = data.probabilities.indexOf(Math.max(...data.probabilities));
      const dominant = EMOTIONS[maxIdx];
      const confidence = Math.round(Math.max(...data.probabilities) * 100);
      
      // Update dominant emotion display
      dominantEmotionEl.textContent = dominant;
      dominantConfidence.textContent = `${confidence}%`;
      
      // Update current live emotion
      currentLiveEmotion = dominant;
      
      // Show live emotion feedback
      showLiveFeedback(dominant, confidence);
      
      // Update analysis status
      updateAnalysisUI('ready', `${dominant} • ${confidence}% • ${analysisTime}ms`);
      
      lastResponseTimeEl.textContent = `${analysisTime}ms`;
      
      // Apply live emotion theming
      applyLiveEmotionTheme(dominant, confidence);
      
    } else {
      updateAnalysisUI('error', 'Analysis failed');
      console.warn('Live analysis failed:', data?.message || 'Unknown error');
    }
  })
  .catch(error => {
    // Check if this is still the current request
    if (currentRequestId !== analysisRequestId) {
      return;
    }
    
    updateAnalysisUI('error', 'Analysis error');
    console.warn('Live analysis error:', error);
  })
  .finally(() => {
    // Check if this is still the current request
    if (currentRequestId === analysisRequestId) {
      isAnalyzing = false;
      userInput.classList.remove('analyzing');
      removeShimmerEffect();
    }
  });
}

// Debounced live analysis - triggers while typing
userInput.addEventListener('input', () => {
  clearTimeout(liveAnalysisTimeout);
  
  const text = userInput.value.trim();
  
  // Clear feedback if text is too short
  if (text.length < 5) {
    hideLiveFeedback();
    updateAnalysisUI('ready', 'Ready to analyze...');
    return;
  }
  
  // Show that we're about to analyze
  if (text.length >= 5) {
    updateAnalysisUI('analyzing', 'Preparing analysis...');
  }
  
  // Debounce the actual analysis call
  liveAnalysisTimeout = setTimeout(() => {
    console.log('Triggering live analysis for text:', text.substring(0, 50) + '...');
    analyzeLiveText();
  }, 500); // 500ms debounce
  
  // Auto-resize textarea
  setTimeout(() => { 
    if (userInput) {
      userInput.style.height = 'auto'; 
      userInput.style.height = `${userInput.scrollHeight}px`; 
    }
  }, 0);
});

// UI update functions for live analysis
function updateAnalysisUI(state, message) {
  if (!analysisStatus) return;
  
  analysisStatus.className = `analysis-status ${state}`;
  analysisStatus.textContent = message;
}

function showLiveFeedback(emotion, confidence) {
  if (!liveEmotionFeedback) return;
  
  liveEmotionFeedback.className = `live-emotion-feedback show ${emotion}`;
  liveEmotionFeedback.textContent = `${emotion.charAt(0).toUpperCase() + emotion.slice(1)} ${confidence}%`;
  
  // Auto-hide after 3 seconds
  setTimeout(() => {
    hideLiveFeedback();
  }, 3000);
}

function hideLiveFeedback() {
  if (liveEmotionFeedback) {
    liveEmotionFeedback.classList.remove('show');
  }
}

// Apply live emotion theme while preserving clean input field appearance
function applyLiveEmotionTheme(emotion, confidence) {
  // Keep theme colors and glow effects for overall app theming
  const root = document.documentElement;
  const strength = clamp(confidence, 20, 100) / 100;
  
  if (emotion === 'depression') {
    root.style.setProperty('--accent1', '#ff6b6b');
    root.style.setProperty('--accent2', '#ff8e8e');
    presenceGlow.classList.remove('hidden');
    presenceGlow.style.boxShadow = `0 0 ${40 * strength}px rgba(255,107,107,${0.18 * strength})`;
  } else if (emotion === 'anxiety') {
    root.style.setProperty('--accent1', '#3ddbd9');
    root.style.setProperty('--accent2', '#1fa2a6');
    presenceGlow.classList.remove('hidden');
    presenceGlow.style.boxShadow = `0 0 ${34 * strength}px rgba(61,219,217,${0.18 * strength})`;
  } else if (emotion === 'frustration') {
    root.style.setProperty('--accent1', '#f7c548');
    root.style.setProperty('--accent2', '#ff9f6b');
    presenceGlow.classList.remove('hidden');
    presenceGlow.style.boxShadow = `0 0 ${36 * strength}px rgba(247,197,72,${0.18 * strength})`;
  } else {
    root.style.setProperty('--accent1', '#6b8cff');
    root.style.setProperty('--accent2', '#00d4ff');
    presenceGlow.classList.add('hidden');
  }
}

function addShimmerEffect() {
  try {
    [depressionBar, anxietyBar, frustrationBar, calmnessBar].forEach(bar => {
      if (bar && bar.parentElement) {
        bar.parentElement.classList.add('live');
      }
    });
  } catch (e) {
    console.warn('Error adding shimmer effect:', e);
  }
}

function removeShimmerEffect() {
  try {
    [depressionBar, anxietyBar, frustrationBar, calmnessBar].forEach(bar => {
      if (bar && bar.parentElement) {
        bar.parentElement.classList.remove('live');
      }
    });
  } catch (e) {
    console.warn('Error removing shimmer effect:', e);
  }
}

function updateBarsAnimated(probs) {
  if (!probs || !Array.isArray(probs) || probs.length !== 4) {
    console.warn('Invalid probabilities array:', probs);
    return;
  }
  
  const percentages = probs.map(x => Math.round(x * 100));
  
  // Stagger the animations slightly for a more organic feel
  setTimeout(() => {
    if (depressionBar && depressionPct) {
      depressionBar.style.width = `${percentages[0]}%`;
      depressionPct.textContent = `${percentages[0]}%`;
    }
  }, 0);
  
  setTimeout(() => {
    if (anxietyBar && anxietyPct) {
      anxietyBar.style.width = `${percentages[1]}%`;
      anxietyPct.textContent = `${percentages[1]}%`;
    }
  }, 100);
  
  setTimeout(() => {
    if (frustrationBar && frustrationPct) {
      frustrationBar.style.width = `${percentages[2]}%`;
      frustrationPct.textContent = `${percentages[2]}%`;
    }
  }, 200);
  
  setTimeout(() => {
    if (calmnessBar && calmnessPct) {
      calmnessBar.style.width = `${percentages[3]}%`;
      calmnessPct.textContent = `${percentages[3]}%`;
    }
  }, 300);
}

/* ---------------- load voices ---------------- */
function loadVoices() {
  if (!speechEnabled) return;
  availableVoices = speechSynthesis.getVoices();
  preferredVoice = availableVoices.find(v => /en-?us|english/i.test(v.lang) && /female|female/i.test(v.name)) || availableVoices.find(v => /en-?us|english/i.test(v.lang)) || availableVoices[0];
}
if (speechEnabled) {
  loadVoices();
  window.speechSynthesis.onvoiceschanged = loadVoices;
}

/* ---------------- message rendering ---------------- */
function pushMessage({role='assistant', meta=null}) {
  const node = el('div','msg '+role);
  const bubble = el('div','bubble');
  const content = el('div','content');
  bubble.appendChild(content);
  if (role === 'assistant') {
    const header = el('div','assistant-header'); header.innerHTML = `<strong>GenAI</strong> <span class="micro">Assistant</span>`;
    bubble.prepend(header);
  }
  node.appendChild(bubble);

  const controlRow = el('div','bubble-controls');
  if (role === 'assistant' && speechEnabled) {
    const playBtn = el('button','play'); playBtn.title='Play/Replay'; playBtn.textContent='▶';
    const pauseBtn = el('button','pause'); pauseBtn.title='Pause/Resume'; pauseBtn.textContent='⏸';
    const stopBtn = el('button','stop'); stopBtn.title='Stop'; stopBtn.textContent='⏹';
    controlRow.appendChild(playBtn); controlRow.appendChild(pauseBtn); controlRow.appendChild(stopBtn);
    node._playBtn = playBtn; node._pauseBtn = pauseBtn; node._stopBtn = stopBtn;
  }

  // Add emotion tag for user messages only (without percentage)
  if (role === 'user' && meta && meta.emotion) {
    const tag = el('div','tag'); 
    tag.textContent = meta.emotion.charAt(0).toUpperCase() + meta.emotion.slice(1);
    controlRow.appendChild(tag);
  }

  node.appendChild(controlRow);
  messagesEl.appendChild(node);
  node.scrollIntoView({behavior:'smooth', block:'end'});
  return {node, content};
}

async function revealText(contentEl, text, speed=7) {
  contentEl.textContent = '';
  for (let i=0;i<text.length;i++){
    contentEl.textContent += text[i];
    messagesEl.scrollTop = messagesEl.scrollHeight;
    await sleep(speed + Math.random()*6);
  }
}

function createUtterance(text, emotion='calmness') {
  if (!speechEnabled) return null;
  const u = new SpeechSynthesisUtterance(text);
  const params = EMOTION_VOICE_MAP[emotion] || EMOTION_VOICE_MAP.calmness;
  u.rate = clamp(params.rate,0.5,2.0);
  u.pitch = clamp(params.pitch,0.1,2.0);
  u.volume = clamp(params.volume,0.0,1.0);
  if (preferredVoice) u.voice = preferredVoice;
  u.lang = preferredVoice ? preferredVoice.lang : 'en-US';
  return u;
}

function attachPlaybackControls(node, utterance) {
  if (!speechEnabled || !utterance || !node) return;
  const playBtn = node._playBtn;
  const pauseBtn = node._pauseBtn;
  const stopBtn = node._stopBtn;

  playBtn.onclick = () => {
    try { speechSynthesis.cancel(); speechSynthesis.speak(utterance); } catch(e){ console.warn(e); }
  };
  pauseBtn.onclick = () => {
    try { if (speechSynthesis.speaking && !speechSynthesis.paused) speechSynthesis.pause(); else if (speechSynthesis.paused) speechSynthesis.resume(); } catch(e) { console.warn(e); }
  };
  stopBtn.onclick = () => { try { speechSynthesis.cancel(); } catch(e){ console.warn(e); } };
}

function addRecent(text) {
  const b = el('button','recent');
  b.textContent = text.length>60 ? text.slice(0,60)+'…' : text;
  b.onclick = () => { userInput.value = text; userInput.focus(); };
  recentReplies.prepend(b);
  while (recentReplies.children.length > 6) recentReplies.lastChild.remove();
}

function appendUser(text, detectedEmotion = null, confidence = null) {
  // Use passed emotion or fall back to current live emotion
  const emotion = detectedEmotion || currentLiveEmotion || 'calmness';
  const conf = confidence || parseInt((dominantConfidence.textContent||'0').replace('%','')) || 0;
  
  // Only show emotion tag if confidence is above threshold and not default calmness
  const showEmotionTag = conf > 30 && emotion !== 'calmness';
  
  const meta = showEmotionTag ? { emotion, conf } : null;
  const { node, content } = pushMessage({ role: 'user', meta });
  content.textContent = text;
  node.scrollIntoView({behavior:'smooth', block:'end'});
}

/* ---------------- core send flow ---------------- */
async function sendMessage(text) {
  if (!text || !text.trim()) return;
  
  // Get the current live emotion BEFORE appending user message
  const currentEmotion = currentLiveEmotion || 'calmness';
  const currentConfidence = parseInt((dominantConfidence.textContent||'0').replace('%','')) || 0;
  
  appendUser(text);
  userInput.value = ''; 
  userInput.style.height = 'auto';
  
  // Clear live analysis state
  hideLiveFeedback();
  updateAnalysisUI('ready', 'Ready to analyze...');

  // update_text (fusion) - use the analyzed emotion from live analysis
  try { 
    fetch('/update_text',{ 
      method:'POST', 
      headers:{'Content-Type':'application/json'}, 
      body: JSON.stringify({ text })
    }); 
  } catch(e){ console.warn('update_text failed', e); }

  // Use current live emotion for tagging and response generation
  const metaForTag = { 
    emotion: currentEmotion, 
    conf: currentConfidence 
  };
  
  const { node, content } = pushMessage({ role:'assistant', meta: metaForTag });

  const typing = el('div','typing'); 
  typing.innerHTML = '<span></span><span></span><span></span>';
  node.querySelector('.bubble').appendChild(typing);

  const t0 = performance.now();
  let reply = '...';
  let detectedEmotion = currentEmotion; // Use the live-detected emotion
  
  try {
    // Send both the text AND the current detected emotion to the backend
    const res = await fetch('/generate_response', {
      method: 'POST', 
      headers: {'Content-Type':'application/json'}, 
      body: JSON.stringify({ 
        text: text,
        detected_emotion: currentEmotion, // Pass the current live emotion
        confidence: currentConfidence
      })
    });
    const j = await res.json();
    if (j && j.success) { 
      reply = j.reply || ''; 
      detectedEmotion = j.emotion || currentEmotion; // Use the emotion we sent
    }
    else reply = j.message || 'Sorry, could not generate a reply.';
  } catch (e) { 
    console.error(e); 
    reply = 'Network error – backend unreachable.'; 
  }

  // prepare utterance and speak immediately (read-as-it-types)
  let utt = null;
  if (voiceEnabled && speechEnabled) {
    utt = createUtterance(reply, detectedEmotion);
    if (utt) { setTimeout(()=> { try { speechSynthesis.speak(utt); } catch(e){ console.warn('speak failed', e); } }, 30); }
  }

  typing.remove();
  await revealText(content, reply, 6);

  if (utt) attachPlaybackControls(node, utt);
  if (reply && reply.length>0) addRecent(reply);
  lastResponseTimeEl.textContent = `${Math.round(performance.now() - t0)} ms`;
}

/* ---------------- emotion polling (for face detection) ---------------- */
async function pollEmotion() {
  const t0 = performance.now();
  try {
    const res = await fetch('/get_emotion_data');
    const j = await res.json();
    if (j && j.success && j.probabilities) {
      const p = j.probabilities;
      // Only update bars if not currently doing live text analysis
      if (!isAnalyzing) {
        updateBars(p);
      }
      const maxIdx = p.indexOf(Math.max(...p));
      const dominant = EMOTIONS[maxIdx];
      const conf = Math.round(Math.max(...p) * 100);
      
      // Only update if not overridden by live text analysis
      if (!isAnalyzing && userInput.value.trim().length < 5) {
        dominantEmotionEl.textContent = dominant;
        dominantConfidence.textContent = `${conf}%`;
        themeByEmotion(dominant, conf);
      }
      
      statusText.textContent = j.face_detected ? 'Face detected' : 'No face';
      frameCount++; totalConfidence += conf;
      totalFramesEl.textContent = frameCount;
      avgAccuracyEl.textContent = `${Math.round(totalConfidence/frameCount)}%`;
    }
  } catch (e) { console.warn('poll error', e); }
  if (!isAnalyzing) {
    lastResponseTimeEl.textContent = `${Math.round(performance.now() - t0)} ms`;
  }
}

function updateBars(probs) {
  if (!probs || !Array.isArray(probs) || probs.length !== 4) {
    console.warn('Invalid probabilities in updateBars:', probs);
    return;
  }
  
  try {
    const p = probs.map(x => Math.round(x * 100));
    
    if (depressionBar && depressionPct) {
      depressionBar.style.width = `${p[0]}%`; 
      depressionPct.textContent = `${p[0]}%`;
    }
    if (anxietyBar && anxietyPct) {
      anxietyBar.style.width = `${p[1]}%`; 
      anxietyPct.textContent = `${p[1]}%`;
    }
    if (frustrationBar && frustrationPct) {
      frustrationBar.style.width = `${p[2]}%`; 
      frustrationPct.textContent = `${p[2]}%`;
    }
    if (calmnessBar && calmnessPct) {
      calmnessBar.style.width = `${p[3]}%`; 
      calmnessPct.textContent = `${p[3]}%`;
    }
  } catch (e) {
    console.warn('Error updating bars:', e);
  }
}

function themeByEmotion(emotion, conf) {
  const root = document.documentElement;
  const strength = clamp(conf,20,100)/100;
  if (emotion === 'depression') {
    root.style.setProperty('--accent1','#ff6b6b'); root.style.setProperty('--accent2','#ff8e8e');
    presenceGlow.classList.remove('hidden'); presenceGlow.style.boxShadow = `0 0 ${40*strength}px rgba(255,107,107,${0.18*strength})`;
  } else if (emotion === 'anxiety') {
    root.style.setProperty('--accent1','#3ddbd9'); root.style.setProperty('--accent2','#1fa2a6');
    presenceGlow.classList.remove('hidden'); presenceGlow.style.boxShadow = `0 0 ${34*strength}px rgba(61,219,217,${0.18*strength})`;
  } else if (emotion === 'frustration') {
    root.style.setProperty('--accent1','#f7c548'); root.style.setProperty('--accent2','#ff9f6b');
    presenceGlow.classList.remove('hidden'); presenceGlow.style.boxShadow = `0 0 ${36*strength}px rgba(247,197,72,${0.18*strength})`;
  } else {
    root.style.setProperty('--accent1','#6b8cff'); root.style.setProperty('--accent2','#00d4ff');
    presenceGlow.classList.add('hidden');
  }
}

/* ---------------- controls wiring ---------------- */
sendBtn.addEventListener('click', ()=> sendMessage(userInput.value));

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { 
    e.preventDefault(); 
    sendMessage(userInput.value); 
  }
  if (e.key === '/' && document.activeElement !== userInput) { 
    e.preventDefault(); 
    userInput.focus(); 
  }
});

document.querySelectorAll('.chip').forEach(c => c.addEventListener('click', e => {
  userInput.value = e.target.textContent; 
  // Trigger live analysis for chip content
  analyzeLiveText();
  sendMessage(userInput.value);
}));

document.querySelectorAll('.quick').forEach(q => q.addEventListener('click', e => {
  userInput.value = q.dataset.text; 
  // Trigger live analysis for quick action content
  analyzeLiveText();
  sendMessage(userInput.value);
}));

startBtn.addEventListener('click', async () => {
  if (isDetecting) return;
  startBtn.disabled = true; startBtn.textContent = 'Starting…';
  try {
    const r = await fetch('/start_detection', { method:'POST' });
    const j = await r.json();
    if (j.success) { 
      isDetecting = true; 
      pollHandle = setInterval(pollEmotion, 350); 
      startBtn.textContent = 'Running'; 
    }
    else { 
      alert('Could not start: '+(j.message||'unknown')); 
      startBtn.textContent = 'Start'; 
    }
  } catch(e) { 
    console.error(e); 
    startBtn.textContent = 'Start'; 
  }
  startBtn.disabled = false;
});

stopBtn.addEventListener('click', async () => {
  if (!isDetecting) return;
  try { 
    await fetch('/stop_detection', { method:'POST' }); 
  } catch(e){ 
    console.warn(e); 
  }
  isDetecting = false; 
  clearInterval(pollHandle); 
  startBtn.textContent = 'Start'; 
  statusText.textContent = 'Stopped';
});

exportBtn.addEventListener('click', () => {
  const nodes = Array.from(messagesEl.querySelectorAll('.msg'));
  const lines = nodes.map(n => {
    const role = n.classList.contains('user') ? 'You' : n.classList.contains('assistant') ? 'Assistant' : 'System';
    const txt = n.querySelector('.content') ? n.querySelector('.content').textContent : n.textContent;
    return `${role}: ${txt}`;
  });
  const blob = new Blob([lines.join('\n\n')], {type:'text/plain'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `genai-${Date.now()}.txt`; a.click();
  URL.revokeObjectURL(url);
});

clearBtn.addEventListener('click', ()=> { 
  messagesEl.innerHTML = '<div class="system-card">Conversation cleared.</div>'; 
  recentReplies.innerHTML = ''; 
  // Reset live analysis state
  hideLiveFeedback();
  updateAnalysisUI('ready', 'Ready to analyze...');
  currentLiveEmotion = 'calmness';
});

if (!speechEnabled) {
  voiceToggle.title = 'Voice not supported by this browser';
  voiceToggle.style.opacity = 0.5;
} else {
  voiceToggle.addEventListener('click', ()=> {
    voiceEnabled = !voiceEnabled; 
    voiceToggle.textContent = voiceEnabled ? '🔊' : '🔈';
    if (!voiceEnabled) speechSynthesis.cancel();
  });
}

/* ---------------- Speech-to-Text (Mic) ---------------- */

function initRecognition() {
  if (!sttSupported) {
    micRecordBtn.title = 'Speech-to-text not supported in this browser';
    micRecordBtn.classList.add('unsupported');
    return;
  }

  recognition = new SpeechRecognition();
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  recognition.lang = 'en-US';
  recognition.continuous = false; // stop on silence

  let interimTranscript = '';

  recognition.onstart = () => {
    isRecording = true;
    micRecordBtn.classList.add('recording');
    micStatus.textContent = 'Listening…';
    transcriptHint.style.opacity = 1; transcriptHint.setAttribute('aria-hidden','false');
  };

  recognition.onresult = (event) => {
    interimTranscript = '';
    let finalTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      const res = event.results[i];
      if (res.isFinal) finalTranscript += res[0].transcript;
      else interimTranscript += res[0].transcript;
    }

    // show interim transcript in hint and live into input
    transcriptHint.textContent = interimTranscript ? interimTranscript : '';
    if (interimTranscript) {
      userInput.value = interimTranscript;
      // Trigger live analysis for speech input
      analyzeLiveText();
    } else if (finalTranscript) {
      userInput.value = finalTranscript;
      transcriptHint.textContent = '';
      // auto-stop visuals
      stopRecognition();
      // Trigger final live analysis and auto-send
      analyzeLiveText();
      setTimeout(() => sendMessage(finalTranscript), 500);
    }
  };

  recognition.onerror = (evt) => {
    console.warn('Speech recognition error', evt);
    stopRecognition();
  };

  recognition.onend = () => {
    // ensure UI shows stopped
    isRecording = false;
    micRecordBtn.classList.remove('recording');
    micStatus.textContent = 'Start';
    transcriptHint.style.opacity = 0; transcriptHint.setAttribute('aria-hidden','true');
  };
}

function startRecognition() {
  if (!sttSupported || !recognition) return;
  try {
    recognition.start();
  } catch (e) {
    console.warn('recognition start error', e);
  }
}

function stopRecognition() {
  if (!sttSupported || !recognition) return;
  try { recognition.stop(); } catch(e) { console.warn(e); }
  isRecording = false;
  micRecordBtn.classList.remove('recording');
  micStatus.textContent = 'Start';
  transcriptHint.style.opacity = 0; transcriptHint.setAttribute('aria-hidden','true');
}

// mic button wiring
micRecordBtn.addEventListener('click', () => {
  if (!sttSupported) {
    alert('Speech-to-text not supported in this browser. Try Chrome or Edge.');
    return;
  }
  if (isRecording) {
    stopRecognition();
  } else {
    startRecognition();
  }
});

// initialize recognition on load
if (sttSupported) initRecognition();

/* ---------------- initialization & greetings ---------------- */
setTimeout(()=> updateBars([0.25,0.25,0.25,0.25]), 200);
setTimeout(()=> {
  const greeting = "Hi – I can analyze your emotions live as you type and listen to your voice. Try typing something!";
  (async ()=> {
    const meta = { emotion: 'calmness', conf: 90 };
    const { node, content } = pushMessage({ role: 'assistant', meta });
    let utt = null;
    if (voiceEnabled && speechEnabled) {
      utt = createUtterance(greeting, 'calmness');
      if (utt) { setTimeout(()=> speechSynthesis.speak(utt), 50); }
      attachPlaybackControls(node, utt);
    }
    await revealText(content, greeting, 6);
    addRecent(greeting);
  })();
}, 450);

// keyboard shortcut to focus input
document.addEventListener('keydown', (e)=> {
  if (e.key === '/' && document.activeElement !== userInput) { 
    e.preventDefault(); 
    userInput.focus(); 
  }
});