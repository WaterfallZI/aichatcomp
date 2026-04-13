/* ═══════════════════════════════════════════════════════════════════════
   AI CHAT COMPANY — LANDING PAGE JAVASCRIPT
   Complete interactive functionality for index.html
═══════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Constants ──────────────────────────────────────────────────────────
const GROQ_API_URL = 'https://openrouter.ai/api/v1/chat/completions';
const MAX_DEMO_HISTORY = 20;
const DEMO_HISTORY_KEY = 'aichat_demo_history';
const COLOR_KEY = 'aichat_color';

const COLORS = [
  { name: 'Emerald',  c: '#10a37f', c2: '#1a7f64', rgb: '16,163,127' },
  { name: 'Blue',     c: '#3b82f6', c2: '#2563eb', rgb: '59,130,246' },
  { name: 'Violet',   c: '#7c3aed', c2: '#6d28d9', rgb: '124,58,237' },
  { name: 'Rose',     c: '#f43f5e', c2: '#e11d48', rgb: '244,63,94' },
  { name: 'Orange',   c: '#f97316', c2: '#ea580c', rgb: '249,115,22' },
  { name: 'Cyan',     c: '#06b6d4', c2: '#0891b2', rgb: '6,182,212' },
  { name: 'Indigo',   c: '#6366f1', c2: '#4f46e5', rgb: '99,102,241' },
  { name: 'Pink',     c: '#ec4899', c2: '#db2777', rgb: '236,72,153' },
  { name: 'Teal',     c: '#14b8a6', c2: '#0d9488', rgb: '20,184,166' },
  { name: 'Amber',    c: '#f59e0b', c2: '#d97706', rgb: '245,158,11' },
];

// ── State ──────────────────────────────────────────────────────────────
let groqKey = null;
let demoHistory = [];
let demoTyping = false;
let currentUser = null;
let waterDropInterval = null;

// ── DOM Ready ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initColorPicker();
  initHeader();
  initScrollReveal();
  initSmoothScroll();
  initMobileMenu();
  initStatsCounter();
  initFoxAnimation();
  initDemoChat();
  checkAuth();
  loadDemoHistory();
});

// ── Auth Check ─────────────────────────────────────────────────────────
async function checkAuth() {
  try {
    const res = await fetch('/api/auth/me', { credentials: 'include' });
    if (!res.ok) return;
    const user = await res.json();
    currentUser = user;
    updateHeaderForUser(user);
  } catch (e) {
    // Not logged in, that's fine
  }
}

function updateHeaderForUser(user) {
  const actions = document.querySelector('.header-actions');
  if (!actions) return;

  const total = user.total_credits === -1 ? '∞' : user.total_credits;
  actions.innerHTML = `
    <div class="header-credits">
      <span class="header-credits-icon">⚡</span>
      <span>${total} credits</span>
    </div>
    <a href="/chat.html" class="btn-signup">Open Chat →</a>
  `;
}

// ── Header Scroll Effect ───────────────────────────────────────────────
function initHeader() {
  const header = document.querySelector('.header');
  if (!header) return;

  const onScroll = () => {
    header.classList.toggle('scrolled', window.scrollY > 10);
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

// ── Scroll Reveal ──────────────────────────────────────────────────────
function initScrollReveal() {
  const elements = document.querySelectorAll('.reveal');
  if (!elements.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  elements.forEach(el => observer.observe(el));
}

// ── Color Picker ───────────────────────────────────────────────────────
function initColorPicker() {
  const picker = document.querySelector('.color-picker');
  if (!picker) return;

  const toggle = picker.querySelector('.color-picker-toggle');
  const panel = picker.querySelector('.color-picker-panel');
  const swatchesEl = picker.querySelector('.color-swatches');

  // Build swatches
  if (swatchesEl) {
    swatchesEl.innerHTML = COLORS.map((c, i) => `
      <button class="swatch" data-index="${i}" title="${c.name}"
        style="background:${c.c}; color:${c.c};"
        aria-label="Theme color: ${c.name}"></button>
    `).join('');

    swatchesEl.querySelectorAll('.swatch').forEach(sw => {
      sw.addEventListener('click', () => {
        const idx = parseInt(sw.dataset.index);
        applyColor(COLORS[idx]);
        swatchesEl.querySelectorAll('.swatch').forEach(s => s.classList.remove('active'));
        sw.classList.add('active');
        localStorage.setItem(COLOR_KEY, idx);
      });
    });
  }

  // Toggle panel
  if (toggle && panel) {
    toggle.addEventListener('click', (e) => {
      e.stopPropagation();
      panel.classList.toggle('open');
    });
    document.addEventListener('click', (e) => {
      if (!picker.contains(e.target)) panel.classList.remove('open');
    });
  }

  // Restore saved color
  const saved = localStorage.getItem(COLOR_KEY);
  if (saved !== null) {
    const idx = parseInt(saved);
    if (COLORS[idx]) {
      applyColor(COLORS[idx]);
      const sw = swatchesEl?.querySelector(`[data-index="${idx}"]`);
      if (sw) sw.classList.add('active');
    }
  } else {
    // Mark default active
    const sw = swatchesEl?.querySelector('[data-index="0"]');
    if (sw) sw.classList.add('active');
  }
}

function applyColor(color) {
  const root = document.documentElement;
  root.style.setProperty('--c', color.c);
  root.style.setProperty('--c2', color.c2);
  root.style.setProperty('--c-rgb', color.rgb);
}

// ── Fox Water Gun Animation ────────────────────────────────────────────
function initFoxAnimation() {
  const foxWrap = document.querySelector('.fox-wrap');
  if (!foxWrap) return;

  const dropsContainer = foxWrap.querySelector('.fox-water-drops');

  foxWrap.addEventListener('mouseenter', () => {
    startWaterDrops(dropsContainer);
  });

  foxWrap.addEventListener('mouseleave', () => {
    stopWaterDrops();
  });

  // 3D tilt on mouse move
  foxWrap.addEventListener('mousemove', (e) => {
    const rect = foxWrap.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = (e.clientX - cx) / (rect.width / 2);
    const dy = (e.clientY - cy) / (rect.height / 2);
    const fox3d = foxWrap.querySelector('.fox-3d');
    if (fox3d) {
      fox3d.style.transform = `scale(1.06) rotateY(${dx * -12}deg) rotateX(${dy * 6}deg)`;
    }
  });

  foxWrap.addEventListener('mouseleave', () => {
    const fox3d = foxWrap.querySelector('.fox-3d');
    if (fox3d) fox3d.style.transform = '';
  });
}

function startWaterDrops(container) {
  if (!container) return;
  stopWaterDrops();

  const spawnDrop = () => {
    const drop = document.createElement('div');
    drop.className = 'water-drop';
    const angle = (Math.random() * 60 - 30) * (Math.PI / 180);
    const dist = 80 + Math.random() * 80;
    const tx = Math.cos(angle) * dist;
    const ty = -Math.sin(angle) * dist + (Math.random() * 40 - 20);
    drop.style.setProperty('--tx', `${tx}px`);
    drop.style.setProperty('--ty', `${ty}px`);
    drop.style.left = `${Math.random() * 10}px`;
    drop.style.top = `${Math.random() * 10}px`;
    drop.style.width = `${6 + Math.random() * 6}px`;
    drop.style.height = `${8 + Math.random() * 8}px`;
    container.appendChild(drop);
    setTimeout(() => drop.remove(), 1000);
  };

  spawnDrop();
  waterDropInterval = setInterval(spawnDrop, 120);
}

function stopWaterDrops() {
  if (waterDropInterval) {
    clearInterval(waterDropInterval);
    waterDropInterval = null;
  }
}

// ── Demo Chat ──────────────────────────────────────────────────────────
function initDemoChat() {
  const input = document.getElementById('demo-input');
  const sendBtn = document.getElementById('demo-send');
  const messagesEl = document.getElementById('demo-messages');
  const modelSel = document.getElementById('demo-model');

  if (!input || !sendBtn || !messagesEl) return;

  // Load Groq key
  loadGroqKey();

  // Send on button click
  sendBtn.addEventListener('click', () => sendDemoMessage());

  // Send on Enter
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendDemoMessage();
    }
  });

  // Auto-resize (if textarea)
  if (input.tagName === 'TEXTAREA') {
    input.addEventListener('input', () => autoResize(input));
  }

  // Image paste support
  input.addEventListener('paste', handleDemoPaste);

  // Render existing history
  renderDemoHistory();
}

async function loadGroqKey() {
  try {
    const res = await fetch('/api/config/groq-key');
    if (res.ok) {
      const data = await res.json();
      groqKey = data.key;
    }
  } catch (e) {
    console.warn('Could not load Groq key:', e);
  }
}

function loadDemoHistory() {
  try {
    const saved = localStorage.getItem(DEMO_HISTORY_KEY);
    if (saved) demoHistory = JSON.parse(saved);
  } catch (e) {
    demoHistory = [];
  }
}

function saveDemoHistory() {
  try {
    // Keep only last MAX_DEMO_HISTORY messages
    if (demoHistory.length > MAX_DEMO_HISTORY) {
      demoHistory = demoHistory.slice(-MAX_DEMO_HISTORY);
    }
    localStorage.setItem(DEMO_HISTORY_KEY, JSON.stringify(demoHistory));
  } catch (e) {}
}

function renderDemoHistory() {
  const messagesEl = document.getElementById('demo-messages');
  if (!messagesEl || !demoHistory.length) return;

  // Show last 6 messages
  const recent = demoHistory.slice(-6);
  recent.forEach(msg => {
    appendDemoMessage(msg.role, msg.content, false);
  });
  scrollDemoToBottom();
}

async function sendDemoMessage() {
  if (demoTyping) return;

  const input = document.getElementById('demo-input');
  const sendBtn = document.getElementById('demo-send');
  const modelSel = document.getElementById('demo-model');

  if (!input) return;
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  if (input.tagName === 'TEXTAREA') autoResize(input);

  // Add user message
  appendDemoMessage('user', text);
  demoHistory.push({ role: 'user', content: text });
  saveDemoHistory();

  // Show typing
  demoTyping = true;
  if (sendBtn) sendBtn.disabled = true;
  const typingEl = showDemoTyping();

  const model = modelSel?.value || 'llama-3.3-70b-versatile';

  try {
    let reply = '';

    if (groqKey) {
      try {
        reply = await callGroqStreaming(model, buildDemoMessages(), typingEl);
      } catch (groqErr) {
        // Groq failed — fallback to server
        removeDemoTyping(typingEl);
        const newTyping = showDemoTyping();
        reply = await callServerChat(model, buildDemoMessages());
        removeDemoTyping(newTyping);
      }
    } else {
      // Fallback to server
      reply = await callServerChat(model, buildDemoMessages());
    }

    removeDemoTyping(typingEl);
    appendDemoMessage('ai', reply);
    demoHistory.push({ role: 'assistant', content: reply });
    saveDemoHistory();

  } catch (err) {
    removeDemoTyping(typingEl);
    const errMsg = getErrorMessage(err);
    appendDemoMessage('ai', `⚠️ ${errMsg}`);
  } finally {
    demoTyping = false;
    if (sendBtn) sendBtn.disabled = false;
    input.focus();
  }
}

function buildDemoMessages() {
  const systemMsg = {
    role: 'system',
    content: 'You are a helpful, friendly AI assistant. Be concise and clear. Use markdown for code blocks when relevant.'
  };
  const history = demoHistory.slice(-MAX_DEMO_HISTORY).map(m => ({
    role: m.role === 'ai' ? 'assistant' : m.role,
    content: m.content
  }));
  return [systemMsg, ...history];
}

async function callGroqStreaming(model, messages, typingEl) {
  let res;
  try {
    res = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${groqKey}`, 'Content-Type': 'application/json', 'HTTP-Referer': location.origin, 'X-Title': 'AI Chat Company' },
      body: JSON.stringify({ model, messages, max_tokens: 1024, temperature: 0.8, stream: true })
    });
  } catch (e) {
    throw new Error('groq_network');
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error?.message || `Groq error ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let fullText = '';

  // Replace typing indicator with streaming bubble
  const messagesEl = document.getElementById('demo-messages');
  let streamBubble = null;

  if (typingEl && messagesEl) {
    typingEl.remove();
    const msgEl = createDemoMsgEl('ai', '');
    messagesEl.appendChild(msgEl);
    streamBubble = msgEl.querySelector('.demo-bubble');
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') break;

      try {
        const parsed = JSON.parse(data);
        const delta = parsed.choices?.[0]?.delta?.content || '';
        if (delta) {
          fullText += delta;
          if (streamBubble) {
            streamBubble.textContent = fullText;
            scrollDemoToBottom();
          }
        }
      } catch (e) {}
    }
  }

  return fullText;
}

async function callServerChat(model, messages) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ model, messages })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    if (err.error === 'no_credits') throw new Error('No credits remaining. Please sign up for more.');
    if (err.error === 'Unauthorized') throw new Error('Please sign in to continue chatting.');
    throw new Error(err.error || `Server error ${res.status}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content || 'No response received.';
}

function appendDemoMessage(role, content, animate = true) {
  const messagesEl = document.getElementById('demo-messages');
  if (!messagesEl) return;

  const msgEl = createDemoMsgEl(role, content, animate);
  messagesEl.appendChild(msgEl);
  scrollDemoToBottom();
}

function createDemoMsgEl(role, content, animate = true) {
  const isUser = role === 'user';
  const div = document.createElement('div');
  div.className = `demo-msg ${isUser ? 'user' : 'ai'}`;
  if (!animate) div.style.animation = 'none';

  div.innerHTML = `
    <div class="demo-av ${isUser ? 'user-av' : 'ai-av'}">${isUser ? 'U' : 'AI'}</div>
    <div class="demo-bubble">${escapeHtml(content)}</div>
  `;
  return div;
}

function showDemoTyping() {
  const messagesEl = document.getElementById('demo-messages');
  if (!messagesEl) return null;

  const div = document.createElement('div');
  div.className = 'demo-msg ai demo-typing-row';
  div.innerHTML = `
    <div class="demo-av ai-av">AI</div>
    <div class="demo-bubble">
      <div class="demo-typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  messagesEl.appendChild(div);
  scrollDemoToBottom();
  return div;
}

function removeDemoTyping(el) {
  if (el && el.parentNode) el.remove();
}

function scrollDemoToBottom() {
  const messagesEl = document.getElementById('demo-messages');
  if (messagesEl) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

function handleDemoPaste(e) {
  const items = e.clipboardData?.items;
  if (!items) return;

  for (const item of items) {
    if (item.type.startsWith('image/')) {
      e.preventDefault();
      const file = item.getAsFile();
      const reader = new FileReader();
      reader.onload = (ev) => {
        const input = document.getElementById('demo-input');
        if (input) input.value = `[Image attached] ${input.value}`;
      };
      reader.readAsDataURL(file);
      break;
    }
  }
}

// ── Stats Counter Animation ────────────────────────────────────────────
function initStatsCounter() {
  const stats = document.querySelectorAll('.hero-stat-num[data-target]');
  if (!stats.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  stats.forEach(el => observer.observe(el));
}

function animateCounter(el) {
  const target = parseFloat(el.dataset.target);
  const suffix = el.dataset.suffix || '';
  const prefix = el.dataset.prefix || '';
  const duration = 1800;
  const start = performance.now();

  const update = (now) => {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = target * eased;

    if (target >= 1000) {
      el.textContent = prefix + Math.floor(current).toLocaleString() + suffix;
    } else if (Number.isInteger(target)) {
      el.textContent = prefix + Math.floor(current) + suffix;
    } else {
      el.textContent = prefix + current.toFixed(1) + suffix;
    }

    if (progress < 1) requestAnimationFrame(update);
    else el.textContent = prefix + (Number.isInteger(target) ? target.toLocaleString() : target) + suffix;
  };

  requestAnimationFrame(update);
}

// ── Smooth Scroll ──────────────────────────────────────────────────────
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', (e) => {
      const id = link.getAttribute('href').slice(1);
      const target = document.getElementById(id);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

// ── Mobile Menu ────────────────────────────────────────────────────────
function initMobileMenu() {
  const btn = document.querySelector('.mobile-menu-btn');
  const nav = document.querySelector('.mobile-nav');
  if (!btn || !nav) return;

  btn.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    btn.classList.toggle('open', open);
    btn.setAttribute('aria-expanded', open);
  });

  // Close on nav link click
  nav.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      nav.classList.remove('open');
      btn.classList.remove('open');
    });
  });

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (!btn.contains(e.target) && !nav.contains(e.target)) {
      nav.classList.remove('open');
      btn.classList.remove('open');
    }
  });
}

// ── Auto-resize textarea ───────────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

// ── Error Messages ─────────────────────────────────────────────────────
function getErrorMessage(err) {
  const msg = err?.message || String(err);
  if (msg === 'groq_network' || msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
    return 'AI temporarily unavailable. Please try again.';
  }
  if (msg.includes('rate_limit') || msg.includes('429')) {
    return 'Rate limit reached. Please wait a moment and try again.';
  }
  if (msg.includes('401') || msg.includes('Unauthorized')) {
    return 'Authentication error. Please sign in.';
  }
  return msg || 'Something went wrong. Please try again.';
}

// ── Utility: Escape HTML ───────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ── Toast Notifications ────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 3000) {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span><span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Keyboard Shortcuts ─────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  // Escape closes color picker
  if (e.key === 'Escape') {
    document.querySelector('.color-picker-panel')?.classList.remove('open');
    document.querySelector('.mobile-nav')?.classList.remove('open');
    document.querySelector('.mobile-menu-btn')?.classList.remove('open');
  }
});

// ── Page Load Animation ────────────────────────────────────────────────
window.addEventListener('load', () => {
  document.body.style.opacity = '0';
  document.body.style.transition = 'opacity 0.3s ease';
  requestAnimationFrame(() => {
    document.body.style.opacity = '1';
  });
});

// ── Additional Utilities & Enhancements ───────────────────────────────

// Debounce helper
function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

// Throttle helper
function throttle(fn, limit) {
  let inThrottle;
  return (...args) => {
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

// Format numbers with K/M suffix
function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return String(n);
}

// Copy text to clipboard with fallback
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (e) {
    // Fallback for older browsers
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  }
}

// Check if element is in viewport
function isInViewport(el) {
  const rect = el.getBoundingClientRect();
  return rect.top < window.innerHeight && rect.bottom > 0;
}

// Get relative time string
function timeAgo(date) {
  const seconds = Math.floor((Date.now() - new Date(date)) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

// Sanitize user input for display
function sanitize(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// Local storage helpers with error handling
const storage = {
  get(key, fallback = null) {
    try { const v = localStorage.getItem(key); return v !== null ? JSON.parse(v) : fallback; }
    catch (e) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; }
    catch (e) { return false; }
  },
  remove(key) {
    try { localStorage.removeItem(key); } catch (e) {}
  }
};

// Detect mobile device
const isMobile = () => window.innerWidth <= 600 || /Mobi|Android/i.test(navigator.userAgent);

// Prefers reduced motion
const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Apply reduced motion if needed
if (prefersReducedMotion()) {
  document.documentElement.style.setProperty('--transition', '0s');
  document.documentElement.style.setProperty('--transition-slow', '0s');
}

// Expose key functions globally for HTML onclick handlers
window.showToast = showToast;
window.copyToClipboard = copyToClipboard;
