/**
 * AI Chat Company — Cartoon SVG Icon Library
 * All icons are hand-drawn style SVG, no emojis
 */
const ICONS = {

  // ── AI Robot ──────────────────────────────────────────────
  robot: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="10" y="16" width="28" height="22" rx="6" fill="#10a37f" stroke="#0d8a6b" stroke-width="2"/>
    <rect x="16" y="22" width="6" height="6" rx="2" fill="white"/>
    <rect x="26" y="22" width="6" height="6" rx="2" fill="white"/>
    <circle cx="19" cy="25" r="2" fill="#0d8a6b"/>
    <circle cx="29" cy="25" r="2" fill="#0d8a6b"/>
    <path d="M18 32 Q24 36 30 32" stroke="white" stroke-width="2" stroke-linecap="round" fill="none"/>
    <rect x="21" y="10" width="6" height="8" rx="2" fill="#10a37f" stroke="#0d8a6b" stroke-width="2"/>
    <circle cx="24" cy="9" r="3" fill="#fbbf24" stroke="#f59e0b" stroke-width="1.5"/>
    <rect x="4" y="22" width="6" height="10" rx="3" fill="#10a37f" stroke="#0d8a6b" stroke-width="2"/>
    <rect x="38" y="22" width="6" height="10" rx="3" fill="#10a37f" stroke="#0d8a6b" stroke-width="2"/>
  </svg>`,

  // ── Chat bubble ───────────────────────────────────────────
  chat: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="4" y="6" width="34" height="26" rx="8" fill="#10a37f" stroke="#0d8a6b" stroke-width="2"/>
    <path d="M10 36 L6 44 L18 38" fill="#10a37f" stroke="#0d8a6b" stroke-width="2" stroke-linejoin="round"/>
    <rect x="10" y="14" width="22" height="3" rx="1.5" fill="white" opacity="0.9"/>
    <rect x="10" y="21" width="16" height="3" rx="1.5" fill="white" opacity="0.7"/>
  </svg>`,

  // ── Star / Credits ────────────────────────────────────────
  star: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M24 4 L28.5 16.5 L42 16.5 L31.5 24.5 L35.5 37 L24 29.5 L12.5 37 L16.5 24.5 L6 16.5 L19.5 16.5 Z" fill="#fbbf24" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round"/>
    <circle cx="24" cy="20" r="4" fill="#fff" opacity="0.4"/>
  </svg>`,

  // ── Lightning / Fast ──────────────────────────────────────
  lightning: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M28 4 L14 26 L22 26 L20 44 L34 22 L26 22 Z" fill="#fbbf24" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round"/>
  </svg>`,

  // ── Code / Developer ──────────────────────────────────────
  code: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="4" y="8" width="40" height="32" rx="6" fill="#1e1e2e" stroke="#3b82f6" stroke-width="2"/>
    <circle cx="12" cy="16" r="2.5" fill="#ff5f57"/>
    <circle cx="20" cy="16" r="2.5" fill="#febc2e"/>
    <circle cx="28" cy="16" r="2.5" fill="#28c840"/>
    <path d="M16 26 L10 30 L16 34" stroke="#60a5fa" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M32 26 L38 30 L32 34" stroke="#60a5fa" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M26 22 L22 38" stroke="#a78bfa" stroke-width="2.5" stroke-linecap="round"/>
  </svg>`,

  // ── Pencil / Write ────────────────────────────────────────
  pencil: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M34 6 L42 14 L16 40 L6 42 L8 32 Z" fill="#fbbf24" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round"/>
    <path d="M30 10 L38 18" stroke="#f59e0b" stroke-width="2"/>
    <path d="M8 32 L16 40" stroke="#f59e0b" stroke-width="2"/>
    <path d="M6 42 L10 38" stroke="#d97706" stroke-width="2" stroke-linecap="round"/>
  </svg>`,

  // ── Chart / Analytics ─────────────────────────────────────
  chart: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="4" y="4" width="40" height="40" rx="8" fill="#eff6ff" stroke="#3b82f6" stroke-width="2"/>
    <rect x="10" y="28" width="6" height="12" rx="2" fill="#3b82f6"/>
    <rect x="20" y="20" width="6" height="20" rx="2" fill="#60a5fa"/>
    <rect x="30" y="12" width="6" height="28" rx="2" fill="#93c5fd"/>
    <path d="M10 22 L20 16 L30 10 L38 8" stroke="#1d4ed8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="10" cy="22" r="2.5" fill="#1d4ed8"/>
    <circle cx="20" cy="16" r="2.5" fill="#1d4ed8"/>
    <circle cx="30" cy="10" r="2.5" fill="#1d4ed8"/>
  </svg>`,

  // ── Teacher / Book ────────────────────────────────────────
  book: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M8 8 Q8 6 10 6 L24 6 L24 42 L10 42 Q8 42 8 40 Z" fill="#10a37f" stroke="#0d8a6b" stroke-width="2"/>
    <path d="M24 6 L38 6 Q40 6 40 8 L40 40 Q40 42 38 42 L24 42 Z" fill="#34d399" stroke="#0d8a6b" stroke-width="2"/>
    <path d="M24 6 L24 42" stroke="#0d8a6b" stroke-width="2"/>
    <rect x="12" y="14" width="8" height="2" rx="1" fill="white" opacity="0.8"/>
    <rect x="12" y="20" width="8" height="2" rx="1" fill="white" opacity="0.6"/>
    <rect x="28" y="14" width="8" height="2" rx="1" fill="white" opacity="0.8"/>
    <rect x="28" y="20" width="8" height="2" rx="1" fill="white" opacity="0.6"/>
  </svg>`,

  // ── Globe / Web ───────────────────────────────────────────
  globe: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="24" cy="24" r="20" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
    <ellipse cx="24" cy="24" rx="8" ry="20" fill="none" stroke="#3b82f6" stroke-width="2"/>
    <path d="M4 24 L44 24" stroke="#3b82f6" stroke-width="2"/>
    <path d="M8 14 Q24 18 40 14" stroke="#3b82f6" stroke-width="1.5" fill="none"/>
    <path d="M8 34 Q24 30 40 34" stroke="#3b82f6" stroke-width="1.5" fill="none"/>
  </svg>`,

  // ── Microphone / Voice ────────────────────────────────────
  mic: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="16" y="4" width="16" height="24" rx="8" fill="#a855f7" stroke="#7c3aed" stroke-width="2"/>
    <path d="M8 24 Q8 38 24 38 Q40 38 40 24" stroke="#7c3aed" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <path d="M24 38 L24 44" stroke="#7c3aed" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M18 44 L30 44" stroke="#7c3aed" stroke-width="2.5" stroke-linecap="round"/>
    <rect x="20" y="10" width="8" height="2" rx="1" fill="white" opacity="0.6"/>
    <rect x="20" y="16" width="8" height="2" rx="1" fill="white" opacity="0.6"/>
  </svg>`,

  // ── Image / Picture ───────────────────────────────────────
  image: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="4" y="8" width="40" height="32" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="2"/>
    <circle cx="16" cy="18" r="5" fill="#fbbf24" stroke="#f59e0b" stroke-width="1.5"/>
    <path d="M4 32 L14 22 L22 30 L30 20 L44 34 L44 38 Q44 40 42 40 L6 40 Q4 40 4 38 Z" fill="#fbbf24" opacity="0.7"/>
  </svg>`,

  // ── Key / Auth ────────────────────────────────────────────
  key: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="18" cy="20" r="12" fill="#fef3c7" stroke="#f59e0b" stroke-width="2.5"/>
    <circle cx="18" cy="20" r="6" fill="#fbbf24" stroke="#f59e0b" stroke-width="2"/>
    <path d="M26 26 L44 44" stroke="#f59e0b" stroke-width="3" stroke-linecap="round"/>
    <rect x="36" y="38" width="6" height="4" rx="1" fill="#f59e0b" transform="rotate(-45 36 38)"/>
    <rect x="30" y="32" width="6" height="4" rx="1" fill="#f59e0b" transform="rotate(-45 30 32)"/>
  </svg>`,

  // ── Shield / Security ─────────────────────────────────────
  shield: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M24 4 L40 10 L40 26 Q40 38 24 44 Q8 38 8 26 L8 10 Z" fill="#dcfce7" stroke="#10a37f" stroke-width="2.5" stroke-linejoin="round"/>
    <path d="M16 24 L21 29 L32 18" stroke="#10a37f" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`,

  // ── User / Person ─────────────────────────────────────────
  user: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="24" cy="16" r="10" fill="#10a37f" stroke="#0d8a6b" stroke-width="2"/>
    <path d="M6 44 Q6 32 24 32 Q42 32 42 44" fill="#10a37f" stroke="#0d8a6b" stroke-width="2" stroke-linejoin="round"/>
    <circle cx="20" cy="14" r="2" fill="white" opacity="0.6"/>
  </svg>`,

  // ── Send arrow ────────────────────────────────────────────
  send: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M22 2 L11 13" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M22 2 L15 22 L11 13 L2 9 Z" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
  </svg>`,

  // ── Trash / Delete ────────────────────────────────────────
  trash: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M3 6 L21 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <path d="M8 6 L8 4 Q8 2 10 2 L14 2 Q16 2 16 4 L16 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <path d="M19 6 L18 20 Q18 22 16 22 L8 22 Q6 22 6 20 L5 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M10 11 L10 17" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <path d="M14 11 L14 17" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  </svg>`,

  // ── Settings gear ─────────────────────────────────────────
  settings: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/>
    <path d="M12 2 L12 5 M12 19 L12 22 M2 12 L5 12 M19 12 L22 12 M4.9 4.9 L7.1 7.1 M16.9 16.9 L19.1 19.1 M19.1 4.9 L16.9 7.1 M7.1 16.9 L4.9 19.1" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  </svg>`,

  // ── Home ──────────────────────────────────────────────────
  home: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M3 12 L12 3 L21 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M5 10 L5 21 Q5 22 6 22 L10 22 L10 16 L14 16 L14 22 L18 22 Q19 22 19 21 L19 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`,

  // ── Plus / New ────────────────────────────────────────────
  plus: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 5 L12 19 M5 12 L19 12" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
  </svg>`,

  // ── Cartoon fox mascot ────────────────────────────────────
  fox: `<svg viewBox="0 0 120 140" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- Body -->
    <ellipse cx="60" cy="100" rx="30" ry="32" fill="#ff8c42"/>
    <!-- Belly -->
    <ellipse cx="60" cy="108" rx="18" ry="20" fill="#ffe0c0"/>
    <!-- Head -->
    <ellipse cx="60" cy="58" rx="28" ry="26" fill="#ff8c42"/>
    <!-- Ears -->
    <path d="M36 40 L28 16 L48 34 Z" fill="#ff8c42" stroke="#e05a00" stroke-width="1.5"/>
    <path d="M84 40 L92 16 L72 34 Z" fill="#ff8c42" stroke="#e05a00" stroke-width="1.5"/>
    <path d="M38 38 L32 22 L46 34 Z" fill="#ffb380"/>
    <path d="M82 38 L88 22 L74 34 Z" fill="#ffb380"/>
    <!-- Eyes -->
    <circle cx="50" cy="56" r="7" fill="white"/>
    <circle cx="70" cy="56" r="7" fill="white"/>
    <circle cx="51" cy="57" r="4" fill="#1a1a2e"/>
    <circle cx="71" cy="57" r="4" fill="#1a1a2e"/>
    <circle cx="52" cy="55" r="1.5" fill="white"/>
    <circle cx="72" cy="55" r="1.5" fill="white"/>
    <!-- Nose -->
    <ellipse cx="60" cy="66" rx="5" ry="3.5" fill="#cc4400"/>
    <!-- Mouth -->
    <path d="M54 70 Q60 75 66 70" stroke="#cc4400" stroke-width="2" fill="none" stroke-linecap="round"/>
    <!-- Cheeks -->
    <ellipse cx="44" cy="64" rx="7" ry="4" fill="#ff6b6b" opacity="0.4"/>
    <ellipse cx="76" cy="64" rx="7" ry="4" fill="#ff6b6b" opacity="0.4"/>
    <!-- Tail -->
    <path d="M88 110 Q110 90 105 70 Q100 55 88 65" fill="#ff8c42" stroke="#e05a00" stroke-width="1.5"/>
    <path d="M95 68 Q108 60 104 72" fill="#fff5e0" stroke="#e05a00" stroke-width="1"/>
    <!-- Legs -->
    <rect x="42" y="126" width="14" height="16" rx="7" fill="#e05a00"/>
    <rect x="64" y="126" width="14" height="16" rx="7" fill="#e05a00"/>
  </svg>`,

  // ── Cartoon AI chip ───────────────────────────────────────
  chip: `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="12" y="12" width="24" height="24" rx="4" fill="#10a37f" stroke="#0d8a6b" stroke-width="2"/>
    <rect x="16" y="16" width="16" height="16" rx="2" fill="#0d8a6b"/>
    <path d="M18 20 L22 24 L18 28" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M30 20 L26 24 L30 28" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M8 18 L12 18 M8 24 L12 24 M8 30 L12 30" stroke="#10a37f" stroke-width="2" stroke-linecap="round"/>
    <path d="M36 18 L40 18 M36 24 L40 24 M36 30 L40 30" stroke="#10a37f" stroke-width="2" stroke-linecap="round"/>
    <path d="M18 8 L18 12 M24 8 L24 12 M30 8 L30 12" stroke="#10a37f" stroke-width="2" stroke-linecap="round"/>
    <path d="M18 36 L18 40 M24 36 L24 40 M30 36 L30 40" stroke="#10a37f" stroke-width="2" stroke-linecap="round"/>
  </svg>`,

};

// Helper to get icon as HTML string
function getIcon(name, size = 24, cls = '') {
  const svg = ICONS[name];
  if (!svg) return '';
  return svg.replace('<svg ', `<svg width="${size}" height="${size}" class="${cls}" `);
}

// Export for use
if (typeof module !== 'undefined') module.exports = { ICONS, getIcon };
