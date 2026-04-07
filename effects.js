// ── AI Chat Company — Visual Effects ─────────────────────────
// Scroll animations + parallax + cursor glow (NO dark overlay)

(function () {

// ── 1. Animated gradient mesh (updates with theme) ───────────
const mesh = document.createElement('div');
mesh.id = 'nc-mesh';
mesh.style.cssText = 'position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;';
mesh.innerHTML = `
    <div id="mesh-1" style="position:absolute;width:800px;height:800px;border-radius:50%;top:-200px;left:-200px;animation:meshFloat1 12s ease-in-out infinite;filter:blur(80px);transition:background 0.6s ease;"></div>
    <div id="mesh-2" style="position:absolute;width:600px;height:600px;border-radius:50%;bottom:-100px;right:-100px;animation:meshFloat2 15s ease-in-out infinite;filter:blur(100px);transition:background 0.6s ease;"></div>
    <div id="mesh-3" style="position:absolute;width:400px;height:400px;border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);animation:meshFloat3 10s ease-in-out infinite;filter:blur(80px);transition:background 0.6s ease;"></div>
`;
const styleEl = document.createElement('style');
styleEl.textContent = `
    @keyframes meshFloat1 {
        0%,100% { transform: translate(0,0) scale(1); }
        33%      { transform: translate(60px,-40px) scale(1.1); }
        66%      { transform: translate(-30px,30px) scale(0.9); }
    }
    @keyframes meshFloat2 {
        0%,100% { transform: translate(0,0) scale(1); }
        50%      { transform: translate(-50px,40px) scale(1.15); }
    }
    @keyframes meshFloat3 {
        0%,100% { transform: translate(-50%,-50%) scale(1); }
        50%      { transform: translate(-50%,-50%) scale(1.2); }
    }
`;
document.head.appendChild(styleEl);
document.body.prepend(mesh);

// ── 2. Scroll-triggered reveal with blur ──────────────────────
const revealObs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
        if (e.isIntersecting) {
            const delay = parseInt(e.target.dataset.delay || 0);
            setTimeout(() => {
                e.target.style.opacity = '1';
                e.target.style.transform = 'translateY(0) scale(1)';
                e.target.style.filter = 'blur(0px)';
            }, delay);
            revealObs.unobserve(e.target);
        }
    });
}, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

function initReveal() {
    const groups = [
        { sel: '.feature-card', stagger: 80 },
        { sel: '.product-card', stagger: 100 },
        { sel: '.tech-card',    stagger: 60 },
        { sel: '.value',        stagger: 80 },
        { sel: '.nomchat-text', stagger: 0 },
        { sel: '.nomchat-visual', stagger: 100 },
        { sel: '.cta-title',    stagger: 0 },
        { sel: '.cta-sub',      stagger: 80 },
        { sel: '.cta-actions',  stagger: 160 },
    ];

    groups.forEach(({ sel, stagger }) => {
        document.querySelectorAll(sel).forEach((el, i) => {
            Object.assign(el.style, {
                opacity: '0',
                transform: 'translateY(28px) scale(0.98)',
                filter: 'blur(6px)',
                transition: 'opacity 0.65s cubic-bezier(0.4,0,0.2,1), transform 0.65s cubic-bezier(0.4,0,0.2,1), filter 0.65s cubic-bezier(0.4,0,0.2,1)',
            });
            el.dataset.delay = i * stagger;
            revealObs.observe(el);
        });
    });

    // Hero elements animate on load
    const heroEls = ['.hero-eyebrow', '.hero-title', '.hero-sub', '.hero-actions', '.stats'];
    heroEls.forEach((sel, i) => {
        const el = document.querySelector(sel);
        if (!el) return;
        Object.assign(el.style, {
            opacity: '0',
            transform: 'translateY(20px)',
            filter: 'blur(4px)',
            transition: `opacity 0.7s ${i * 0.1}s cubic-bezier(0.4,0,0.2,1), transform 0.7s ${i * 0.1}s cubic-bezier(0.4,0,0.2,1), filter 0.7s ${i * 0.1}s cubic-bezier(0.4,0,0.2,1)`,
        });
        requestAnimationFrame(() => requestAnimationFrame(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
            el.style.filter = 'blur(0px)';
        }));
    });
}

// ── 3. Parallax ───────────────────────────────────────────────
function initParallax() {
    window.addEventListener('scroll', () => {
        const y = window.scrollY;
        const hero = document.querySelector('.hero-inner');
        if (hero) {
            hero.style.transform = `translateY(${y * 0.2}px)`;
            hero.style.opacity   = `${Math.max(0, 1 - y / 700)}`;
        }
        // Mesh orbs parallax
        const orbs = mesh.querySelectorAll('div');
        orbs.forEach((orb, i) => {
            const speed = [0.06, 0.1, 0.04][i] || 0.06;
            orb.style.marginTop = `${y * speed}px`;
        });
    }, { passive: true });
}

// ── 4. Cursor glow ────────────────────────────────────────────
function initCursorGlow() {
    const glow = document.createElement('div');
    Object.assign(glow.style, {
        position: 'fixed', pointerEvents: 'none', zIndex: '1',
        width: '500px', height: '500px', borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(16,163,127,0.05) 0%, transparent 65%)',
        transform: 'translate(-50%, -50%)',
        left: '-999px', top: '-999px',
        transition: 'left 1s cubic-bezier(0.4,0,0.2,1), top 1s cubic-bezier(0.4,0,0.2,1)',
        filter: 'blur(20px)',
    });
    document.body.appendChild(glow);

    window.addEventListener('mousemove', e => {
        glow.style.left = e.clientX + 'px';
        glow.style.top  = e.clientY + 'px';
    }, { passive: true });
}

// ── 5. Smooth anchor scroll ───────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
        const target = document.querySelector(a.getAttribute('href'));
        if (!target) return;
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
});

// ── 6. Header scroll ──────────────────────────────────────────
const header = document.querySelector('.header');
if (header) {
    window.addEventListener('scroll', () => {
        header.classList.toggle('scrolled', window.scrollY > 20);
    }, { passive: true });
}

// ── Init ──────────────────────────────────────────────────────
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { initReveal(); initParallax(); initCursorGlow(); });
} else {
    initReveal(); initParallax(); initCursorGlow();
}

})();
