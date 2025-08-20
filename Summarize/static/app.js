document.addEventListener('DOMContentLoaded', () => {
  // THEME TOGGLE
  const root = document.documentElement;
  const toggleInput = document.getElementById('theme-toggle');
  const themeMeta = document.querySelector('meta[name="theme-color"]');

  const themeColors = {
    dark: '#121a35',
    light: '#eaf1ff'
  };

  const applyTheme = (mode) => {
    root.setAttribute('data-theme', mode);
    if (themeMeta) themeMeta.setAttribute('content', mode === 'dark' ? themeColors.dark : themeColors.light);
    try { localStorage.setItem('theme', mode); } catch (_) {}
  };

  let initial = 'dark';
  try {
    const stored = localStorage.getItem('theme');
    if (stored === 'light' || stored === 'dark') initial = stored;
    else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) initial = 'light';
  } catch (_) {}
  applyTheme(initial);
  if (toggleInput) toggleInput.checked = initial === 'light';

  if (toggleInput) {
    toggleInput.addEventListener('change', () => {
      const mode = toggleInput.checked ? 'light' : 'dark';
      applyTheme(mode);
    });
  }

  const form = document.getElementById('summary-form');
  const overlay = document.getElementById('loading-overlay');
  const textarea = document.getElementById('text');
  const counter = document.getElementById('char-count');
  const copyBtn = document.getElementById('copy-btn');
  const summaryText = document.getElementById('summary-text');

  if (form && overlay) {
    form.addEventListener('submit', () => {
      overlay.classList.remove('hidden');
    });
  }

  if (textarea && counter) {
    const update = () => {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 600) + 'px';
      counter.textContent = `${textarea.value.length} characters`;
    };
    textarea.addEventListener('input', update);
    update();
  }

  if (copyBtn && summaryText) {
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(summaryText.innerText);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => (copyBtn.textContent = 'Copy'), 1200);
      } catch (_) {
        copyBtn.textContent = 'Copy failed';
        setTimeout(() => (copyBtn.textContent = 'Copy'), 1200);
      }
    });
  }
});



