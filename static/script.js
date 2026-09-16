// static/script.js
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('input');
const sendBtn    = document.getElementById('sendBtn');
const statusEl   = document.getElementById('status');
const resetBtn   = document.getElementById('resetBtn');

let busy = false;

/* ---------- 消息渲染 ---------- */
function appendMessage(role, content) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '👤' : '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  const contentEl = document.createElement('div');
  contentEl.className = 'content';
  contentEl.textContent = content;

  bubble.appendChild(contentEl);
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();

  return { wrap, bubble, contentEl };
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ---------- 状态栏 ---------- */
async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const d = await res.json();
    const limit = d.per_session_limit ? ` / ${d.per_session_limit}` : '';
    statusEl.textContent =
      `🪙 会话 ${d.session_tokens}${limit}  ·  累计 ${d.total_tokens}`;
  } catch (_) { /* 忽略 */ }
}

/* ---------- 发送 ---------- */
function setBusy(v) {
  busy = v;
  sendBtn.disabled = v;
  sendBtn.textContent = v ? '…' : '发送';
}

async function send() {
  const text = inputEl.value.trim();
  if (!text || busy) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';

  appendMessage('user', text);

  const pending = appendMessage('assistant', '正在思考…');
  pending.wrap.classList.add('pending');
  setBusy(true);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      pending.contentEl.textContent = `⚠️ ${data.detail || '请求失败'}`;
      pending.wrap.classList.remove('pending');
      pending.wrap.classList.add('error');
    } else {
      pending.contentEl.textContent = data.answer || '（无内容）';
      pending.wrap.classList.remove('pending');

      if (data.blocked) {
        pending.wrap.classList.add('blocked');
      } else if (data.elapsed) {
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent =
          `⏱ ${data.elapsed.toFixed(2)}s  ·  🪙 ${data.total_tokens} tokens`;
        pending.bubble.appendChild(meta);
      }
    }
  } catch (e) {
    pending.contentEl.textContent = `⚠️ 网络错误: ${e.message}`;
    pending.wrap.classList.remove('pending');
    pending.wrap.classList.add('error');
  } finally {
    setBusy(false);
    refreshStatus();
    inputEl.focus();
  }
}

/* ---------- 重置 ---------- */
async function resetSession() {
  if (!confirm('确定重置当前会话的 Token 预算吗？')) return;
  try {
    await fetch('/api/reset', { method: 'POST' });
    appendMessage('assistant', '♻️ 会话 Token 预算已重置。');
    refreshStatus();
  } catch (e) {
    alert('重置失败: ' + e.message);
  }
}

/* ---------- 事件绑定 ---------- */
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
});

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

sendBtn.addEventListener('click', send);
resetBtn.addEventListener('click', resetSession);

/* ---------- 启动 ---------- */
refreshStatus();
inputEl.focus();