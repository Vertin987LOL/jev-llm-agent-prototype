const token = new URLSearchParams(location.hash.slice(1)).get('token');
history.replaceState(null, '', location.pathname);
const headers = {'X-Jev-Token': token || ''};
const elements = Object.fromEntries(['messages','trace','prompt','send','notice','activity','goal-status','goal-text','resume-goal','stop-goal','heartbeat-toggle','heartbeat-once','pending-input','pending-save','pending-clear','pending-text','interval','model-label','model-name','approval-modal','approval-tool','approval-args','approve','deny'].map(id => [id, document.getElementById(id)]));
let cursor = 0;
let mode = 'chat';
let waitingApproval = false;
let provider = 'ollama';

function text(tag, content, className) {
  const node = document.createElement(tag);
  node.textContent = content == null ? '' : String(content);
  if (className) node.className = className;
  return node;
}

function bubble(role, content, className) {
  document.querySelector('.welcome')?.remove();
  const node = text('article', '', 'bubble ' + className);
  node.append(text('span', role, 'role'), text('div', content));
  elements.messages.append(node);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function traceCard(kind, title, body, extra) {
  elements.trace.querySelector('.trace-empty')?.remove();
  const card = text('div', '', 'trace-item ' + kind);
  card.append(text('div', title, 'trace-title'), text('div', body, 'trace-body'));
  if (extra !== undefined) {
    const details = document.createElement('details');
    details.open = true;
    details.append(text('summary', '完整参数与结果'), text('pre', extra));
    card.append(details);
  }
  elements.trace.append(card);
  elements.trace.scrollTop = elements.trace.scrollHeight;
}

function render(event) {
  if (event.kind === 'user') return bubble(event.action === 'goal' ? '持续目标' : '你', event.text, 'user');
  if (event.kind === 'answer') return bubble('代理 · ' + ({completed:'完成',needs_input:'需要补充信息',budget:'达到步数上限',blocked:'已阻塞'}[event.status] || event.status), event.text, 'answer');
  if (event.kind === 'error') return bubble('运行错误', event.text, 'answer error');
  if (event.kind === 'status') return traceCard('status', '状态', event.text);
  if (event.kind === 'heartbeat') return traceCard('heartbeat', '心跳 · Jev ' + event.choice, event.thought);
  if (event.kind === 'approval') return traceCard('status', '等待确认 · ' + event.tool, JSON.stringify(event.args, null, 2));
  if (event.kind === 'approval_result') return traceCard('status', '人工确认', event.allowed ? '已允许本次调用' : '已拒绝本次调用');
  if (event.kind !== 'trace') return;
  const step = event.step;
  if (step.type === 'advisor') return traceCard('advisor', provider === 'deepseek' ? 'DeepSeek · 想法' : '本地模型 · 想法', step.text);
  if (step.type === 'decision') return traceCard('decision', 'Jev · 第 ' + step.turn + ' 步', step.choice + ' — ' + (step.description || ''));
  if (step.type === 'completion') return traceCard('completion', 'Jev · 完成核对', step.choice);
  if (step.type === 'feedback') return traceCard('status', '核对反馈', step.text);
  if (step.type === 'tier') return traceCard('decision', '任务档位', step.choice);
  if (step.type === 'selected_tool') return traceCard('tool', '工具 · ' + step.tool, step.description, JSON.stringify(step.schema, null, 2));
  if (step.type === 'draft') return traceCard('status', '工具草案', step.path);
  if (step.type === 'tool') {
    const call = step.event;
    return traceCard('tool', '执行 · ' + call.tool, call.name || '',
      '参数：\n' + JSON.stringify(call.args, null, 2) + '\n\n' + (call.error ? '错误：\n' + call.error : '结果：\n' + call.result));
  }
}

function update(state) {
  provider = state.provider;
  elements['model-label'].textContent = provider === 'deepseek' ? 'CLOUD MODEL' : 'LOCAL MODEL';
  elements.activity.classList.toggle('busy', state.busy);
  elements.activity.lastChild.textContent = state.busy ? ' 运行中' : ' 等待输入';
  elements.send.disabled = state.busy;
  elements['resume-goal'].disabled = state.busy || !state.goal || state.goal.status !== 'budget';
  elements['stop-goal'].disabled = state.busy || !state.goal;
  elements['goal-status'].textContent = state.goal ? ({running:'运行中',budget:'已暂停',blocked:'已阻塞',error:'出错'}[state.goal.status] || state.goal.status) : '尚无目标';
  elements['goal-text'].textContent = state.goal?.text || '说出任务，或先随便聊聊。';
  elements['heartbeat-toggle'].setAttribute('aria-pressed', String(state.heartbeat));
  elements['pending-text'].textContent = state.pending_task || '暂无待办';
  elements.interval.textContent = state.interval;
  elements['model-name'].textContent = state.model;
  elements['approval-modal'].hidden = !state.approval;
  if (state.approval) {
    elements['approval-tool'].textContent = state.approval.tool;
    elements['approval-args'].textContent = JSON.stringify(state.approval.args, null, 2);
  }
  waitingApproval = !!state.approval;
}

async function command(action, value = '') {
  elements.notice.textContent = '';
  try {
    const response = await fetch('/api/command', {method:'POST', headers:{...headers,'Content-Type':'application/json'}, body:JSON.stringify({action,text:value})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '操作失败');
    await poll();
  } catch (error) { elements.notice.textContent = error.message; }
}

async function poll() {
  const response = await fetch('/api/events?after=' + cursor, {headers});
  if (!response.ok) throw new Error('连接失败，请从启动器重新打开');
  const state = await response.json();
  update(state);
  for (const event of state.events) { cursor = event.id; render(event); }
}

elements.send.onclick = () => {
  const value = elements.prompt.value.trim();
  if (value) { elements.prompt.value = ''; command(mode, value); }
};
elements.prompt.onkeydown = event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); elements.send.click();
  }
};
document.querySelectorAll('[data-mode]').forEach(button => button.onclick = () => {
  mode = button.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach(item => item.classList.toggle('active', item === button));
  elements.prompt.placeholder = mode === 'goal' ? '描述要持续完成的任务…' : '说出想法，或者描述你希望完成的事…';
});
document.querySelectorAll('[data-prompt]').forEach(button => button.onclick = () => { elements.prompt.value = button.dataset.prompt; elements.prompt.focus(); });
elements['resume-goal'].onclick = () => command('resume');
elements['stop-goal'].onclick = () => command('stop_goal');
elements['heartbeat-toggle'].onclick = () => command(elements['heartbeat-toggle'].getAttribute('aria-pressed') === 'true' ? 'heartbeat_off' : 'heartbeat_on');
elements['heartbeat-once'].onclick = () => command('heartbeat_once');
elements['pending-save'].onclick = () => { const value = elements['pending-input'].value.trim(); if (value) { elements['pending-input'].value=''; command('pending', value); } };
elements['pending-clear'].onclick = () => command('clear_pending');
elements.approve.onclick = () => command('approve');
elements.deny.onclick = () => command('deny');
if (!token) elements.notice.textContent = '缺少本机会话凭据，请从启动器打开页面。';
else { poll().catch(error => elements.notice.textContent = error.message); setInterval(() => poll().catch(error => elements.notice.textContent = error.message), 550); }
