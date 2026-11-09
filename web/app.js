const $ = (selector) => document.querySelector(selector);
const completionSound = new window.CompletionSound();
const serverLocalTimes = new Map();

function rememberLocalTimes(value) {
  if (!value || typeof value !== 'object') return;
  Object.values(value).forEach(rememberLocalTimes);
  if (Array.isArray(value)) return;
  for (const [key, local] of Object.entries(value)) {
    if (key.endsWith('_local') && typeof local === 'string' && typeof value[key.slice(0, -6)] === 'string') {
      serverLocalTimes.set(value[key.slice(0, -6)], local);
    }
  }
  while (serverLocalTimes.size > 3000) serverLocalTimes.delete(serverLocalTimes.keys().next().value);
}

function updateSoundButton() {
  const button = $("#toggleSound");
  button.setAttribute("aria-pressed", String(completionSound.enabled));
  button.setAttribute("aria-label", `完成提示音：${completionSound.enabled ? "已开启" : "已关闭"}`);
  button.title = `完成提示音：${completionSound.enabled ? "已开启" : "已关闭"}`;
  $("#soundState").textContent = completionSound.enabled ? "已开启" : "已关闭";
}

async function monitorCompletions() {
  try { completionSound.observe(await api("/api/runs?limit=100")); } catch (_) {}
  window.setTimeout(monitorCompletions, 3000);
}

const state = {
  analyses: [],
  current: null,
  pollTimer: null,
  currentRunId: null,
  lastRunStatus: null,
  runs: [],
  selectionEpoch: 0,
  reportEpoch: 0,
  submitting: false,
  currentConversationId: null,
  conversationBusy: false,
  runConfig: null,
  historyView: "active",
  historyEpoch: 0,
  currentArchived: false,
  currentTitle: null,
  menuConversation: null,
  dialogTarget: null,
};

const statusLabels = { queued: "排队中", running: "调查中", cancelling: "停止中", awaiting_approval: "等待审批", completed: "已完成", failed: "执行失败", cancelled: "已取消", interrupted: "已中断" };

function runStatusLabel(run) {
  const basic = statusLabels[run.status] || run.status;
  if (run.status !== 'completed') return basic;
  const outcome = run.investigation_outcome || run.investigation_progress?.outcome;
  return {needs_clarification:'待确认', budget_exhausted:'调查未完成',
    partial_result:'有限结果', insufficient_evidence:'调查未完成', needs_verification:'待核验'}[outcome] || basic;
}

const sentimentColors = {
  positive: "#555555",
  negative: "#999999",
  neutral: "#bbbbbb",
  mixed: "#dddddd",
};

const sentimentLabels = {
  positive: "正面",
  negative: "负面",
  neutral: "中性",
  mixed: "褒贬混合",
};

function percent(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function signed(value, suffix = "") {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${Number(value).toFixed(1)}${suffix}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2400);
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  const payload = await response.json();
  rememberLocalTimes(payload);
  return payload;
}

async function loadAnalyses() {
  const previous = $("#analysisSelect").value;
  state.analyses = await api("/api/analyses");
  const select = $("#analysisSelect");
  if (!state.analyses.length) {
    select.innerHTML = '<option value="">暂无分析结果</option>';
    return;
  }
  select.innerHTML = state.analyses.map((item) => {
    const date = localDateText(item.published_at_local || item.published_at);
    return `<option value="${item.appid}:${item.news_id}">${escapeHtml(item.title)} · ${date}</option>`;
  }).join("");
  if (state.analyses.some((item) => `${item.appid}:${item.news_id}` === previous)) select.value = previous;
}

function renderTopics(items) {
  const maxAbs = Math.max(10, ...items.map((item) => Math.abs(item.change_pp)));
  $("#topicChart").innerHTML = items.map((item, index) => {
    const positive = item.change_pp >= 0;
    const width = Math.max(2, Math.abs(item.change_pp) / maxAbs * 48);
    return `
      <div class="topic-row">
        <span class="topic-name">${escapeHtml(item.label)}</span>
        <div class="topic-track">
          <span class="topic-bar ${positive ? "positive" : "negative"}" style="width:${width}%;animation-delay:${index * 60}ms"></span>
        </div>
        <span class="topic-value ${positive ? "positive" : "negative"}">${signed(item.change_pp, " pp")}</span>
      </div>`;
  }).join("") || '<p class="trace-empty">没有可展示的主题变化。</p>';
}

function renderSentiment(counts) {
  const order = ["positive", "negative", "neutral", "mixed"];
  const actualTotal = order.reduce((sum, key) => sum + (counts[key] || 0), 0);
  const total = actualTotal || 1;
  let cursor = 0;
  const segments = order.map((key) => {
    const start = cursor;
    cursor += (counts[key] || 0) / total * 100;
    return `${sentimentColors[key]} ${start}% ${cursor}%`;
  });
  $("#sentimentDial").style.background = actualTotal ? `conic-gradient(${segments.join(",")})` : "#eeeeee";
  $("#positiveSampleRate").textContent = actualTotal ? percent((counts.positive || 0) / total, 0) : "—";
  $("#sentimentLegend").innerHTML = order.map((key) => `
    <div class="sentiment-item" style="--legend-color:${sentimentColors[key]}">
      <span>${sentimentLabels[key]}</span><strong>${counts[key] || 0}</strong>
    </div>`).join("");
}

function renderEvidence(items) {
  const display = items.length ? items : [];
  $("#evidenceGrid").innerHTML = display.map((item, index) => `
    <article class="evidence-card" data-index="${String(index + 1).padStart(2, "0")}">
      <div class="evidence-tags">
        ${item.topics.map((topic) => `<span class="tag">${escapeHtml(topic)}</span>`).join("")}
        ${item.actionable ? '<span class="tag hot">可执行</span>' : ""}
        ${item.update_related ? '<span class="tag hot">更新相关</span>' : ""}
      </div>
      <h4>${escapeHtml(item.summary)}</h4>
      <blockquote>“${escapeHtml(item.evidence)}”</blockquote>
      <footer class="evidence-footer">
        <span>SEVERITY ${item.severity}/5</span>
        <span>CONF ${Math.round(item.confidence * 100)}%</span>
      </footer>
      <span class="evidence-reference">REVIEW ID ${escapeHtml(item.review_id)} · ${item.votes_up || 0} 人认为有帮助</span>
    </article>`).join("") || '<p class="sample-caveat">没有可展示的更新后证据。</p>';
}

function renderDashboard(data) {
  $("#dashboard .metric-card:nth-child(1) .metric-label").textContent="已获取窗口 · 好评率变化";
  $("#dashboard .metric-card:nth-child(2) .metric-label").textContent="评论量变化";
  $("#dashboard .sentiment-panel").hidden=false;
  state.current = data;
  const game = data.game;
  const full = data.full_window;
  const sample = data.sample;
  $("#patchTitle").textContent = game.patch_title;
  $("#patchDate").textContent = localDateText(game.published_at_local || game.published_at, true);
  $("#patchLink").href = game.patch_url;
  $("#patchLink").hidden = false;
  $("#windowLabel").textContent = `± ${game.window_days} DAYS`;
  $("#positiveDelta").textContent = signed(full.positive_rate_change_pp, " pp");
  $("#positiveRates").textContent = `${percent(full.before_positive_rate)} → ${percent(full.after_positive_rate)}`;
  $("#volumeDelta").textContent = signed(full.review_count_change);
  $("#volumeCounts").textContent = `${full.before_total.toLocaleString()} → ${full.after_total.toLocaleString()} 条`;
  $("#severityValue").textContent = Number(sample.after_severity || 0).toFixed(2);
  $("#severityDelta").textContent = `${Number(sample.before_severity || 0).toFixed(2)} → ${Number(sample.after_severity || 0).toFixed(2)} / 5`;
  $("#relatedRate").textContent = percent(sample.after_update_related_rate, 0);
  $("#actionableCount").textContent = `${sample.after_actionable_count} 条可执行问题`;
  $("#reportLink").href = data.report_url;
  $("#reportLink").hidden = false;
  $("#downloadReport").href = `${data.report_url}?download=true`;
  $("#downloadReport").hidden = false;
  $("#reportContent").textContent = "正在读取完整报告…";
  renderTopics(data.topic_changes);
  renderSentiment(sample.after_sentiments);
  renderEvidence(data.evidence);
  const warningPanel = $("#warningPanel");
  warningPanel.hidden = !data.warnings.length;
  $("#warningList").innerHTML = data.warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

async function loadDashboard() {
  const selectValue = $("#analysisSelect").value;
  if (selectValue.startsWith("task:")) {
    const run=await api(`/api/runs/${selectValue.slice(5)}`);
    return loadTaskInvestigation(run);
  }
  let [appid, newsId] = selectValue ? selectValue.split(":") : [$("#appidInput").value.trim(), ""];
  if (!appid) throw new Error("请输入 AppID");
  const query = newsId ? `?news_id=${encodeURIComponent(newsId)}` : "";
  const epoch = ++state.reportEpoch;
  $("#dashboard").classList.add("loading");
  try {
    const data = await api(`/api/dashboard/${encodeURIComponent(appid)}${query}`);
    if (epoch !== state.reportEpoch) return;
    renderDashboard(data);
    const response = await fetch(data.report_url);
    if (!response.ok) throw new Error("完整报告读取失败");
    const report = await response.text();
    if (epoch === state.reportEpoch) $("#reportContent").textContent = report;
  } finally {
    if (epoch === state.reportEpoch) $("#dashboard").classList.remove("loading");
  }
}

function openDrawer() {
  $("#welcomePanel").hidden = true;
  $("#taskContent").hidden = false;
}

function resetTaskView() {
  $("#turnTiming").hidden = true;
  $("#llmStatus").hidden = true;
  $("#runPlan").innerHTML = "<li>等待规划器生成计划…</li>";
  $("#runEvents").innerHTML = '<p class="trace-empty">等待 Agent 选择第一个工具…</p>';
  $("#finalAnswerPanel").hidden = true;
  $("#approvalPanel").hidden = true;
  $("#cancelRun").hidden = true;
  $("#recoverRun").hidden = true;
  $("#runDrawer").classList.remove("running");
  $("#runMetrics").textContent = "选择任务后显示用量统计。";
}

function newInvestigation() {
  window.clearTimeout(state.pollTimer);
  state.selectionEpoch++;
  state.currentRunId = null;
  state.currentConversationId = null;
  state.conversationBusy = false;
  state.runConfig = null;
  state.currentArchived = false;
  state.currentTitle = null;
  rememberConversation(null);
  state.lastRunStatus = null;
  resetTaskView();
  $("#welcomePanel").hidden = false;
  $("#taskContent").hidden = true;
  $("#workspaceTitle").textContent = "游鉴 GameLens";
  $("#runStatus").textContent = "待开始";
  $("#goalInput").value = "";
  $("#goalInput").minLength = 5;
  $("#goalInput").placeholder = "输入你想了解的游戏问题";
  $("#composerNote").textContent = "发送开始新会话 · Enter 发送，Shift + Enter 换行";
  $("#conversationHistory").innerHTML = "";
  $("#appidInput").readOnly = false;
  $("#submitRun").disabled = state.submitting;
  $("#goalInput").focus();
  renderRunHistory();
  closeMobileSidebar();
}

function setInspector(open) {
  $("#dashboard").hidden = !open;
  $(".workbench").classList.toggle("inspector-open", open);
  $("#toggleInspector").setAttribute("aria-expanded", String(open));
}

function closeMobileSidebar() {
  $(".workbench").classList.remove("mobile-sidebar-open");
  $("#sidebarBackdrop").hidden = true;
  updateSidebarControl();
}

function updateSidebarControl() {
  const workbench = $(".workbench");
  const expanded = window.innerWidth <= 800 ? workbench.classList.contains("mobile-sidebar-open") : !workbench.classList.contains("sidebar-collapsed");
  $("#toggleSidebar").setAttribute("aria-expanded", String(expanded));
}

function toggleSidebar() {
  const workbench = $(".workbench");
  if (window.innerWidth <= 800) {
    const open = workbench.classList.toggle("mobile-sidebar-open");
    $("#sidebarBackdrop").hidden = !open;
    if (open) $("#newInvestigation").focus();
  } else workbench.classList.toggle("sidebar-collapsed");
  updateSidebarControl();
}

function switchTab(name) {
  document.querySelectorAll("[data-tab]").forEach((tab) => {
    const selected = tab.dataset.tab === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.querySelectorAll("[data-panel]").forEach((panel) => { panel.hidden = panel.dataset.panel !== name; });
}

function beijingDateParts(value) {
  if (!value) return null;
  value = serverLocalTimes.get(value) || value;
  // JavaScript otherwise interprets naive ISO using the browser's timezone.
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}(?:T[\d:.]+)?$/.test(value)) {
    value = `${value.includes("T") ? value : `${value}T00:00:00`}+08:00`;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return Object.fromEntries(new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(date).map(({type, value}) => [type, value]));
}

function shortTime(value) {
  const parts = beijingDateParts(value);
  return parts ? `${parts.month}/${parts.day} ${parts.hour}:${parts.minute}` : "—";
}

function localDateText(value, includeTime = false) {
  const parts = beijingDateParts(value);
  if (!parts) return "—";
  return `${parts.year}-${parts.month}-${parts.day}${includeTime ? ` ${parts.hour}:${parts.minute}` : ""}`;
}

async function loadRunHistory() {
  const epoch = ++state.historyEpoch;
  const conversations = await api(`/api/conversations?limit=100&view=${state.historyView}&q=${encodeURIComponent($("#historySearch").value.trim())}`);
  if (epoch !== state.historyEpoch) return;
  state.runs = conversations.map((item) => ({...item, run_id: item.last_run_id, goal: item.title }));
  renderRunHistory();
}

function renderRunHistory() {
  const query = $("#historySearch").value.trim().toLowerCase();
  const runs = state.runs.filter((run) => `${run.goal} ${run.appid}`.toLowerCase().includes(query));
  const container = $("#runHistory");
  container.innerHTML = runs.length
    ? runs.map((run) => `
      <div class="history-item">
        <button class="history-run ${run.conversation_id === state.currentConversationId ? "selected" : ""}" type="button" data-run-id="${escapeHtml(run.run_id)}" aria-current="${run.conversation_id === state.currentConversationId ? "true" : "false"}">
          <span class="history-goal">${escapeHtml(run.goal)}</span>
          <span class="history-status ${escapeHtml(run.status)}">${run.pinned ? "置顶 · " : ""}${run.archived ? "归档 · " : ""}${escapeHtml(runStatusLabel(run))} · ${run.appid} · ${run.turn_count}轮</span>
          <time class="history-time">${shortTime(run.updated_at_local || run.updated_at)}</time>
        </button><button type="button" class="conversation-more" data-conversation-id="${escapeHtml(run.conversation_id)}" aria-label="会话操作：${escapeHtml(run.goal)}" aria-haspopup="menu" aria-expanded="false">⋯</button>
      </div>`).join("")
    : `<p class="trace-empty">${query ? "没有匹配的调查。" : state.historyView === "archived" ? "没有已归档会话。" : "还没有 Agent 运行记录。"}</p>`;
  container.querySelectorAll("[data-run-id]").forEach((button) => {
    button.addEventListener("click", () => openSavedRun(button.dataset.runId));
  });
  container.querySelectorAll("[data-conversation-id]").forEach((button) => {
    button.addEventListener("click", () => openConversationMenu(button));
  });
}

function closeConversationMenu(returnFocus = false) {
  const opener = document.querySelector('.conversation-more[aria-expanded="true"]');
  $("#conversationMenu").hidden = true;
  if (opener) { opener.setAttribute("aria-expanded", "false"); if (returnFocus) opener.focus(); }
}

function openConversationMenu(button) {
  const wasOpen = button.getAttribute("aria-expanded") === "true";
  closeConversationMenu();
  if (wasOpen) return;
  const item = state.runs.find((run) => run.conversation_id === button.dataset.conversationId);
  if (!item) return;
  state.menuConversation = item;
  button.setAttribute("aria-expanded", "true");
  const menu = $("#conversationMenu");
  menu.innerHTML = `<button type="button" role="menuitem" data-action="rename">重命名</button><button type="button" role="menuitem" data-action="pin">${item.pinned ? "取消置顶" : "置顶"}</button><button type="button" role="menuitem" data-action="archive">${item.archived ? "恢复归档" : "归档"}</button><button type="button" role="menuitem" data-action="delete" class="danger">永久删除</button>`;
  menu.hidden = false;
  const rect = button.getBoundingClientRect();
  menu.style.left = `${Math.max(8,Math.min(rect.left,window.innerWidth-178))}px`;
  menu.style.top = `${Math.max(8,Math.min(rect.bottom+5,window.innerHeight-menu.offsetHeight-8))}px`;
  menu.querySelector("button").focus();
}

async function conversationAction(action) {
  const item = state.menuConversation;
  if (!item) return;
  closeConversationMenu();
  if (action === "rename" || action === "delete") {
    state.dialogTarget = {item,action};
    const removing = action === "delete";
    $("#conversationDialogTitle").textContent = removing ? "永久删除会话？" : "重命名会话";
    $("#conversationDialogMessage").textContent = removing ? "此会话的全部问答和专属任务记录将永久删除，无法恢复。共用的游戏数据和分析报告会保留。" : "设置一个便于查找的名称。";
    $("#conversationTitleLabel").hidden = removing;
    $("#conversationTitleInput").hidden = removing;
    $("#conversationTitleInput").required = !removing;
    $("#conversationTitleInput").value = item.goal.slice(0,100);
    $("#conversationDialogError").textContent = "";
    $("#confirmConversationEdit").textContent = removing ? "永久删除" : "保存";
    $("#confirmConversationEdit").classList.toggle("danger",removing);
    $("#conversationDialog").showModal();
    if (!removing) $("#conversationTitleInput").select();
    return;
  }
  try {
    const changes = action === "pin" ? {pinned: !item.pinned} : {archived: !item.archived};
    await api(`/api/conversations/${item.conversation_id}`, {method:"PATCH",body:JSON.stringify(changes)});
    if (action === "archive" && state.currentConversationId === item.conversation_id) newInvestigation();
    await loadRunHistory();
    showToast(action === "pin" ? (item.pinned ? "已取消置顶" : "已置顶") : (item.archived ? "已恢复归档会话" : "已归档"));
  } catch (error) { showToast(error.message); }
}

async function submitConversationEdit(event) {
  event.preventDefault();
  const target = state.dialogTarget;
  if (!target) return;
  const button = $("#confirmConversationEdit");
  button.disabled = true;
  $("#cancelConversationEdit").disabled = true;
  try {
    const id = target.item.conversation_id;
    if (target.action === "delete") {
      await api(`/api/conversations/${id}`,{method:"DELETE"});
      if (state.currentConversationId === id) newInvestigation();
      showToast("会话已永久删除，无法恢复");
    } else {
      const title = $("#conversationTitleInput").value.trim();
      if (!title) throw new Error("会话名称不能为空");
      await api(`/api/conversations/${id}`,{method:"PATCH",body:JSON.stringify({title})});
      if (state.currentConversationId === id) { state.currentTitle = title; $("#workspaceTitle").title = `会话：${title}`; }
      showToast("会话已重命名");
    }
    $("#conversationDialog").close();
    await loadRunHistory();
  } catch (error) { $("#conversationDialogError").textContent = error.message; }
  finally { button.disabled = false; $("#cancelConversationEdit").disabled = false; }
}

function openSavedRun(runId) {
  closeMobileSidebar();
  window.clearTimeout(state.pollTimer);
  state.selectionEpoch++;
  state.currentRunId = runId;
  state.currentConversationId = null;
  state.conversationBusy = true;
  state.runConfig = null;
  state.currentArchived = false;
  state.currentTitle = null;
  $("#submitRun").disabled = true;
  $("#conversationHistory").innerHTML = "";
  state.lastRunStatus = null;
  openDrawer();
  resetTaskView();
  $("#workspaceTitle").textContent = "载入调查…";
  $("#runQuestion").textContent = "正在读取调查目标…";
  $("#taskAppid").textContent = "";
  renderRunHistory();
  $("#runLogs").textContent = "正在读取持久化运行记录…";
  $("#finalAnswerPanel").hidden = true;
  $("#runMetrics").textContent = "正在读取模型调用数据…";
  pollRun(runId);
}

async function startAgent(event) {
  event.preventDefault();
  const appid = Number($("#appidInput").value);
  const goal = $("#goalInput").value.trim();
  if (!Number.isInteger(appid) || appid <= 0) return showToast("AppID 必须是正整数");
  if (goal.length < (state.currentConversationId ? 1 : 5)) return showToast(state.currentConversationId ? "请输入追问内容" : "请写下至少 5 个字的调查目标");
  if (state.conversationBusy) return showToast("本轮尚未结束，请先完成审批或停止任务");
  if (state.submitting) return;
  state.submitting = true;
  $("#submitRun").disabled = true;
  window.clearTimeout(state.pollTimer);
  const epoch = ++state.selectionEpoch;
  state.currentRunId = null;
  openDrawer();
  resetTaskView();
  $("#runDrawer").classList.add("running");
  $("#runStatus").textContent = "提交中";
  $("#workspaceTitle").textContent = `游戏 ${appid} · 版本调查`;
  $("#runQuestion").textContent = goal;
  $("#taskAppid").textContent = `APPID ${appid}`;
  $("#runLogs").textContent = "正在提交 Agent 任务…";
  $("#runPlan").innerHTML = "<li>等待规划器生成计划…</li>";
  $("#runEvents").innerHTML = '<p class="trace-empty">等待 Agent 选择第一个工具…</p>';
  $("#finalAnswerPanel").hidden = true;
  $("#approvalPanel").hidden = true;
  $("#runMetrics").textContent = "等待模型调用数据…";
  try {
    const run = await api("/api/runs", {
      method: "POST",
      // Web tasks always use automatic evidence search, including follow-ups
      // opened from historical off/on runs. CLI/API retain their debug modes.
      body: JSON.stringify({ ...state.runConfig, appid, goal, investigation_profile: "focused", rag_mode: "auto",
        window_days: state.runConfig?.window_days ?? 7, refresh_data: false, force_reclassify: false,
        conversation_id: state.currentConversationId }),
    });
    if (epoch !== state.selectionEpoch) { loadRunHistory().catch(() => {}); return; }
    state.currentRunId = run.run_id;
    completionSound.observe([{run_id: run.run_id, status: "queued"}]);
    state.currentConversationId = run.conversation_id;
    state.conversationBusy = true;
    rememberConversation(run.conversation_id);
    $("#goalInput").value = "";
    state.lastRunStatus = null;
    $("#cancelRun").hidden = false;
    loadRunHistory().catch(() => {});
    pollRun(run.run_id);
  } catch (error) {
    if (epoch === state.selectionEpoch) {
      $("#runStatus").textContent = "提交失败";
      $("#runDrawer").classList.remove("running");
      $("#runLogs").textContent = error.message;
      showToast(error.message);
    }
  } finally {
    state.submitting = false;
    $("#submitRun").disabled = state.conversationBusy;
  }
}

async function pollRun(runId) {
  const epoch = state.selectionEpoch;
  window.clearTimeout(state.pollTimer);
  try {
    const run = await api(`/api/runs/${runId}`);
    completionSound.observe([run]);
    if (state.currentRunId !== runId || epoch !== state.selectionEpoch) return;
    const statusChanged = run.status !== state.lastRunStatus;
    state.currentConversationId = run.conversation_id;
    state.runConfig = {...run.request};
    rememberConversation(run.conversation_id);
    $("#appidInput").value = run.request.appid;
    $("#appidInput").readOnly = true;
    $("#goalInput").minLength = 1;
    $("#goalInput").placeholder = "继续追问这次调查，例如：刚才的崩溃问题有哪些证据？";
    $("#runStatus").textContent = runStatusLabel(run);
    $("#workspaceTitle").textContent = run.request.goal || state.currentTitle || `游戏 ${run.request.appid} · 版本调查`;
    $("#workspaceTitle").title = state.currentTitle ? `会话：${state.currentTitle}\n本轮：${run.request.goal}` : run.request.goal;
    $("#runQuestion").textContent = run.request.goal;
    $("#taskAppid").textContent = `APPID ${run.request.appid}`;
    const metrics = run.telemetry;
    renderTurnTiming(run);
    const model = run.model_status;
    const currentError = model?.last_error;
    const active = ["queued","running","cancelling","awaiting_approval"].includes(run.status);
    const modelNode = $("#llmStatus");
    modelNode.hidden = !(active && (model?.pending || currentError));
    if (!modelNode.hidden) {
      const last = currentError ? `${currentError.error_code || currentError.error_type || `HTTP${currentError.http_status}`}：${currentError.error_detail || "接口请求失败"}` : "";
      const message = model.pending ? `模型${model.retrying ? "重试" : "请求"}中，等待返回可校验的结果。${model.retrying && last ? `上次 ${last}` : ""}` : last;
      if (modelNode.textContent !== message) modelNode.textContent = message;
    }
    if (metrics) {
      const tokens = metrics.prompt_tokens === null ? "未知" : `${metrics.prompt_tokens} 输入 / ${metrics.completion_tokens} 输出${metrics.usage_complete ? "" : "（部分用量）"}`;
      const cost = metrics.estimated_cost === null ? "未知或未配置单价" : `${metrics.currency} ${Number(metrics.estimated_cost).toFixed(6)}`;
      $("#runMetrics").textContent = `${metrics.model_calls} 次模型调用 · ${metrics.http_requests} 次请求 · ${metrics.retries} 次重试 · ${metrics.fallbacks} 次兼容降级\nToken：${tokens}\n费用估算：${cost}\n请求耗时合计：${(metrics.latency_ms / 1000).toFixed(2)}s · ${metrics.failed_requests} 次请求失败 · ${metrics.failed_calls} 次模型调用失败 · ${metrics.pending_requests} 次未完成请求\n模型：${metrics.models.join(" / ") || "—"}`;
    }
    const cancellable = ["queued", "running", "cancelling", "awaiting_approval"].includes(run.status);
    $("#recoverRun").hidden=!(run.status==="interrupted" || (run.status==="failed" && run.return_code!==0));
    $("#recoverRun").disabled=false;
    state.conversationBusy = cancellable || state.currentArchived;
    $("#submitRun").disabled = state.conversationBusy || state.submitting;
    $("#composerNote").textContent = cancellable ? "本轮正在进行 · 完成或停止后可继续追问" : "在当前会话继续追问 · 更换游戏请新建调查";
    $("#cancelRun").hidden = !cancellable;
    $("#cancelRun").disabled = run.status === "cancelling";
    if (statusChanged) {
      state.lastRunStatus = run.status;
      loadRunHistory().catch(() => {});
      const conversation = await api(`/api/conversations/${run.conversation_id}`);
      if (state.currentRunId !== runId || epoch !== state.selectionEpoch) return;
      renderConversationHistory(conversation, runId);
      state.currentTitle = conversation.title;
      state.currentArchived = Boolean(conversation.archived);
      $("#workspaceTitle").textContent = run.request.goal || conversation.title;
      $("#workspaceTitle").title = `会话：${conversation.title}\n本轮：${run.request.goal}`;
      if (state.currentArchived) {
        state.conversationBusy = true;
        $("#submitRun").disabled = true;
        $("#composerNote").textContent = "会话已归档 · 恢复归档后可继续追问";
      }
    }
    $("#runPlan").innerHTML = run.plan.length ? run.plan
        .map((item) => `<li>${escapeHtml(item)}</li>`)
        .join("") : "<li>暂无调查计划。</li>";
    $("#runEvents").innerHTML = run.events.length
      ? run.events.map((item) => `
          <article class="tool-event ${item.ok ? "ok" : "failed"}">
            <span class="tool-event-index">${String(item.step).padStart(2, "0")}</span>
            <div>
              <strong>${escapeHtml(item.tool)}</strong>
              ${item.data?.execution ? `<span class="tool-duration">${(item.data.execution.elapsed_ms / 1000).toFixed(2)}s</span>` : ""}
              <p>${escapeHtml(item.summary)}</p>
            </div>
          </article>`).join("")
      : '<p class="trace-empty">等待 Agent 选择第一个工具…</p>';
    const logs = run.logs.length ? run.logs.join("\n") : "等待输出…";
    const logNode = $("#runLogs");
    logNode.textContent = logs;
    logNode.scrollTop = logNode.scrollHeight;
    if (run.status === "awaiting_approval" && run.pending_approval) {
      $("#runDrawer").classList.remove("running");
      $("#approvalPanel").hidden = false;
      $("#approvalTool").textContent = run.pending_approval.tool;
      $("#approvalMessage").textContent = run.pending_approval.message;
      if (statusChanged) showToast("Agent 等待你的审批");
      return;
    }
    $("#approvalPanel").hidden = true;
    if (["queued", "running", "cancelling"].includes(run.status)) {
      $("#runDrawer").classList.toggle("running", run.status !== "cancelling");
      state.pollTimer = window.setTimeout(() => pollRun(runId), 900);
      return;
    }
    $("#runDrawer").classList.remove("running");
    if (run.final_answer) {
      $("#finalAnswerPanel").hidden = false;
      window.GameLensMarkdown.render($("#finalAnswer"), run.final_answer);
    }
    $("#viewFullReport").hidden = run.status!=="completed" || !run.artifacts?.final_report;
    if (statusChanged) {
      $("#planDetails").open = run.status!=="completed";
      $("#toolDetails").open = run.status!=="completed";
    }
    if (run.status === "completed") {
      if (statusChanged) showToast(runStatusLabel(run) === "已完成" ? "Agent 工作流完成" : `本轮已结束：${runStatusLabel(run)}`);
      await loadAnalyses();
      if (state.currentRunId !== runId || epoch !== state.selectionEpoch) return;
      const reportPath = run.artifacts?.final_report?.replaceAll("\\", "/");
      if (reportPath?.endsWith("_investigation_report.md") || reportPath?.endsWith("_complete_report.md")) { await loadTaskInvestigation(run); return; }
      const match = reportPath ? state.analyses.find((item) => reportPath.endsWith(`/${item.appid}_${item.news_id}_impact_report.md`)) : state.analyses.find((item) => item.appid === run.request.appid);
      if (match) { $("#analysisSelect").value = `${match.appid}:${match.news_id}`; await loadDashboard(); }
    } else if (run.status === "cancelled") {
      showToast("Agent 任务已取消");
    } else if (run.status === "interrupted") {
      showToast("任务因服务重启而中断，可查看已保存轨迹");
    } else {
      showToast("Agent 运行失败，请检查日志");
    }
  } catch (error) {
    if (state.currentRunId !== runId || epoch !== state.selectionEpoch) return;
    $("#runStatus").textContent = "读取失败";
    $("#runLogs").textContent += `\n${error.message}`;
    state.conversationBusy = false;
    $("#submitRun").disabled = false;
  }
}

async function loadTaskInvestigation(run) {
  const epoch=++state.reportEpoch;
  const response=await fetch(`/api/runs/${run.run_id}/report`);
  if (!response.ok) throw new Error("调查报告读取失败");
  const report=await response.text();
  if (epoch!==state.reportEpoch) return;
  const key=`task:${run.run_id}`;
  if (!Array.from($("#analysisSelect").options).some(option=>option.value===key)) {
    const option=document.createElement("option"); option.value=key; option.textContent=`本轮调查 · ${run.request.goal}`; $("#analysisSelect").prepend(option);
  }
  $("#analysisSelect").value=key;
  $("#patchTitle").textContent="本轮可追溯证据调查";
  $("#patchDate").textContent=shortTime(run.started_at_local || run.started_at);
  $("#windowLabel").textContent="仅代表本地已获取评论快照";
  $("#patchLink").hidden=true;
  $("#reportLink").href=`/api/runs/${run.run_id}/report`;
  $("#reportLink").hidden=false;
  $("#downloadReport").href=`/api/runs/${run.run_id}/report?download=true`;
  $("#downloadReport").hidden=false;
  $("#reportContent").textContent=report;
  const data=[...run.events].reverse().find(event=>event.ok && event.tool==="query_reviews")?.data;
  const discussion=[...run.events].reverse().find(event=>event.ok && event.data?.binding_verified && event.data?.coverage?.scope === "announcement_discussion_only")?.data;
  $("#dashboard .metric-card:nth-child(1) .metric-label").textContent="筛选评论好评率";
  $("#dashboard .metric-card:nth-child(2) .metric-label").textContent="匹配评论数量";
  $("#positiveDelta").textContent=data ? percent(data.stats.positive_rate) : "—";
  $("#positiveRates").textContent="分母是全部匹配的本地评论";
  $("#volumeDelta").textContent=data ? String(data.stats.count) : "—";
  $("#volumeCounts").textContent="不是Steam总体数量";
  if (discussion) {
    $("#windowLabel").textContent="仅代表该公告的公开讨论快照";
    $("#dashboard .metric-card:nth-child(1) .metric-label").textContent="已采集公告留言";
    $("#dashboard .metric-card:nth-child(2) .metric-label").textContent="讨论页报告回复数";
    $("#positiveDelta").textContent=String(discussion.coverage.collected_count ?? "—");
    $("#positiveRates").textContent=discussion.coverage.complete ? "已覆盖当时公开主题" : "部分采集，不代表全部留言";
    $("#volumeDelta").textContent=String(discussion.coverage.reported_total ?? "—");
    $("#volumeCounts").textContent="采集时快照；不是独立玩家数";
  }
  for (const id of ["severityValue","severityDelta","relatedRate","actionableCount"]) $(`#${id}`).textContent="—";
  $("#topicChart").innerHTML='<p class="trace-empty">群体/版本统计详情见报告；本轮未额外进行LLM主题分类。</p>';
  if (discussion) $("#topicChart").textContent="留言已关联到具体公告；观点归纳和限制见调查结论与完整报告。";
  $("#dashboard .sentiment-panel").hidden=true;
  const originals=new Map();
  for (const event of run.events) {
    if (!event.ok) continue;
    const data=event.data||{};
    for (const review of [...(data.reviews||[]),...(data.review?[data.review]:[])]) originals.set(review.review_id,review);
  }
  const announcements=new Map();
  for(const event of run.events) if(event.ok) for(const news of event.data?.announcements||[]) announcements.set(news.news_id,news);
  const comments=new Map();
  for (const event of run.events) if (event.ok) for (const comment of event.data?.comments||[]) comments.set(`${comment.news_id}:${comment.comment_id}`,comment);
  $("#evidenceGrid").innerHTML=[...originals.values()].map(review=>`<article class="evidence-card"><h4>商店评价 ${escapeHtml(review.review_id)}</h4><blockquote>${escapeHtml(review.text)}</blockquote><footer class="evidence-footer"><span>${escapeHtml(review.language)} · 评论时游玩 ${(review.playtime_at_review_minutes/60).toFixed(1)}h</span><span>${review.is_positive?"Steam好评":"Steam差评"}</span></footer></article>`).join("") + [...comments.values()].map(comment=>`<article class="evidence-card"><h4>公告 ${escapeHtml(comment.news_id)} · 留言 ${escapeHtml(comment.comment_id)}</h4><blockquote>${escapeHtml(comment.text)}</blockquote><footer class="evidence-footer"><span>公告讨论区 · ${escapeHtml(shortTime(comment.created_at))}${comment.contains_quoted_text ? " · 含引用文字" : ""}</span><span>非商店评价</span></footer></article>`).join("") + [...announcements.values()].map(news=>`<article class="evidence-card"><h4>公告 ${escapeHtml(news.news_id)} · ${escapeHtml(news.title)}</h4><blockquote>${escapeHtml(news.contents||news.title)}</blockquote></article>`).join("") || '<p class="trace-empty">本轮没有返回可展示的原文，请查看统计报告。</p>';
  $("#warningPanel").hidden=false;
  $("#warningList").innerHTML="<li>本地快照可能不完整，关键词命中不代表确认缺陷，统计差异不能证明因果关系。</li>";
  switchTab("report");
}

async function recoverCurrentRun() {
  const id=state.currentRunId;
  if (!id) return;
  $("#recoverRun").disabled=true;
  try {
    await api(`/api/runs/${id}/recover`,{method:"POST",body:"{}"});
    if (state.currentRunId===id) { state.lastRunStatus=null; pollRun(id); }
  } catch (error) { $("#recoverRun").disabled=false; showToast(error.message); }
}

function rememberConversation(id) {
  try { if (id) localStorage.setItem("patchpulse.conversation", id); else localStorage.removeItem("patchpulse.conversation"); } catch (_) {}
}

function renderConversationHistory(conversation, currentRunId) {
  const turns = conversation.turns.filter((turn) => turn.run_id !== currentRunId);
  $("#conversationHistory").innerHTML = turns.map((turn) => `
    <article class="question-message"><div><p class="message-caption">第 ${turn.turn_index} 轮 · ${escapeHtml(runStatusLabel(turn))}</p><p>${escapeHtml(turn.request.goal)}</p></div></article>
    <article class="previous-answer"><p class="turn-heading">游鉴 GameLens · 第 ${turn.turn_index} 轮</p><div class="answer-prose" data-turn-answer></div>${historicalTiming(turn)}</article>
  `).join("");
  $("#conversationHistory").querySelectorAll("[data-turn-answer]").forEach((node, index) => {
    const turn = turns[index];
    window.GameLensMarkdown.render(node, turn.final_answer || (turn.status === "completed" ? "本轮没有保存回答。" : "本轮未完成，未生成最终结论。"));
  });
}

function renderTurnTiming(run) {
  const timing = window.GameLensTiming.fallback(run);
  $("#turnTiming").hidden = false;
  $("#turnElapsed").textContent = `用时 ${window.GameLensTiming.duration(timing.elapsed_ms)}`;
  $("#turnTimingDetail").textContent = timingDetail(timing);
}

function timingDetail(timing) {
  return `总用时：${window.GameLensTiming.duration(timing.elapsed_ms)}\n模型请求累计：${window.GameLensTiming.duration(timing.model_request_ms)}\n工具累计：${window.GameLensTiming.duration(timing.tool_ms)}\n请求：${timing.http_requests ?? "未知"} 次 · 重试：${timing.retries ?? "未知"} 次\n${timing.note || "按已保存的起止时间计算；分项耗时可能重叠，不能相加。"}`;
}

function historicalTiming(turn) {
  const timing = window.GameLensTiming.fallback(turn);
  return `<details class="turn-timing"><summary><span>用时 ${escapeHtml(window.GameLensTiming.duration(timing.elapsed_ms))}</span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg></summary><p>${escapeHtml(timingDetail(timing))}</p></details>`;
}

async function cancelCurrentRun() {
  if (!state.currentRunId) return;
  const button = $("#cancelRun");
  const runId = state.currentRunId;
  button.disabled = true;
  try {
    await api(`/api/runs/${runId}/cancel`, {
      method: "POST",
      body: "{}",
    });
    if (state.currentRunId !== runId) return;
    $("#runStatus").textContent = "停止中";
    showToast("正在停止 Agent 进程");
    pollRun(state.currentRunId);
  } catch (error) {
    button.disabled = false;
    showToast(error.message);
  }
}

async function submitApproval(approved) {
  if (!state.currentRunId) return;
  $("#approvalPanel").hidden = true;
  $("#runDrawer").classList.add("running");
  const runId = state.currentRunId;
  $("#runStatus").textContent = "恢复中";
  try {
    await api(`/api/runs/${runId}/approval`, {
      method: "POST",
      body: JSON.stringify({ approved, comment: approved ? "用户在Web界面批准" : "用户在Web界面拒绝" }),
    });
    if (state.currentRunId === runId) pollRun(runId);
  } catch (error) {
    if (state.currentRunId !== runId) return;
    $("#approvalPanel").hidden = false;
    $("#runDrawer").classList.remove("running");
    $("#runStatus").textContent = "审批失败";
    $("#runLogs").textContent += `\n${error.message}`;
  }
}

function updateClock() {
  $("#clock").textContent = new Date().toLocaleTimeString("en-GB", { hour12: false, timeZone: "Asia/Shanghai" });
}

async function boot() {
  $("#viewFullReport").addEventListener("click", async () => {
    const id=state.currentRunId;
    if (!id) return;
    const button=$("#viewFullReport");button.disabled=true;button.textContent="正在打开报告…";
    try {
      const run=await api(`/api/runs/${id}`);
      if (state.currentRunId!==id) return;
      await loadTaskInvestigation(run);
      if (state.currentRunId!==id) return;
      setInspector(true);switchTab("report");$("#tabReport").focus();
    } catch(error) { showToast(error.message); }
    finally {button.disabled=false;button.textContent="查看完整报告";}
  });
  updateSoundButton();
  document.addEventListener("pointerdown", () => completionSound.unlock(), {passive: true});
  document.addEventListener("keydown", () => completionSound.unlock(), {passive: true});
  $("#toggleSound").addEventListener("click", () => {
    completionSound.setEnabled(!completionSound.enabled);
    updateSoundButton();
    showToast(completionSound.enabled ? "已开启任务完成提示音" : "已关闭任务完成提示音");
  });
  monitorCompletions();
  updateClock();
  window.setInterval(updateClock, 1000);
  $("#controlForm").addEventListener("submit", startAgent);
  $("#loadButton").addEventListener("click", () => loadDashboard().catch((error) => showToast(error.message)));
  $("#analysisSelect").addEventListener("change", () => loadDashboard().catch((error) => showToast(error.message)));
  $("#newInvestigation").addEventListener("click", newInvestigation);
  $("#toggleInspector").addEventListener("click", () => setInspector($("#dashboard").hidden));
  $("#closeInspector").addEventListener("click", () => { setInspector(false); $("#toggleInspector").focus(); });
  $("#toggleSidebar").addEventListener("click", toggleSidebar);
  $("#sidebarBackdrop").addEventListener("click", () => { closeMobileSidebar(); $("#toggleSidebar").focus(); });
  window.addEventListener("resize", closeMobileSidebar);
  updateSidebarControl();
  let searchTimer;
  $("#historySearch").addEventListener("input", () => { renderRunHistory(); window.clearTimeout(searchTimer); searchTimer = window.setTimeout(() => loadRunHistory().catch(error=>showToast(error.message)),200); });
  $("#historyView").addEventListener("change", () => { state.historyView = $("#historyView").value; closeConversationMenu(); loadRunHistory().catch(error=>showToast(error.message)); });
  $("#conversationMenu").addEventListener("click", event => { const item = event.target.closest("[data-action]"); if (item) conversationAction(item.dataset.action); });
  $("#conversationMenu").addEventListener("keydown", event => {
    const buttons = Array.from($("#conversationMenu").querySelectorAll("button"));
    const index = buttons.indexOf(document.activeElement);
    if (["ArrowDown","ArrowUp"].includes(event.key)) { event.preventDefault(); buttons[(index+(event.key==="ArrowDown"?1:-1)+buttons.length)%buttons.length].focus(); }
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); closeConversationMenu(true); }
  });
  document.addEventListener("click",event=>{if (!event.target.closest("#conversationMenu,.conversation-more")) closeConversationMenu();});
  $("#runHistory").addEventListener("scroll",()=>closeConversationMenu());
  window.addEventListener("resize",()=>closeConversationMenu());
  $("#conversationEditForm").addEventListener("submit",submitConversationEdit);
  $("#cancelConversationEdit").addEventListener("click",()=>$("#conversationDialog").close());
  document.querySelectorAll("[data-goal]").forEach((button) => button.addEventListener("click", () => {
    const input = $("#goalInput");
    input.value = button.dataset.goal; input.focus();
    const date = input.value.indexOf("YYYY-MM-DD");
    if (date >= 0) input.setSelectionRange(date, date + 10);
  }));
  $("#goalInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("#controlForm").requestSubmit(); }
  });
  document.addEventListener("keydown", (event) => {
    if ($("#conversationDialog").open) return;
    if (event.key === "Escape") {
      if (!$("#dashboard").hidden) { setInspector(false); $("#toggleInspector").focus(); }
      else if ($(".workbench").classList.contains("mobile-sidebar-open")) { closeMobileSidebar(); $("#toggleSidebar").focus(); }
    }
    if (event.key.toLowerCase() === "n" && !event.ctrlKey && !event.altKey && !event.metaKey && !event.isComposing && !event.target.closest("input,textarea,select,[contenteditable]")) { event.preventDefault(); newInvestigation(); }
  });
  const tabs = Array.from(document.querySelectorAll("[data-tab]"));
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      const direction = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (direction) { event.preventDefault(); const next = tabs[(index + direction + tabs.length) % tabs.length]; switchTab(next.dataset.tab); next.focus(); }
    });
  });
  $("#approveApproval").addEventListener("click", () => submitApproval(true));
  $("#denyApproval").addEventListener("click", () => submitApproval(false));
  $("#cancelRun").addEventListener("click", cancelCurrentRun);
  $("#recoverRun").addEventListener("click",recoverCurrentRun);
  $("#refreshRuns").addEventListener("click", () => loadRunHistory().catch((error) => showToast(error.message)));
  try {
    await Promise.all([loadAnalyses(), loadRunHistory()]);
    if (state.analyses.length) await loadDashboard();
    let saved = null;
    try { saved = localStorage.getItem("patchpulse.conversation"); } catch (_) {}
    if (saved) {
      const existing = state.runs.find((run) => run.conversation_id === saved);
      if (existing) openSavedRun(existing.run_id);
      else rememberConversation(null);
    }
  } catch (error) {
    showToast(error.message);
  }
}

document.addEventListener("DOMContentLoaded", boot);
