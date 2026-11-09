// Local browser regression. Mock run APIs so no real Agent/model is started.
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');

async function main() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--mute-audio'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const base = process.env.GAMELENS_URL || process.env.PATCH_PULSE_URL || 'http://127.0.0.1:8765';
    // Serve production assets through local interception. No server startup,
    // database lifespan mutations or real Agent/model requests are required.
    if (process.env.GAMELENS_STATIC_ONLY === '1') {
      const root = path.resolve(__dirname, '..', 'web');
      await page.route('**/*', async route => {
        const url = new URL(route.request().url());
        if (url.pathname.startsWith('/api/')) return route.fallback();
        const relative = url.pathname === '/' ? 'index.html' : url.pathname.replace(/^\/static\//, '');
        const file = path.resolve(root, relative);
        if (!file.startsWith(root + path.sep) || !fs.existsSync(file)) return route.abort();
        const type = {'.html':'text/html','.js':'application/javascript','.css':'text/css','.svg':'image/svg+xml'}[path.extname(file)];
        await route.fulfill({path:file,contentType:type});
      });
      const analyses = ['fixture-a','fixture-b'].map(news_id=>({appid:730,news_id,title:'离线演示更新',published_at:'2026-10-01T00:00:00Z'}));
      await page.route('**/api/analyses', route => route.fulfill({contentType:'application/json',body:JSON.stringify(analyses)}));
      await page.route('**/api/dashboard/**', route => route.fulfill({contentType:'application/json',body:JSON.stringify({
        game:{appid:730,patch_title:'离线演示更新',published_at:'2026-10-01T00:00:00Z',patch_url:'#',window_days:7},
        full_window:{before_total:10,after_total:10,before_positive_rate:0.5,after_positive_rate:0.5,positive_rate_change_pp:0,review_count_change:0},
        sample:{after_actionable_count:0,after_sentiments:{}},topic_changes:[],evidence:[],warnings:[],report_url:'/api/mock_report',
      })}));
      await page.route('**/api/mock_report', route=>route.fulfill({contentType:'text/plain',body:'# 离线演示报告\n\n'+ '这段文字只用于验证报告查看器，不是真实调查结果。'.repeat(8)}));
    }
    const request = { appid: 730, goal: '评测回放：检查游戏更新后的玩家反馈', classify_count: 10 };
    const emptyUsage = { model_calls: 0, http_requests: 0, retries: 0, fallbacks: 0, failed_requests: 0, failed_calls: 0, pending_requests: 0, prompt_tokens: null, completion_tokens: null, latency_ms: 0, estimated_cost: null, models: [] };
    const fixture = id => ({ run_id: id, conversation_id: ['submitted-demo','followup-demo'].includes(id) ? 'new-session' : `${id}-session`, turn_index: id === 'followup-demo' ? 2 : 1, request, status: id === 'approval-demo' ? 'awaiting_approval' : 'completed', plan: ['检查数据', '评估证据'], events: [{ step: 1, tool: 'inspect_data', ok: true, summary: '评测回放：已有数据可复用', data: { execution: { elapsed_ms: 120 } } }], logs: ['评测回放'], final_answer: id === 'approval-demo' ? '' : '评测回放：完整窗口与分类样本需分开解读。', pending_approval: id === 'approval-demo' ? { tool: 'classify_review_sample', message: '评测回放：本次操作需要确认' } : null, telemetry: emptyUsage });
    const runs = ['done-demo', 'approval-demo'].map(id => ({ run_id: id, appid: 730, goal: id === 'done-demo' ? '评测回放：已完成调查' : '评测回放：等待审批调查', status: fixture(id).status, updated_at: new Date().toISOString() }));
    let submissions = 0;
    let approvals = 0;
    let cancellations = 0;
    let postedRequest;
    const postedRequests = [];
    let submittedComplete = false;
    const conversationEdits = new Map();
    const removedConversations = new Set();
    let delayCompleted = false;
    let investigationDemo=false;
    let recoveryDemo=false;
    let recoverCalls=0;
    await page.route('**/api/runs**', async route => {
      const url = new URL(route.request().url());
      const method = route.request().method();
      let result;
      if (url.pathname.endsWith('/report')) {
        await route.fulfill({contentType:'text/plain',body:'# 评测回放调查报告\n\n本地快照包含1条匹配评论；原文：评测原文崩溃反馈。\n\n## 核验细节\n字段路径、来源哈希和工具数据只放完整报告。'});
        return;
      }
      if (method === 'POST') {
        if (url.pathname.endsWith('/recover')) { recoverCalls++; result={status:'queued'}; }
        else if (url.pathname.endsWith('/approval')) { approvals++; result = { status: 'queued' }; }
        else if (url.pathname.endsWith('/cancel')) { cancellations++; result = { status: 'cancelled' }; }
        else {
          submissions++; postedRequest = route.request().postDataJSON(); postedRequests.push(postedRequest);
          const id = submissions === 1 ? 'submitted-demo' : 'followup-demo';
          result = { run_id: id, status: 'queued', conversation_id: 'new-session', turn_index: submissions };
        }
      } else if (url.pathname === '/api/runs') result = runs;
      else {
        const id = url.pathname.split('/')[3];
        if (id === 'done-demo' && delayCompleted) await new Promise(resolve => setTimeout(resolve, 250));
        result = fixture(id);
        if (id==='done-demo' && investigationDemo) {
          result.artifacts={final_report:'D:/GameReview_Agent/reports/730_done-demo_complete_report.md'};
          result.events=[{step:1,tool:'query_reviews',ok:true,summary:'评测查询',data:{stats:{count:1,positive_rate:0},reviews:[{review_id:'fixture-review',text:'评测原文崩溃反馈。',language:'schinese',playtime_at_review_minutes:6000,is_positive:false}]}}];
        }
        if (id==='done-demo' && recoveryDemo) { result.status=recoverCalls?'running':'interrupted'; result.final_answer=''; result.return_code=-1; }
        if (id === 'submitted-demo') { result.request = postedRequests[0]; result.status = submittedComplete ? 'completed' : 'running'; result.final_answer = submittedComplete ? '评测回放：第一轮调查完成。' : ''; }
        if (id === 'followup-demo') { result.request = postedRequests[1]; result.status = 'running'; result.final_answer = ''; }
        if (id === 'approval-demo' && approvals) { result.status = 'running'; result.pending_approval = null; }
        if (id === 'approval-demo' && cancellations) { result.status = 'cancelled'; result.pending_approval = null; }
      }
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify(result) });
    });
    await page.route('**/api/conversations**', async route => {
      const url = new URL(route.request().url());
      let result;
      let catalog = runs.map(run => ({...run, conversation_id: `${run.run_id}-session`, last_run_id: run.run_id, title: run.goal, turn_count: 1}));
      if (submissions) catalog.unshift({conversation_id:'new-session', last_run_id: submissions === 1 ? 'submitted-demo' : 'followup-demo', title: postedRequests[0].goal, appid:730, turn_count:submissions, status: submittedComplete && submissions === 1 ? 'completed' : 'running', updated_at:new Date().toISOString()});
      catalog = catalog.map(item=>({...item,...conversationEdits.get(item.conversation_id)})).filter(item=>!removedConversations.has(item.conversation_id));
      catalog.sort((a,b)=>Number(Boolean(b.pinned))-Number(Boolean(a.pinned)));
      if (route.request().method() === 'PATCH') {
        const cid = url.pathname.split('/')[3];
        const changes = route.request().postDataJSON();
        conversationEdits.set(cid,{...conversationEdits.get(cid),...changes});
        result = {...catalog.find(item=>item.conversation_id===cid),...changes};
      } else if (route.request().method() === 'DELETE') {
        const cid = url.pathname.split('/')[3];
        removedConversations.add(cid);
        result = {deleted:true,conversation_id:cid};
      } else if (url.pathname === '/api/conversations') {
        const view = url.searchParams.get('view') || 'active';
        const query = (url.searchParams.get('q') || '').toLowerCase();
        result = catalog.filter(item=>(view==='all' || Boolean(item.archived)===(view==='archived')) && `${item.title} ${item.appid}`.toLowerCase().includes(query));
      }
      else {
        const cid = url.pathname.split('/')[3];
        const ids = cid === 'new-session' ? (submissions === 1 ? ['submitted-demo'] : ['submitted-demo','followup-demo']) : [cid.replace(/-session$/, '')];
        const turns = ids.map(id => {
          const turn = fixture(id);
          if (id === 'submitted-demo') { turn.request = postedRequests[0]; turn.final_answer = '评测回放：第一轮调查完成。'; }
          if (id === 'followup-demo') { turn.request = postedRequests[1]; turn.status = 'running'; }
          return turn;
        });
        result = { conversation_id:cid, appid:730, title:'评测回放会话', turns, telemetry:emptyUsage };
      }
      await route.fulfill({contentType:'application/json', body:JSON.stringify(result)});
    });
    await page.goto(base);
    await page.locator('.history-run').first().waitFor();
    if (process.env.GAMELENS_STATIC_ONLY !== '1') await page.waitForFunction(() => document.querySelector('#patchTitle').textContent !== '尚未载入分析');
    await page.waitForFunction(() => !document.querySelector('#dashboard').classList.contains('loading'));
    const screenshots = path.join(__dirname, '..', '.ui-preview');
    fs.mkdirSync(screenshots, { recursive: true });
    await page.screenshot({ path: path.join(screenshots, 'workbench-desktop.png') });
    assert.equal(await page.locator('#ragModeInput').count(), 0);
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({width,height:960});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
      const fits = await page.locator('#searchHelp').evaluate(el => {
        const bounds=el.getBoundingClientRect();
        return bounds.left >= 0 && bounds.right <= window.innerWidth && el.scrollWidth <= el.clientWidth;
      });
      assert.ok(fits, `Search helper must fit at ${width}px`);
      if (width===320) await page.screenshot({path:path.join(screenshots,'workbench-compact.png')});
    }
    assert.equal(await page.locator('#toggleSound').getAttribute('aria-pressed'),'true');
    await page.locator('#toggleSound').click();
    assert.equal(await page.locator('#soundState').innerText(),'已关闭');
    await page.locator('#toggleSound').click();
    assert.equal(await page.locator('#soundState').innerText(),'已开启');
    assert.equal(await page.locator('#dashboard').isVisible(), false);
    await page.locator('#toggleInspector').click();
    assert.equal(await page.locator('#dashboard').isVisible(), true);
    await page.screenshot({ path: path.join(screenshots, 'workbench-evidence.png') });
    assert.equal(await page.locator('#welcomePanel').isVisible(), true);
    await page.locator('[data-goal]').first().click();
    assert.ok((await page.locator('#goalInput').inputValue()).includes('缓存评论'));
    await page.locator('#historySearch').fill('已完成');
    assert.equal(await page.locator('.history-run').count(), 1);
    await page.locator('#historySearch').fill('');
    await page.locator('[data-run-id="done-demo"]').click();
    await page.locator('#finalAnswerPanel').waitFor({ state: 'visible' });
    assert.ok((await page.locator('#runQuestion').innerText()).includes('评测回放'));
    assert.equal(await page.locator('#welcomePanel').isVisible(), false);
    await page.locator('#tabEvidence').click();
    assert.equal(await page.locator('#panelEvidence').isVisible(), true);
    await page.locator('#tabEvidence').press('ArrowRight');
    assert.equal(await page.locator('#panelReport').isVisible(), true);
    await page.waitForFunction(() => document.querySelector('#reportContent').textContent.length > 100);
    const previousAppid = await page.locator('#appidInput').inputValue();
    await page.locator('#analysisSelect').selectOption({ index: 1 });
    assert.equal(await page.locator('#appidInput').inputValue(), previousAppid);
    delayCompleted = true;
    await page.locator('[data-run-id="done-demo"]').click();
    await page.locator('[data-run-id="approval-demo"]').click();
    await page.locator('#approvalPanel').waitFor({ state: 'visible' });
    await page.waitForTimeout(350);
    assert.equal(await page.locator('#runStatus').innerText(), '等待审批');
    assert.equal(await page.locator('#finalAnswerPanel').isVisible(), false);
    await page.locator('#approveApproval').click();
    await page.waitForFunction(() => document.querySelector('#runStatus').textContent === '调查中');
    assert.equal(approvals, 1);
    await page.locator('#cancelRun').click();
    await page.waitForFunction(() => document.querySelector('#runStatus').textContent === '已取消');
    assert.equal(cancellations, 1);
    await page.locator('#newInvestigation').click();
    assert.equal(await page.locator('#welcomePanel').isVisible(), true);
    await page.locator('#goalInput').fill('评测回放：提交新的独立调查');
    assert.equal(await page.locator('#ragModeInput').count(), 0);
    assert.equal(await page.locator('#goalInput').getAttribute('aria-describedby'), 'searchHelp');
    await page.locator('#goalInput').press('Enter');
    await page.waitForFunction(() => document.querySelector('#runStatus').textContent === '调查中');
    assert.equal(submissions, 1);
    assert.equal(postedRequest.appid, 730);
    assert.equal(postedRequest.conversation_id, null);
    assert.equal(postedRequest.rag_mode, 'auto');
    assert.equal(postedRequest.investigation_profile, 'focused');
    assert.equal(await page.locator('#sampleInput').count(), 0);
    submittedComplete = true;
    await page.waitForFunction(() => document.querySelector('#runStatus').textContent === '已完成');
    await page.locator('#goalInput').fill('展开');
    // Simulate a historical run configuration. Hidden options must not leak
    // into new Web follow-ups via the runConfig object spread.
    await page.evaluate(() => { state.runConfig.rag_mode = 'on'; });
    await page.locator('#goalInput').press('Enter');
    await page.waitForFunction(() => document.querySelector('#runStatus').textContent === '调查中');
    assert.equal(submissions, 2);
    assert.equal(postedRequests[1].conversation_id, 'new-session');
    assert.equal(postedRequests[1].rag_mode, 'auto');
    await page.waitForFunction(() => document.querySelector('#workspaceTitle').textContent === '展开');
    assert.equal(await page.locator('#workspaceTitle').innerText(), postedRequests[1].goal);
    await page.waitForFunction(() => document.querySelector('#conversationHistory').textContent.includes('第一轮调查完成'));
    assert.ok((await page.locator('#conversationHistory').innerText()).includes('第一轮调查完成'));
    assert.equal(await page.locator('#appidInput').getAttribute('readonly'), '');
    await page.reload();
    await page.locator('#toggleSound').waitFor();
    assert.equal(await page.locator('#toggleSound').getAttribute('aria-pressed'),'true');
    await page.waitForFunction(() => document.querySelector('#runStatus').textContent === '调查中');
    await page.waitForFunction(() => document.querySelector('#conversationHistory').textContent.includes('第一轮调查完成'));
    assert.ok((await page.locator('#conversationHistory').innerText()).includes('第一轮调查完成'));
    await page.screenshot({path:path.join(screenshots,'conversation-memory.png')});
    await page.locator('#newInvestigation').click();
    if (!await page.locator('#dashboard').isVisible()) await page.locator('#toggleInspector').click();
    await page.locator('#tabOverview').click();
    for (const width of [1024, 768, 390]) {
      await page.setViewportSize({ width, height: 900 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `horizontal overflow at ${width}`);
      const bounds = await page.locator('#submitRun').boundingBox();
      assert.ok(bounds.width > 0);
    }
    await page.locator('#closeInspector').click();
    await page.locator('#toggleSidebar').click();
    assert.equal(await page.locator('#sidebar').isVisible(), true);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#sidebar').isVisible(), false);
    await page.screenshot({ path: path.join(screenshots, 'workbench-mobile.png'), fullPage: true });
    await page.setViewportSize({width:1440,height:960});
    investigationDemo=true;
    delayCompleted=false;
    await page.locator('[data-run-id="done-demo"]').click();
    await page.waitForFunction(()=>document.querySelector('#patchTitle').textContent==='本轮可追溯证据调查');
    await page.locator('#toggleInspector').click();
    assert.ok((await page.locator('#reportContent').innerText()).includes('评测回放调查报告'));
    await page.locator('#tabEvidence').click();
    assert.ok((await page.locator('#evidenceGrid').innerText()).includes('评测原文崩溃反馈'));
    await page.locator('#closeInspector').click();
    assert.equal(await page.locator('#planDetails').getAttribute('open'),null);
    assert.equal(await page.locator('#toolDetails').getAttribute('open'),null);
    assert.ok(!(await page.locator('#finalAnswer').innerText()).includes('核验细节'));
    await page.locator('#viewFullReport').click();
    await page.locator('#dashboard').waitFor({state:'visible'});
    assert.equal(await page.locator('#dashboard').isVisible(),true);
    assert.equal(await page.locator('#panelReport').isVisible(),true);
    assert.ok((await page.locator('#reportContent').innerText()).includes('核验细节'));
    assert.ok((await page.locator('#downloadReport').getAttribute('href')).endsWith('?download=true'));
    await page.screenshot({path:path.join(screenshots,'summary-full-report.png')});
    await page.setViewportSize({width:390,height:900});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'report overflow on mobile');
    await page.setViewportSize({width:1440,height:960});
    await page.locator('#closeInspector').click();
    recoveryDemo=true;
    await page.locator('[data-run-id="done-demo"]').click();
    await page.locator('#recoverRun').waitFor({state:'visible'});
    await page.locator('#recoverRun').click();
    await page.waitForFunction(()=>document.querySelector('#runStatus').textContent==='调查中');
    assert.equal(recoverCalls,1);
    recoveryDemo=false;
    investigationDemo=false;
    await page.locator('#newInvestigation').click();
    const more = page.locator('[data-conversation-id="done-demo-session"]');
    await more.click();
    await page.locator('#conversationMenu [data-action="rename"]').click();
    await page.locator('#conversationTitleInput').fill('评测：CS2 崩溃问题');
    await page.locator('#confirmConversationEdit').click();
    await page.waitForFunction(()=>document.querySelector('#runHistory').textContent.includes('评测：CS2 崩溃问题'));
    await more.click();
    await page.locator('#conversationMenu [data-action="pin"]').click();
    await page.waitForFunction(()=>document.querySelector('.history-run').dataset.runId==='done-demo');
    await more.click();
    await page.screenshot({path:path.join(screenshots,'conversation-menu.png')});
    await page.locator('#conversationMenu [data-action="archive"]').click();
    await more.waitFor({state:'hidden'});
    await page.locator('#historyView').selectOption('archived');
    await more.waitFor({state:'visible'});
    await more.click();
    assert.equal(await page.locator('#conversationMenu [data-action="archive"]').innerText(),'恢复归档');
    await page.locator('#conversationMenu [data-action="archive"]').click();
    await more.waitFor({state:'hidden'});
    await page.locator('#historyView').selectOption('active');
    await more.waitFor({state:'visible'});
    await more.click();
    await page.locator('#conversationMenu [data-action="pin"]').click();
    await page.waitForFunction(()=>!document.querySelector('[data-run-id="done-demo"]').textContent.includes('置顶'));
    await more.click();
    await page.locator('#conversationMenu [data-action="delete"]').click();
    await page.locator('#cancelConversationEdit').click();
    assert.equal(removedConversations.has('done-demo-session'),false);
    await more.click();
    await page.locator('#conversationMenu [data-action="delete"]').click();
    await page.locator('#confirmConversationEdit').click();
    await more.waitFor({state:'hidden'});
    assert.equal(removedConversations.has('done-demo-session'),true);
    assert.deepEqual(errors, []);
    console.log('PASS: rename, pin/unpin, archive/restore, permanent delete/cancel; multi-turn, sound, reload recovery and responsive layout; no live model calls.');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
