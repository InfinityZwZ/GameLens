// Offline timer rendering; all browser requests are intercepted, no server/API.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function main() {
  const web = path.resolve(__dirname, '../web');
  const sandbox = {window:{}};
  vm.runInNewContext(fs.readFileSync(path.join(web, 'timing.js'), 'utf8'), sandbox);
  const timer = sandbox.window.GameLensTiming;
  assert.equal(timer.duration(234000), '3m 54s');
  assert.equal(timer.duration(3600000), '1h 0m 0s');
  assert.equal(timer.duration(null), '未知');
  assert.equal(timer.duration(Infinity), '未知');
  assert.equal(timer.duration(-1), '0s');
  const legacy = {started_at:'2026-10-11T00:00:00Z',finished_at:'2026-10-11T00:03:54Z'};
  assert.equal(timer.fallback(legacy).elapsed_ms, 234000);
  const {chromium} = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');
  const browser = await chromium.launch({channel:'chrome',headless:true,args:['--mute-audio']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:960}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname.startsWith('/api/')) {
        return route.fulfill({contentType:'application/json',body:'[]'});
      }
      const file = path.resolve(web, url.pathname === '/' ? 'index.html' : url.pathname.replace(/^\/static\//,''));
      if (!file.startsWith(web+path.sep) || !fs.existsSync(file)) return route.abort();
      const type = {'.html':'text/html','.js':'application/javascript','.css':'text/css','.svg':'image/svg+xml'}[path.extname(file)];
      await route.fulfill({path:file,contentType:type});
    });
    await page.goto('http://gamelens.test/');
    await page.waitForFunction(()=>typeof renderTurnTiming === 'function');
    const timing = {elapsed_ms:234000,model_request_ms:90000,tool_ms:60000,http_requests:2,retries:1,
      note:'离线组件演示：分项可能重叠，不能相加，不代表真实任务耗时。'};
    await page.evaluate(timing=>{
      openDrawer();
      document.querySelector('#workspaceTitle').textContent='每轮计时 · 离线界面验证';
      document.querySelector('#runQuestion').textContent='离线计时组件演示，不是实际游戏调查结果。';
      document.querySelector('#runPlan').innerHTML='<li>已完成组件验证。</li>';
      document.querySelector('#runEvents').textContent='仅使用固定展示数据，无真实模型请求。';
      document.querySelector('#finalAnswerPanel').hidden=false;
      document.querySelector('#finalAnswer').textContent='计时显示在每一轮回答下方；点击可查看分项。';
      renderTurnTiming({timing});
      renderConversationHistory({turns:[{run_id:'previous',turn_index:1,status:'completed',
        request:{goal:'上一轮离线演示'},final_answer:'上一轮的用时记录刷新后仍可查看。',timing:{...timing,elapsed_ms:120000}}]},'current');
    },timing);
    assert.equal(await page.locator('#turnElapsed').innerText(),'用时 3m 54s');
    assert.equal(await page.locator('#conversationHistory .turn-timing summary').innerText(),'用时 2m 0s');
    const summary=page.locator('#turnTiming summary');
    await summary.focus();await page.keyboard.press('Enter');
    assert.equal(await page.locator('#turnTiming').evaluate(node=>node.open),true);
    await page.evaluate(timing=>renderTurnTiming({timing:{...timing,elapsed_ms:270000}}),timing);
    assert.equal(await page.locator('#turnTiming').evaluate(node=>node.open),true);
    assert.equal(await page.locator('#turnElapsed').innerText(),'用时 4m 30s');
    await page.evaluate(timing=>renderTurnTiming({timing}),timing);
    if (process.env.GAMELENS_SCREENSHOTS) {
      fs.mkdirSync(process.env.GAMELENS_SCREENSHOTS,{recursive:true});
      await page.screenshot({path:path.join(process.env.GAMELENS_SCREENSHOTS,'turn-timing-desktop.png')});
      await page.locator('#turnTiming').screenshot({path:path.join(process.env.GAMELENS_SCREENSHOTS,'turn-timing-detail.png')});
    }
    await page.setViewportSize({width:390,height:844});
    assert(await summary.isVisible());
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.emulateMedia({reducedMotion:'reduce'});
    if (process.env.GAMELENS_SCREENSHOTS) await page.screenshot({path:path.join(process.env.GAMELENS_SCREENSHOTS,'turn-timing-mobile.png')});
    assert.deepEqual(errors,[]);
    console.log('PASS: per-turn elapsed time, old timestamps, keyboard disclosure, stable polling, mobile layout; zero live API calls.');
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
