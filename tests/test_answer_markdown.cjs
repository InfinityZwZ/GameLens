// Production UI and security regression; all requests intercepted, zero live API calls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');

async function main() {
  const web = path.resolve(__dirname, '../web');
  const answer = '当前资料无法比较更新前后的好评率，**不能判断这次更新是否改善了口碑**。\n\n' +
    '**样本统计**\n\n- 评论总数：**5,000 条**。\n- 好评：**4,166 条（83.32%）**；差评：834 条（16.68%）。\n\n' +
    '**主要反馈**\n\n- **正面**：玩法深度与社交体验。\n- **负面**：准星改动、外挂和匹配体验。\n\n' +
    '**范围与限制**\n\n这些数字只代表当前缓存的评论，不能推广为 Steam 总体，也不能证明更新导致了口碑变化。详细证据请查看完整报告。';
  const old = '当前缓存只覆盖部分评论，不能据此确认更新前后的变化。' +
    '现有正面反馈集中于玩法深度和社交体验，有玩家描述了与朋友共同游玩的经历。'.repeat(4) +
    '现有负面反馈提到了卡顿和匹配体验，但这仍只是玩家反馈，不能确认缺陷的实际发生频率。'.repeat(3) +
    '以上只代表已核对的本地资料，不能代表全部玩家。';
  const run = {run_id:'format-demo',conversation_id:'format-session',turn_index:2,
    request:{appid:730,goal:'离线排版演示：查看评论统计与主要反馈'},status:'completed',plan:[],events:[],logs:[],
    final_answer:answer,started_at:'2026-10-11T00:00:00Z',finished_at:'2026-10-11T00:03:54Z'};
  const previous = {...run,run_id:'previous',turn_index:1,final_answer:'**已有发现**\n\n1. 发现了卡顿反馈。\n2. 尚未确认原因。'};
  const browser = await chromium.launch({channel:'chrome',headless:true,args:['--mute-audio']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100}});
    const errors = [], requests = []; let writes = 0;
    page.on('pageerror', error=>errors.push(error.message));
    await page.route('**/*', async route=>{
      const request = route.request(), url = new URL(request.url()); requests.push(url.href);
      if (request.method() !== 'GET') writes++;
      if (url.pathname.startsWith('/api/')) {
        let body = [];
        if (url.pathname === '/api/runs/format-demo') body = run;
        if (url.pathname === '/api/conversations/format-session') body = {conversation_id:'format-session',appid:730,
          title:run.request.goal,turns:[previous,run]};
        return route.fulfill({contentType:'application/json',body:JSON.stringify(body)});
      }
      const file = path.resolve(web,url.pathname === '/' ? 'index.html' : url.pathname.replace(/^\/static\//,''));
      if (!file.startsWith(web+path.sep) || !fs.existsSync(file)) return route.abort();
      await route.fulfill({path:file,contentType:{'.html':'text/html','.js':'application/javascript','.css':'text/css','.svg':'image/svg+xml'}[path.extname(file)]});
    });
    await page.goto('http://gamelens.test/');
    await page.waitForFunction(()=>typeof openSavedRun === 'function' && Boolean(window.GameLensMarkdown));
    await page.evaluate(()=>openSavedRun('format-demo'));
    await page.locator('#finalAnswer strong').first().waitFor();
    assert.equal(await page.locator('#finalAnswer ul').count(),2);
    assert.equal(await page.locator('#finalAnswer li').count(),4);
    assert.equal(await page.locator('#finalAnswer > p').count(),5);
    assert.equal(await page.locator('#conversationHistory .answer-prose ol li').count(),2);
    assert.equal(await page.locator('#turnElapsed').innerText(),'用时 3m 54s');
    assert.equal(await page.locator('#finalAnswer').evaluate(node=>getComputedStyle(node).fontSize),'16px');
    assert.equal(await page.locator('#finalAnswer').evaluate(node=>node.tagName),'DIV');
    assert.equal(await page.locator('#finalAnswer').evaluate(node=>node.querySelector('p').getBoundingClientRect().height > 0),true);
    await page.evaluate(()=>{window.answerFirstNode=document.querySelector('#finalAnswer').firstChild;});
    await page.evaluate(()=>pollRun('format-demo'));
    assert(await page.evaluate(()=>window.answerFirstNode === document.querySelector('#finalAnswer').firstChild));
    await page.locator('#planDetails').evaluate(node=>node.open=false);
    await page.locator('#toolDetails').evaluate(node=>node.open=false);
    await page.locator('#runDrawer').evaluate(node=>node.scrollTop=node.scrollHeight);
    const output = process.env.GAMELENS_SCREENSHOTS;
    if (output) {
      fs.mkdirSync(output,{recursive:true});
      await page.screenshot({path:path.join(output,'answer-desktop.png')});
      await page.locator('#finalAnswerPanel').screenshot({path:path.join(output,'answer-detail.png')});
    }
    for (const width of [320,390,768]) {
      await page.setViewportSize({width,height:1100});
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      const fits = await page.locator('#finalAnswer').evaluate(node=>node.scrollWidth<=node.clientWidth);
      assert(fits,`Formatted answer must reflow at ${width}px`);
      if (output && width===390) await page.screenshot({path:path.join(output,'answer-mobile.png')});
    }
    await page.setViewportSize({width:1440,height:1100});
    await page.evaluate(old=>window.GameLensMarkdown.render(document.querySelector('#finalAnswer'),old),old);
    assert(await page.locator('#finalAnswer > p').count()>2);
    const displayed = await page.locator('#finalAnswer').textContent();
    assert.equal(displayed,old); // All content unchanged: DOM boundaries only.
    const quote = '玩家说“'+'一句话。'.repeat(65)+'不要拆开。”';
    await page.evaluate(text=>window.GameLensMarkdown.render(document.querySelector('#finalAnswer'),text),quote);
    assert.equal(await page.locator('#finalAnswer > p').count(),1);
    const fenced = '代码里面的长段落。'.repeat(50) + '\n\n' + '代码的第二个段落。'.repeat(50);
    await page.evaluate(text=>window.GameLensMarkdown.render(document.querySelector('#finalAnswer'),'```\n'+text+'\n```'),fenced);
    assert.equal(await page.locator('#finalAnswer pre code').textContent(),fenced);
    const literal = '# 标题\n\n3. **重点**与*强调*\n   - 嵌套项目\n4. 第二项\n\n> 逐字引用\n\n`<script>`\n\n```json\n{"raw":"<img onerror=alert(1)>"}\n```\n\n[来源](https://example.com/report)\n\n---';
    await page.evaluate(text=>window.GameLensMarkdown.render(document.querySelector('#finalAnswer'),text),literal);
    assert.equal(await page.locator('#finalAnswer h3').innerText(),'标题');
    assert.equal(await page.locator('#finalAnswer > ol').getAttribute('start'),'3');
    assert.equal(await page.locator('#finalAnswer ol ul li').innerText(),'嵌套项目');
    assert.equal(await page.locator('#finalAnswer blockquote').innerText(),'逐字引用');
    assert.equal(await page.locator('#finalAnswer pre code').innerText(),'{"raw":"<img onerror=alert(1)>"}');
    assert.equal(await page.locator('#finalAnswer a').getAttribute('rel'),'noopener noreferrer');
    await page.locator('#finalAnswer a').focus();
    assert.equal(await page.locator('#finalAnswer a').evaluate(node=>document.activeElement === node),true);
    const bad = '<script>window.pwned=1</script>\n\n<img src="https://evil.test/image" onerror="window.pwned=1">\n\n' +
      '[恶意](javascript:alert(1)) [数据](data:text/html,bad) [本地](file:///secret) [伪装](//evil.test) ' +
      '[属性](https://example.com/\"onclick=bad) ![图片](https://evil.test/image)\n\n' +
      '[正常](https://example.com/\"onmouseover=\"alert(1))';
    await page.evaluate(text=>window.GameLensMarkdown.render(document.querySelector('#finalAnswer'),text),bad);
    assert.equal(await page.locator('#finalAnswer script,#finalAnswer img,#finalAnswer iframe').count(),0);
    assert.equal(await page.locator('#finalAnswer [onerror],#finalAnswer [onclick],#finalAnswer [onmouseover]').count(),0);
    assert.equal(await page.evaluate(()=>Boolean(window.pwned)),false);
    assert.equal(await page.locator('#finalAnswer a').evaluateAll(nodes=>nodes.every(n=>/^https?:/.test(n.href))),true);
    assert.equal(requests.some(url=>url.includes('evil.test') || url.includes('example.com')),false);
    await page.evaluate(()=>window.GameLensMarkdown.render(document.querySelector('#finalAnswer'),''));
    assert.equal(await page.locator('#finalAnswer').textContent(),'');
    assert.equal(writes,0); assert.deepEqual(errors,[]);
    console.log('PASS: formatted current/history answers, legacy paragraphs, safe Markdown, XSS/URL isolation, stable polling, responsive typography; zero live API calls.');
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
