// All requests intercepted; no server, account, source scrape, or model invocation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');

async function main() {
  const web = path.resolve(__dirname,'../web');
  const browser = await chromium.launch({channel:'chrome',headless:true,args:['--mute-audio']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:960},timezoneId:'America/Los_Angeles'});
    const errors=[]; let writes=0;
    page.on('pageerror',error=>errors.push(error.message));
    const run={run_id:'announcement-demo',conversation_id:'announcement-session',turn_index:1,status:'completed',investigation_outcome:'partial_result',
      request:{appid:730,goal:'离线界面演示：调查2026-10-08更新公告下的玩家反馈'}, plan:[],logs:[],
      final_answer:'**主要发现**\n\n- 玩家留言中存在不同体验。\n\n**范围与限制**\n\n仅为已采集的公告讨论快照。',
      events:[{step:1,tool:'query_announcement_comments',ok:true,summary:'离线公告来源展示',data:{
        binding_verified:true,news_id:'1001',coverage:{scope:'announcement_discussion_only',collected_count:20,reported_total:80,complete:false},
        comments:[{news_id:'1001',comment_id:'2001',text:'<script>unsafe</script>更新后有卡顿反馈',created_at:'2026-10-08T23:00:00Z',created_at_local:'2026-10-09T10:00:00+11:00',contains_quoted_text:true}],
        announcements:[{news_id:'1001',title:'离线更新公告',contents:'官方更新正文'}]}}],
      artifacts:{final_report:'D:/GameReview_Agent/reports/730_announcement-demo_complete_report.md'}};
    await page.route('**/*',async route=>{
      const request=route.request(),url=new URL(request.url());
      if(request.method() !== 'GET') writes++;
      if(url.pathname.startsWith('/api/')) {
        if(url.pathname.endsWith('/report')) return route.fulfill({contentType:'text/plain',body:'# 离线公告反馈报告\n\n仅用于验证界面。'});
        let data=[];
        if(url.pathname === '/api/runs/announcement-demo') data=run;
        if(url.pathname === '/api/conversations') data=[{conversation_id:'announcement-session',last_run_id:run.run_id,
          title:run.request.goal,status:run.status,investigation_outcome:run.investigation_outcome,appid:730,turn_count:1}];
        if(url.pathname === '/api/conversations/announcement-session') data={title:run.request.goal,turns:[run]};
        return route.fulfill({contentType:'application/json',body:JSON.stringify(data)});
      }
      const file=path.resolve(web,url.pathname==='/'?'index.html':url.pathname.replace(/^\/static\//,''));
      if(!file.startsWith(web+path.sep)||!fs.existsSync(file)) return route.abort();
      await route.fulfill({path:file,contentType:{'.html':'text/html','.js':'application/javascript','.css':'text/css','.svg':'image/svg+xml'}[path.extname(file)]});
    });
    await page.goto('http://gamelens.test/');
    const template=page.getByRole('button',{name:'调查玩家对某日期更新版本的反馈内容',exact:true});
    await template.waitFor();
    assert.equal(await page.locator('.suggestion').count(),4);
    await template.click();
    assert(await page.locator('#goalInput').inputValue().then(value=>value.includes('公告下的玩家留言')));
    assert.equal(await page.locator('#goalInput').evaluate(node=>node.value.slice(node.selectionStart,node.selectionEnd)),'YYYY-MM-DD');
    assert.equal(writes,0);
    const output=process.env.GAMELENS_SCREENSHOTS;
    if(output) {fs.mkdirSync(output,{recursive:true});await page.screenshot({path:path.join(output,'announcement-template-desktop.png')});}
    for(const width of [320,390,768,1440]) {
      await page.setViewportSize({width,height:960});
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      assert(await template.evaluate(node=>node.scrollWidth<=node.clientWidth && node.getBoundingClientRect().height>=44));
      if(output&&width===390) await page.screenshot({path:path.join(output,'announcement-template-mobile.png')});
    }
    await page.evaluate(()=>openSavedRun('announcement-demo'));
    await page.waitForFunction(()=>document.querySelector('#positiveDelta').textContent==='20');
    assert.equal(await page.locator('#runStatus').innerText(),'有限结果');
    assert(await page.locator('#runHistory').innerText().then(text=>text.includes('有限结果')));
    assert(await page.locator('#toast').innerText().then(text=>text.includes('有限结果')));
    const labels=await page.evaluate(()=>['partial_result','budget_exhausted','needs_clarification','insufficient_evidence','needs_verification','complete',null]
      .map(outcome=>runStatusLabel({status:'completed',investigation_outcome:outcome})));
    assert.deepEqual(labels,['有限结果','调查未完成','待确认','调查未完成','待核验','已完成','已完成']);
    await page.evaluate(()=>renderConversationHistory({turns:[{run_id:'previous',turn_index:1,status:'completed',
      investigation_progress:{outcome:'budget_exhausted'},request:{goal:'离线未完成记录'},final_answer:'未完成'}]},'other'));
    assert(await page.locator('#conversationHistory').innerText().then(text=>text.includes('调查未完成')));
    await page.locator('#toggleInspector').click();
    await page.locator('#tabOverview').click();
    assert.equal(await page.locator('#volumeDelta').innerText(),'80');
    assert.equal(await page.locator('#positiveRates').innerText(),'部分采集，不代表全部留言');
    await page.locator('#tabEvidence').click();
    assert(await page.locator('#evidenceGrid').innerText().then(text=>text.includes('非商店评价')&&text.includes('含引用文字')));
    assert(await page.locator('#evidenceGrid').innerText().then(text=>text.includes('10/09 10:00')));
    assert(await page.locator('#searchHelp').innerText().then(text=>text.includes('系统本地时间')));
    assert.equal(await page.locator('#evidenceGrid script').count(),0);
    if(output) await page.screenshot({path:path.join(output,'announcement-evidence-desktop.png')});
    assert.equal(writes,0);assert.deepEqual(errors,[]);
    console.log('PASS: fourth template, mobile layout, independent evidence, honest outcomes in header/history/turns, server-local date, escaped content; zero live calls.');
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
