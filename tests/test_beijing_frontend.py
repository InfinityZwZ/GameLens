"""Execute real frontend date formatters under a non-Beijing browser clock."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]

def test_frontend_formats_all_offsets_and_naive_dates_as_beijing():
    node=shutil.which('node')
    bundled=(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
    node=node or (str(bundled) if bundled.is_file() else None)
    if not node:pytest.skip('Node.js is not installed; frontend runtime test not measured')
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const start=source.includes('function beijingDateParts')?source.indexOf('function beijingDateParts'):source.indexOf('function shortTime');
const end=source.indexOf('async function loadRunHistory',start);
const element={textContent:''};const context=vm.createContext({Intl,Date,Map,Number,$:()=>element});
vm.runInContext('const serverLocalTimes=new Map();\n'+source.slice(start,end),context);
assert.equal(context.shortTime('2026-10-08T22:36:49Z'),'10/09 06:36');
assert.equal(context.shortTime('2026-10-09T09:36:49+11:00'),'10/09 06:36');
assert.equal(context.localDateText('2026-12-31T18:00:00Z',true),'2027-01-01 02:00');
assert.equal(context.localDateText('2026-10-09T00:00:00',true),'2026-10-09 00:00');
assert.equal(context.localDateText('2026-10-09',true),'2026-10-09 00:00');
assert.equal(context.shortTime('invalid'),'—');
assert.equal(context.localDateText(null),'—');
const clock=source.slice(source.indexOf('function updateClock()'),source.indexOf('async function boot()'));
vm.runInContext(clock,context);context.updateClock();
assert.equal(element.textContent,new Date().toLocaleTimeString('en-GB',{hour12:false,timeZone:'Asia/Shanghai'}));
console.log('Beijing frontend runtime assertions passed (host TZ='+process.env.TZ+')');
'''
    result=subprocess.run([node,'-e',script,str(ROOT/'web/app.js')],env={**os.environ,'TZ':'America/Los_Angeles'},
        capture_output=True,text=True,encoding='utf-8',timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
