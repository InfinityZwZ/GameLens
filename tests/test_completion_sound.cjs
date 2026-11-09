const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

async function main() {
  const values = new Map();
  const scheduled = [];
  class Audio {
    constructor() { this.state = 'running'; this.currentTime = 0; this.destination = {}; }
    async resume() { this.state = 'running'; }
    createOscillator() { return {frequency:{value:0},connect(){},disconnect(){},start(at){ scheduled.push(at); },stop(){}}; }
    createGain() { return {gain:{setValueAtTime(){},linearRampToValueAtTime(){},exponentialRampToValueAtTime(){}},connect(){},disconnect(){}}; }
  }
  const sandbox = {window:{AudioContext:Audio}, localStorage:{getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value)}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../web/completion_sound.js'),'utf8'),sandbox);
  const sound = new sandbox.window.CompletionSound();
  await sound.unlock();
  sound.observe([{run_id:'old',status:'completed'}]);
  assert.equal(scheduled.length,0); // Viewing historical completion stays silent.
  sound.observe([{run_id:'fresh',status:'queued'}]);
  sound.observe([{run_id:'fresh',status:'completed'}]);
  sound.observe([{run_id:'fresh',status:'completed'}]);
  assert.equal(scheduled.length,2); // Exactly one two-note notification.
  sound.setEnabled(false);
  sound.observe([{run_id:'muted',status:'running'}]);
  sound.observe([{run_id:'muted',status:'completed'}]);
  assert.equal(scheduled.length,2);
  const restored = new sandbox.window.CompletionSound();
  assert.equal(restored.enabled,false);
  restored.setEnabled(true);
  await restored.unlock();
  restored.observe([{run_id:'fresh',status:'running'}]);
  restored.observe([{run_id:'fresh',status:'completed'}]);
  restored.observe([{run_id:'cancelled',status:'running'}]);
  restored.observe([{run_id:'cancelled',status:'cancelled'}]);
  assert.equal(scheduled.length,2); // Reload and cancellation stay silent.
  const suspended = new sandbox.window.CompletionSound();
  suspended.observe([{run_id:'background',status:'running'}]);
  suspended.observe([{run_id:'background',status:'completed'}]);
  assert.equal(scheduled.length,2);
  await suspended.unlock();
  await suspended.unlock();
  assert.equal(scheduled.length,4); // Completion queued before browser activation plays once.
  console.log('PASS: one-shot completion, historical silence, mute persistence, cancellation, and deferred audio activation.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
