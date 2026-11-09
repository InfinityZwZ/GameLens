// Local two-note completion chime; no downloads or external services.
class CompletionSound {
  constructor() {
    this.enabled = true;
    this.statuses = new Map();
    this.pending = new Set();
    this.notified = new Set();
    this.context = null;
    this.nextSoundTime = 0;
    try {
      this.enabled = localStorage.getItem('patchpulse.completionSound') !== 'off';
      this.notified = new Set(JSON.parse(localStorage.getItem('patchpulse.soundNotified') || '[]'));
    } catch (_) {}
  }

  setEnabled(enabled) {
    this.enabled = enabled;
    if (!enabled) this.pending.clear();
    try { localStorage.setItem('patchpulse.completionSound', enabled ? 'on' : 'off'); } catch (_) {}
    if (enabled) this.unlock();
  }

  async unlock() {
    if (!this.enabled) return;
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (!Audio) return;
    try {
      this.context = this.context || new Audio();
      if (this.context.state === 'suspended') await this.context.resume();
      if (this.context.state === 'running') {
        for (const id of this.pending) this.play();
        this.pending.clear();
      }
    } catch (_) {}
  }

  observe(runs) {
    const active = new Set(['queued', 'running', 'awaiting_approval']);
    for (const run of runs) {
      const before = this.statuses.get(run.run_id);
      this.statuses.set(run.run_id, run.status);
      if (run.status !== 'completed' || !active.has(before) || this.notified.has(run.run_id)) continue;
      this.notified.add(run.run_id);
      try { localStorage.setItem('patchpulse.soundNotified', JSON.stringify([...this.notified].slice(-200))); } catch (_) {}
      if (!this.enabled) continue;
      if (this.context?.state === 'running') this.play();
      else this.pending.add(run.run_id);
    }
    if (this.statuses.size > 300) {
      for (const [id,status] of this.statuses) {
        if (!active.has(status) && !this.pending.has(id)) this.statuses.delete(id);
        if (this.statuses.size <= 200) break;
      }
    }
  }

  play() {
    if (!this.enabled || this.context?.state !== 'running') return;
    const start = Math.max(this.context.currentTime, this.nextSoundTime);
    this.nextSoundTime = start + 0.5;
    [660,880].forEach((frequency,index) => {
      const oscillator = this.context.createOscillator();
      const gain = this.context.createGain();
      const at = start + index * 0.15;
      oscillator.type = 'sine';
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0,at);
      gain.gain.linearRampToValueAtTime(0.075,at + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.001,at + 0.23);
      oscillator.connect(gain);
      gain.connect(this.context.destination);
      oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
      oscillator.start(at);
      oscillator.stop(at + 0.25);
    });
  }
}
window.CompletionSound = CompletionSound;
