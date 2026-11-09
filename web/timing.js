/* Content-free per-turn timing. Server values are authoritative. */
(function (root) {
  function duration(ms) {
    if (ms === null || ms === undefined || !Number.isFinite(Number(ms))) return "未知";
    const seconds = Math.max(0, Math.floor(Number(ms) / 1000));
    const hours = Math.floor(seconds / 3600), minutes = Math.floor(seconds % 3600 / 60);
    return `${hours ? `${hours}h ` : ""}${minutes || hours ? `${minutes}m ` : ""}${seconds % 60}s`;
  }
  function fallback(run) {
    if (run.timing) return run.timing;
    const start = Date.parse(run.started_at), end = Date.parse(run.finished_at || run.updated_at);
    return {elapsed_ms:Number.isFinite(start) && Number.isFinite(end) ? Math.max(0,end-start) : null};
  }
  root.GameLensTiming = {duration, fallback};
})(typeof window !== "undefined" ? window : globalThis);
