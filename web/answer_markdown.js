/* Safe answer-only Markdown subset, without CDN, HTML parsing, or model calls.
 * Supported: paragraphs, headings, lists, quotes, rules, emphasis, code, links.
 * Other syntax remains text. Raw HTML/images never create DOM or fetch sources.
 */
(() => {
  "use strict";
  const rendered = new WeakMap();
  const listItem = line => /^( *)([-+*]|\d{1,9}[.)])\s+(.+)$/.exec(line);
  const heading = line => /^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
  const fence = line => /^ {0,3}(`{3,}|~{3,})[^`]*$/.exec(line);
  const rule = line => /^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line);
  const blockStart = line => heading(line) || fence(line) || listItem(line) || /^\s*>/.test(line) || rule(line);

  function safeLink(raw) {
    // No relative, data, file, javascript, protocol-relative, or control URLs.
    if (!/^https?:\/\//i.test(raw) || /[\s\u0000-\u001f\u007f]/.test(raw)) return null;
    try {
      const url = new URL(raw);
      return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null;
    } catch (_) { return null; }
  }

  function inline(parent, source, depth = 0, allowLinks = true) {
    if (depth > 8) { parent.append(document.createTextNode(source)); return; }
    let cursor = 0, plain = "";
    const flush = () => { if (plain) { parent.append(document.createTextNode(plain)); plain = ""; } };
    while (cursor < source.length) {
      const tail = source.slice(cursor);
      let match, tag;
      if (tail[0] === "\\" && /^\\[\\`*_{}\[\]()#+.!>~-]/.test(tail)) {
        plain += tail[1]; cursor += 2; continue;
      }
      if (tail[0] === "\n") {
        flush(); parent.append(document.createElement("br")); cursor++; continue;
      }
      // Code is literal, including Markdown and HTML inside it.
      if ((match = /^(`+)([\s\S]*?)\1(?!`)/.exec(tail))) {
        flush(); const node = document.createElement("code"); node.textContent = match[2]; parent.append(node);
        cursor += match[0].length; continue;
      }
      // Images are displayed literally, never loaded. Keep the entire token intact.
      if ((match = /^!\[[^\]\n]*\]\([^\s\n]*\)/.exec(tail))) {
        plain += match[0]; cursor += match[0].length; continue;
      }
      if (allowLinks && (match = /^\[([^\]\n]+)\]\(([^\s\n]+?)\)/.exec(tail))) {
        const href = safeLink(match[2]);
        if (href) {
          flush(); const node = document.createElement("a"); node.href = href;
          node.target = "_blank"; node.rel = "noopener noreferrer";
          inline(node, match[1], depth + 1, false); parent.append(node);
        } else { plain += match[0]; }
        cursor += match[0].length; continue;
      }
      if ((match = /^\*\*(?=\S)([\s\S]*?\S)\*\*/.exec(tail)) ||
          (match = /^__(?=\S)([\s\S]*?\S)__/.exec(tail))) tag = "strong";
      else if ((match = /^\*(?=\S)([^*\n]*?\S)\*/.exec(tail))) tag = "em";
      if (tag) {
        flush(); const node = document.createElement(tag); inline(node, match[1], depth + 1, allowLinks);
        parent.append(node); cursor += match[0].length; continue;
      }
      plain += source[cursor++];
    }
    flush();
  }

  function blocks(parent, lines, depth = 0) {
    if (depth > 8) { const p = document.createElement("p"); p.textContent = lines.join("\n"); parent.append(p); return; }
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) { index++; continue; }
      const codeFence = fence(line);
      if (codeFence) {
        const body = [], marker = codeFence[1]; index++;
        const closing = new RegExp("^ {0,3}" + marker[0] + "{" + marker.length + ",}\\s*$");
        while (index < lines.length && !closing.test(lines[index])) body.push(lines[index++]);
        if (index < lines.length) index++;
        const pre = document.createElement("pre"), code = document.createElement("code");
        code.textContent = body.join("\n"); pre.append(code); parent.append(pre); continue;
      }
      const title = heading(line);
      if (title) {
        // The answer sits under the page h1 and the investigation h2.
        const node = document.createElement("h" + Math.min(6, title[1].length + 2));
        inline(node, title[2]); parent.append(node); index++; continue;
      }
      if (rule(line)) { parent.append(document.createElement("hr")); index++; continue; }
      if (/^\s*>/.test(line)) {
        const body = [];
        while (index < lines.length && /^\s*>/.test(lines[index])) body.push(lines[index++].replace(/^\s*> ?/, ""));
        const quote = document.createElement("blockquote"); blocks(quote, body, depth + 1); parent.append(quote); continue;
      }
      const first = listItem(line);
      if (first) {
        const base = first[1].length, ordered = /^\d/.test(first[2]);
        const list = document.createElement(ordered ? "ol" : "ul");
        if (ordered) list.start = Number.parseInt(first[2], 10);
        while (index < lines.length) {
          const item = listItem(lines[index]);
          if (!item || item[1].length !== base || /^\d/.test(item[2]) !== ordered) break;
          const contentIndent = item[0].length - item[3].length;
          const body = [item[3]]; index++;
          while (index < lines.length) {
            if (!lines[index].trim()) {
              let next = index + 1;
              while (next < lines.length && !lines[next].trim()) next++;
              const following = listItem(lines[next] || "");
              if (following && following[1].length === base) { index = next; break; }
              if (next < lines.length && /^ */.exec(lines[next])[0].length > base) {
                body.push(""); index++; continue;
              }
              break;
            }
            const indentation = /^ */.exec(lines[index])[0].length;
            if (indentation <= base) break;
            body.push(lines[index++].slice(Math.min(indentation, contentIndent)));
          }
          const li = document.createElement("li"); blocks(li, body, depth + 1); list.append(li);
        }
        parent.append(list); continue;
      }
      const paragraph = [line]; index++;
      while (index < lines.length && lines[index].trim() && !blockStart(lines[index])) paragraph.push(lines[index++]);
      const node = document.createElement("p"); inline(node, paragraph.join("\n")); parent.append(node);
    }
  }

  function legacyParagraphs(text) {
    // Old saved conclusions may have no breaks. Only insert display whitespace
    // after Chinese sentence endings outside quotes; never infer new findings.
    if (text.includes("`") || text.split("\n").some(blockStart)) return text;
    if (/\n\s*\n/.test(text)) return text.split(/\n[ \t]*\n/).map(legacyParagraphs).join("\n\n");
    if (text.length < 240) return text;
    const quotes = [], pairs = {"“":"”", "「":"」", "『":"』", "‘":"’"};
    let output = "", paragraphSize = 0;
    for (const character of text) {
      if (pairs[character]) quotes.push(pairs[character]);
      else if (character === quotes.at(-1)) quotes.pop();
      else if (character === '"') {
        if (quotes.at(-1) === '"') quotes.pop(); else quotes.push('"');
      }
      output += character; paragraphSize++;
      if (!quotes.length && /[。！？]/.test(character) && paragraphSize >= 120) {
        output += "\n\n"; paragraphSize = 0;
      }
    }
    return output.trimEnd();
  }

  function render(container, value) {
    const original = String(value ?? "");
    // Polling should not replace unchanged content or disrupt selection/focus.
    if (rendered.get(container) === original && container.childNodes.length) return;
    const text = legacyParagraphs(original.replace(/\r\n?/g, "\n"));
    const fragment = document.createDocumentFragment();
    blocks(fragment, text.split("\n"));
    container.replaceChildren(fragment); rendered.set(container, original);
  }

  window.GameLensMarkdown = Object.freeze({render});
})();
