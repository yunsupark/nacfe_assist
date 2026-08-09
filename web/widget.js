// NACFE Knowledge Base -- embeddable Q&A widget.
//
// Drop this on any page:
//   <div id="nacfe-assist"></div>
//   <script src="https://<wherever-this-file-is-hosted>/widget.js"
//           data-api="https://nacfe-assist.nacfe.workers.dev/query" defer></script>
//
// `data-api` defaults to the value below if omitted. Everything (CSS, DOM, fetch logic) is
// self-contained in this one file, scoped under #nacfe-assist-root, so it cannot collide with
// the host page's styles or scripts. No dependencies, no build step.
(function () {
  "use strict";

  var DEFAULT_API = "https://nacfe-assist.nacfe.workers.dev/query";
  var script = document.currentScript;
  var apiUrl = (script && script.getAttribute("data-api")) || DEFAULT_API;
  var mount = document.getElementById("nacfe-assist") || (function () {
    var el = document.createElement("div");
    el.id = "nacfe-assist";
    (script && script.parentNode ? script.parentNode : document.body).appendChild(el);
    return el;
  })();

  var STYLE_ID = "nacfe-assist-style";
  if (!document.getElementById(STYLE_ID)) {
    var style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = [
      "#nacfe-assist-root{font-family:Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a;",
      "max-width:720px;margin:0 auto;box-sizing:border-box}",
      "#nacfe-assist-root *{box-sizing:border-box}",
      "#nacfe-assist-root .na-label{font-size:13px;font-weight:700;letter-spacing:.04em;",
      "text-transform:uppercase;color:#ab1428;margin:0 0 8px}",
      "#nacfe-assist-root form{display:flex;gap:8px;margin:0 0 4px}",
      "#nacfe-assist-root input[type=text]{flex:1;padding:11px 14px;font-size:15px;",
      "border:1px solid #c7c7c7;border-radius:4px;font-family:inherit;outline:none}",
      "#nacfe-assist-root input[type=text]:focus{border-color:#ab1428;",
      "box-shadow:0 0 0 2px rgba(171,20,40,.15)}",
      "#nacfe-assist-root button{padding:11px 20px;font-size:15px;font-weight:600;",
      "background:#ab1428;color:#fff;border:none;border-radius:4px;cursor:pointer;",
      "font-family:inherit;white-space:nowrap}",
      "#nacfe-assist-root button:hover:not(:disabled){background:#8f1121}",
      "#nacfe-assist-root button:disabled{background:#c7c7c7;cursor:default}",
      "#nacfe-assist-root .na-hint{font-size:12px;color:#6b6b6b;margin:0 0 20px}",
      "#nacfe-assist-root .na-result{border-top:1px solid #e2e2e2;padding-top:16px;",
      "margin-top:16px;display:none}",
      "#nacfe-assist-root .na-result.na-visible{display:block}",
      "#nacfe-assist-root .na-answer{font-size:15px;line-height:1.6;white-space:pre-wrap}",
      "#nacfe-assist-root .na-answer p{margin:0 0 12px}",
      "#nacfe-assist-root .na-warning{background:#fff5e6;border:1px solid #f0c987;",
      "border-radius:4px;padding:10px 12px;font-size:13px;margin:0 0 14px;color:#7a4e00}",
      "#nacfe-assist-root .na-sources{margin-top:16px}",
      "#nacfe-assist-root .na-sources-label{font-size:12px;font-weight:700;",
      "text-transform:uppercase;letter-spacing:.04em;color:#6b6b6b;margin:0 0 8px}",
      "#nacfe-assist-root .na-source{font-size:13px;padding:8px 10px;background:#f7f7f7;",
      "border-radius:4px;margin:0 0 6px;color:#333}",
      "#nacfe-assist-root .na-source b{color:#001961}",
      "#nacfe-assist-root .na-loading{font-size:14px;color:#6b6b6b;padding:12px 0}",
      "#nacfe-assist-root .na-error{font-size:14px;color:#ab1428;padding:12px 0}",
      "#nacfe-assist-root .na-footer{font-size:11px;color:#9a9a9a;margin-top:18px}",
      "#nacfe-assist-root .na-footer a{color:#001961}",
    ].join("");
    document.head.appendChild(style);
  }

  mount.innerHTML = [
    '<div id="nacfe-assist-root">',
    '<p class="na-label">Ask NACFE\'s Research</p>',
    '<form id="na-form">',
    '<input type="text" id="na-input" placeholder="e.g. What was Frito-Lay’s fuel economy in the Messy Middle demonstration?" autocomplete="off" />',
    '<button type="submit" id="na-submit">Ask</button>',
    '</form>',
    '<p class="na-hint">Answers are grounded only in NACFE’s published reports, videos, and studies — not general knowledge.</p>',
    '<div class="na-loading" id="na-loading" style="display:none">Searching NACFE’s research library…</div>',
    '<div class="na-error" id="na-error" style="display:none"></div>',
    '<div class="na-result" id="na-result">',
    '<div class="na-warning" id="na-warning" style="display:none"></div>',
    '<div class="na-answer" id="na-answer"></div>',
    '<div class="na-sources" id="na-sources" style="display:none">',
    '<p class="na-sources-label">Sources</p>',
    '<div id="na-sources-list"></div>',
    '</div>',
    '</div>',
    '<p class="na-footer">Powered by NACFE’s research library. Answers cite specific reports and demonstrations — verify against the original source for critical decisions.</p>',
    '</div>',
  ].join("");

  var form = mount.querySelector("#na-form");
  var input = mount.querySelector("#na-input");
  var submitBtn = mount.querySelector("#na-submit");
  var loadingEl = mount.querySelector("#na-loading");
  var errorEl = mount.querySelector("#na-error");
  var resultEl = mount.querySelector("#na-result");
  var warningEl = mount.querySelector("#na-warning");
  var answerEl = mount.querySelector("#na-answer");
  var sourcesEl = mount.querySelector("#na-sources");
  var sourcesListEl = mount.querySelector("#na-sources-list");

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  // The answer stage writes plain paragraphs separated by blank lines -- render each as
  // its own <p> rather than dumping one unbroken block of text.
  function renderAnswer(text) {
    var paragraphs = text.split(/\n\s*\n/).filter(function (p) { return p.trim(); });
    answerEl.innerHTML = paragraphs.map(function (p) {
      return "<p>" + escapeHtml(p.trim()) + "</p>";
    }).join("");
  }

  function renderSources(sources) {
    if (!sources || !sources.length) {
      sourcesEl.style.display = "none";
      return;
    }
    sourcesListEl.innerHTML = sources.map(function (s) {
      return '<div class="na-source"><b>' + escapeHtml(s.id) + "</b> — " +
        escapeHtml(s.why || "") + "</div>";
    }).join("");
    sourcesEl.style.display = "block";
  }

  function setLoading(isLoading) {
    loadingEl.style.display = isLoading ? "block" : "none";
    submitBtn.disabled = isLoading;
    input.disabled = isLoading;
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var question = input.value.trim();
    if (!question) return;

    errorEl.style.display = "none";
    resultEl.classList.remove("na-visible");
    setLoading(true);

    fetch(apiUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: question }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().catch(function () { return {}; }).then(function (body) {
            throw new Error(body.error || ("request failed (" + res.status + ")"));
          });
        }
        return res.json();
      })
      .then(function (data) {
        setLoading(false);
        renderAnswer(data.answer || "");
        renderSources(data.selected_sources);
        if (data.recency_warning) {
          warningEl.textContent = data.recency_warning;
          warningEl.style.display = "block";
        } else {
          warningEl.style.display = "none";
        }
        resultEl.classList.add("na-visible");
      })
      .catch(function (err) {
        setLoading(false);
        errorEl.textContent = "Something went wrong: " + err.message;
        errorEl.style.display = "block";
      });
  });
})();
