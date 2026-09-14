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
      "text-transform:uppercase;color:#ab1428;margin:0}",
      "#nacfe-assist-root .na-label-row{display:flex;align-items:center;",
      "justify-content:space-between;gap:8px;margin:0 0 8px}",
      "#nacfe-assist-root .na-info-btn{width:18px;height:18px;flex-shrink:0;border-radius:50%;",
      "border:1px solid #c7c7c7;background:#fff;color:#6b6b6b;font-size:12px;font-weight:700;",
      "line-height:1;cursor:pointer;padding:0;display:flex;align-items:center;",
      "justify-content:center;font-family:Georgia,'Times New Roman',serif;font-style:italic}",
      "#nacfe-assist-root .na-info-btn:hover{border-color:#ab1428;color:#ab1428}",
      "#nacfe-assist-root .na-info{font-size:13px;line-height:1.6;color:#444;",
      "background:#f7f7f7;border-radius:4px;padding:10px 12px;margin:0 0 14px}",
      "#nacfe-assist-root .na-info p{margin:0 0 8px}",
      "#nacfe-assist-root .na-info p:last-child{margin-bottom:0}",
      "#nacfe-assist-root .na-info ul{margin:0 0 8px;padding-left:20px}",
      "#nacfe-assist-root .na-info li{margin:0 0 2px}",
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
      "#nacfe-assist-root .na-notice{background:#fff5e6;border:1px solid #f0c987;",
      "border-radius:4px;padding:10px 12px;font-size:13px;margin:0 0 14px;color:#7a4e00}",
      "#nacfe-assist-root .na-notice b{color:#5c3b00}",
      "#nacfe-assist-root .na-sponsor{display:flex;align-items:center;gap:10px;",
      "font-size:13px;color:#6b6b6b;margin:18px 0 0;padding-top:14px;",
      "border-top:1px solid #e2e2e2}",
      "#nacfe-assist-root .na-sponsor b{color:#333;font-weight:600}",
      "#nacfe-assist-root .na-sponsor a{color:#001961}",
      "#nacfe-assist-root .na-sponsor img{width:24px;height:24px;border-radius:4px;",
      "object-fit:contain;flex-shrink:0}",
      "#nacfe-assist-root .na-turnstile{margin:0 0 10px}",
      "#nacfe-assist-root .na-turnstile:empty{margin:0}",
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
      "#nacfe-assist-root .na-source b a{color:inherit;text-decoration:none}",
      "#nacfe-assist-root .na-source b a:hover{text-decoration:underline}",
      "#nacfe-assist-root .na-feedback{display:flex;align-items:center;gap:10px;",
      "margin-top:18px;padding-top:14px;border-top:1px solid #e2e2e2}",
      "#nacfe-assist-root .na-feedback-label{font-size:13px;color:#6b6b6b}",
      "#nacfe-assist-root .na-feedback-btn{padding:6px 12px;font-size:13px;font-weight:600;",
      "background:#fff;color:#333;border:1px solid #c7c7c7;border-radius:20px;cursor:pointer;",
      "font-family:inherit;white-space:nowrap}",
      "#nacfe-assist-root .na-feedback-btn:hover:not(:disabled){border-color:#ab1428;",
      "color:#ab1428}",
      "#nacfe-assist-root .na-feedback-btn:disabled{cursor:default;opacity:.5}",
      "#nacfe-assist-root .na-feedback-btn.na-selected{background:#ab1428;color:#fff;",
      "border-color:#ab1428;opacity:1}",
      "#nacfe-assist-root .na-feedback-thanks{font-size:13px;color:#6b6b6b;",
      "font-style:italic}",
      "#nacfe-assist-root .na-history{margin-top:14px}",
      "#nacfe-assist-root .na-history-item{border-top:1px solid #e2e2e2;padding:10px 0}",
      "#nacfe-assist-root .na-history-item summary{cursor:pointer;font-size:14px;",
      "font-weight:600;color:#333;list-style:none}",
      "#nacfe-assist-root .na-history-item summary::-webkit-details-marker{display:none}",
      "#nacfe-assist-root .na-history-item summary:before{content:\"\\25B8\\A0\";color:#ab1428}",
      "#nacfe-assist-root .na-history-item[open] summary:before{content:\"\\25BE\\A0\"}",
      "#nacfe-assist-root .na-history-body{margin-top:10px}",
      // A query genuinely takes 20-45s (median 21s, p90 46s measured in production), so this
      // has to read as deliberate work in progress rather than a stalled page. A moving bar
      // plus an elapsed counter is far more legible at that duration than three small dots.
      "#nacfe-assist-root .na-loading{margin:16px 0;padding:16px;border:1px solid #e2d2d4;",
      "border-radius:6px;background:#fdf7f8}",
      "#nacfe-assist-root .na-loading-bar{height:4px;border-radius:2px;background:#f0dfe2;",
      "overflow:hidden;margin:0 0 12px}",
      "#nacfe-assist-root .na-loading-fill{height:100%;width:40%;border-radius:2px;",
      "background:#ab1428;animation:na-sweep 1.4s ease-in-out infinite}",
      "@keyframes na-sweep{0%{transform:translateX(-100%)}100%{transform:translateX(350%)}}",
      "#nacfe-assist-root .na-loading-text{margin:0;font-size:15px;font-weight:600;color:#1a1a1a}",
      "#nacfe-assist-root .na-loading-sub{margin:4px 0 0;font-size:13px;color:#6b6b6b}",
      "@media (prefers-reduced-motion:reduce){",
      "#nacfe-assist-root .na-loading-fill{animation:none;width:100%;opacity:.5}}",
      "#nacfe-assist-root .na-error{font-size:14px;color:#ab1428;padding:12px 0}",
      "#nacfe-assist-root .na-footer{font-size:11px;color:#9a9a9a;margin-top:18px}",
      "#nacfe-assist-root .na-footer a{color:#001961}",
      "font-size:14px;color:#6b6b6b;margin:0 0 6px}",
      "flex-shrink:0;object-fit:contain}",
      // Desktop-width viewports get a wider column and larger type -- the base rules above
      // stay mobile-sized so phones/narrow embeds aren't affected.
      "@media (min-width:640px){",
      "#nacfe-assist-root{max-width:900px}",
      "#nacfe-assist-root .na-label{font-size:15px}",
      "#nacfe-assist-root input[type=text]{font-size:17px;padding:13px 16px}",
      "#nacfe-assist-root button{font-size:17px;padding:13px 24px}",
      "#nacfe-assist-root .na-hint{font-size:14px}",
      "#nacfe-assist-root .na-info{font-size:14px}",
      "#nacfe-assist-root .na-answer{font-size:18px;line-height:1.7}",
      "#nacfe-assist-root .na-warning{font-size:15px}",
      "#nacfe-assist-root .na-sources-label{font-size:14px}",
      "#nacfe-assist-root .na-source{font-size:15px;padding:10px 12px}",
      "#nacfe-assist-root .na-loading-text{font-size:17px}",
      "#nacfe-assist-root .na-error{font-size:16px}",
      "#nacfe-assist-root .na-feedback-label{font-size:15px}",
      "#nacfe-assist-root .na-feedback-btn{font-size:15px;padding:8px 16px}",
      "#nacfe-assist-root .na-history-item summary{font-size:16px}",
      "#nacfe-assist-root .na-feedback-thanks{font-size:15px}",
      "#nacfe-assist-root .na-footer{font-size:13px}",
      "}",
    ].join("");
    document.head.appendChild(style);
  }

  mount.innerHTML = [
    '<div id="nacfe-assist-root">',
    '<div class="na-label-row">',
    '<p class="na-label">Ask NACFE\'s Research</p>',
    '<button type="button" class="na-info-btn" id="na-info-btn" aria-expanded="false" aria-controls="na-info" aria-label="About this tool">i</button>',
    '</div>',
    '<div class="na-info" id="na-info" style="display:none">',
    '<p>Answers are drawn from these NACFE published sources:</p>',
    '<ul>',
    '<li>Current Technology (Confidence Reports)</li>',
    '<li>Emerging Technology (Guidance Reports)</li>',
    '<li>Fleet Efficiency Study (latest)</li>',
    '<li>Run on Less Reports</li>',
    '<li>Thought Leadership Reports</li>',
    '<li>Collaboration Reports</li>',
    '<li>Mike &amp; Friends Podcasts</li>',
    '<li>Run on Less Messy Middle Bootcamps</li>',
    '<li>Run on Less fleet profile videos</li>',
    '</ul>',
    '<p>Source library last updated 07/2026.</p>',
    '</div>',
    '<div class="na-notice" id="na-notice" role="status" style="display:none"></div>',
    '<form id="na-form">',
    '<input type="text" id="na-input" maxlength="1000" aria-label="Ask a question about NACFE\u2019s research" placeholder="e.g. What was Frito-Lay’s fuel economy in the Messy Middle demonstration?" autocomplete="off" />',
    '<button type="submit" id="na-submit">Ask</button>',
    '</form>',
    '<div class="na-turnstile" id="na-turnstile"></div>',
    '<p class="na-hint">Answers are grounded only in NACFE’s published reports, videos, and studies — not general knowledge.</p>',
    '<div class="na-loading" id="na-loading" role="status" aria-live="polite" style="display:none">',
    '<div class="na-loading-bar"><div class="na-loading-fill"></div></div>',
    '<p class="na-loading-text" id="na-loading-text">Searching NACFE’s research library</p>',
    '<p class="na-loading-sub">Reading full reports takes a moment — usually 20–45 seconds.<span id="na-loading-elapsed"></span></p>',
    '</div>',
    '<div class="na-error" id="na-error" role="alert" style="display:none"></div>',
    '<div class="na-result" id="na-result" aria-live="polite">',
    '<div class="na-warning" id="na-warning" style="display:none"></div>',
    '<div class="na-answer" id="na-answer"></div>',
    '<div class="na-sources" id="na-sources" style="display:none">',
    '<p class="na-sources-label">Sources</p>',
    '<div id="na-sources-list"></div>',
    '</div>',
    '<div class="na-feedback" id="na-feedback" style="display:none">',
    '<span class="na-feedback-label">Was this helpful?</span>',
    '<button type="button" class="na-feedback-btn" data-rating="yes">Yes</button>',
    '<button type="button" class="na-feedback-btn" data-rating="partly">Partly</button>',
    '<button type="button" class="na-feedback-btn" data-rating="no">No</button>',
    '<span class="na-feedback-thanks" id="na-feedback-thanks" style="display:none">Thanks for the feedback!</span>',
    '</div>',
    '</div>',
    '<div class="na-history" id="na-history"></div>',
    '<p class="na-sponsor" id="na-sponsor" style="display:none"></p>',
    '<p class="na-footer">Powered by NACFE’s research library. Answers cite specific reports and demonstrations — verify against the original source for critical decisions. Questions are logged (without any identifying information) to help NACFE see what the industry is asking; please don’t enter personal or confidential details.</p>',
    '</div>',
  ].join("");

  var form = mount.querySelector("#na-form");
  var input = mount.querySelector("#na-input");
  var submitBtn = mount.querySelector("#na-submit");
  var loadingEl = mount.querySelector("#na-loading");
  var loadingTextEl = mount.querySelector("#na-loading-text");
  var elapsedEl = mount.querySelector("#na-loading-elapsed");
  var errorEl = mount.querySelector("#na-error");
  var resultEl = mount.querySelector("#na-result");
  var warningEl = mount.querySelector("#na-warning");
  var answerEl = mount.querySelector("#na-answer");
  var sourcesEl = mount.querySelector("#na-sources");
  var sourcesListEl = mount.querySelector("#na-sources-list");
  var historyEl = mount.querySelector("#na-history");
  var feedbackEl = mount.querySelector("#na-feedback");
  var feedbackBtns = mount.querySelectorAll(".na-feedback-btn");
  var noticeEl = mount.querySelector("#na-notice");
  var infoBtn = mount.querySelector("#na-info-btn");
  var infoEl = mount.querySelector("#na-info");
  var sponsorEl = mount.querySelector("#na-sponsor");
  var feedbackThanksEl = mount.querySelector("#na-feedback-thanks");
  // Derive /feedback from /query, matching on the path only so a data-api carrying a query
  // string or a relative path still resolves. If the path doesn't end in /query there's
  // nothing sensible to derive, so leave feedback off rather than POSTing ratings at the
  // query endpoint (which the old blind `.replace()` would have done).
  // Same derivation as the feedback endpoint: match on path so a data-api carrying a query
  // string still resolves.
  function siblingEndpoint(name) {
    try {
      var parsed = new URL(apiUrl, document.baseURI);
      if (!/\/query$/.test(parsed.pathname)) return null;
      parsed.pathname = parsed.pathname.replace(/\/query$/, name);
      return parsed.toString();
    } catch (err) {
      return null;
    }
  }
  var statusApiUrl = siblingEndpoint("/status");
  var eventApiUrl = siblingEndpoint("/event");

  /**
   * Fire-and-forget analytics beacon. sendBeacon survives the page navigating away, which a
   * plain fetch does not -- without it the sponsor click, whose whole purpose is to be
   * followed by a navigation, would be the one event most likely to go unrecorded.
   */
  function sendEvent(type) {
    if (!eventApiUrl) {
      // Audible rather than silent: this returning early is exactly how a broken analytics
      // wire-up looks from the outside -- everything renders, nothing is ever counted.
      if (window.console && console.warn) console.warn("nacfe-assist: no /event endpoint; not counting " + type);
      return;
    }
    var payload = JSON.stringify({ type: type, page_url: location.origin + location.pathname });
    try {
      if (navigator.sendBeacon) {
        navigator.sendBeacon(eventApiUrl, new Blob([payload], { type: "application/json" }));
        return;
      }
      fetch(eventApiUrl, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: payload,
        keepalive: true,
      }).catch(function () {});
    } catch (err) {
      // analytics must never break the widget
    }
  }

  var feedbackApiUrl = (function () {
    try {
      var parsed = new URL(apiUrl, document.baseURI);
      if (!/\/query$/.test(parsed.pathname)) return null;
      parsed.pathname = parsed.pathname.replace(/\/query$/, "/feedback");
      return parsed.toString();
    } catch (err) {
      return null;
    }
  })();
  var currentQueryId = null;
  // Server-issued HMAC over the query id; the API rejects ratings without it, so that
  // sequential ids can't be enumerated and rated by anyone who never saw the answer.
  var currentFeedbackToken = null;
  var REQUEST_TIMEOUT_MS = 120000;
  // The question behind whatever answer is currently shown in #na-result, if any -- needed to
  // label it when the next submit archives it into history.
  var shownQuestion = null;
  // Prior turns sent to the API so a follow-up like "what about cng" can be answered with
  // awareness of what it's following up on. In-memory only -- lost on page reload, same as
  // the collapsed history above. Capped client-side (server enforces its own, independent
  // cap): unlike the free display history, every turn here gets replayed into the routing and
  // answering prompts on every later question, so an uncapped list would make a long
  // conversation's later questions progressively more expensive and slower.
  var CONVERSATION_TURNS_SENT = 4;
  var conversationHistory = [];

  // Turnstile. The sitekey is substituted by the Worker when it serves this file, so an
  // embedder never has to carry it; data-sitekey on the script tag overrides for local work.
  // Empty sitekey => the widget runs unprotected, which is the state a deployment is in
  // before the Turnstile secret is provisioned.
  var TURNSTILE_SITEKEY =
    (script && script.getAttribute("data-sitekey")) || "__TURNSTILE_SITEKEY__";
  if (TURNSTILE_SITEKEY.indexOf("__TURNSTILE") === 0) TURNSTILE_SITEKEY = "";
  var turnstileWidgetId = null;

  // Substituted by the Worker when it serves this file, same as the sitekey. Empty name means
  // there is no sponsor, so no block renders and no click tracking exists.
  var SPONSOR = {
    name: "__SPONSOR_NAME__", tagline: "__SPONSOR_TAGLINE__", url: "__SPONSOR_URL__",
    logoUrl: "__SPONSOR_LOGO_URL__",
  };
  if (SPONSOR.name.indexOf("__SPONSOR") === 0) SPONSOR = { name: "", tagline: "", url: "", logoUrl: "" };


  // Everything below builds DOM nodes and sets textContent rather than concatenating HTML
  // strings. The previous escapeHtml() escaped via textContent -> innerHTML, which handles
  // < > & but NOT quotes -- and its output was interpolated straight into href="...", one
  // character away from breaking out of the attribute. Nodes sidestep the whole class.

  /** Only ever render http(s) links: a "javascript:" or "data:" url in a source record would
   * otherwise become a live link in the reader's page. */
  // "resumes on 1 October" beats "next month": the ceiling resets at midnight UTC on the
  // first, and a reader deciding whether to come back deserves the actual date.
  function formatResetDate(iso) {
    if (!iso) return null;
    var d = new Date(iso);
    if (isNaN(d.getTime())) return null;
    try {
      // Formatted in UTC, matching the budget period, which is a UTC calendar month. Local
      // formatting renders a 1 October reset as "September 30" for any reader west of UTC --
      // locally true, but it names the wrong month and reads as a bug.
      return d.toLocaleDateString(undefined, { month: "long", day: "numeric", timeZone: "UTC" });
    } catch (err) {
      return d.toISOString().slice(0, 10);
    }
  }

  /** The one place budget-limited state is rendered, used both by the on-load status check
   * and by a refused query, so the reader never sees two differently-worded versions of it. */
  function showBudgetNotice(message, resetsAt) {
    if (!noticeEl) return;
    noticeEl.textContent = "";
    var strong = document.createElement("b");
    strong.textContent = "Monthly research budget reached. ";
    noticeEl.appendChild(strong);
    var when = formatResetDate(resetsAt);
    noticeEl.appendChild(document.createTextNode(
      message || ("Questions asked here before are still answered instantly" +
        (when ? "; new ones resume on " + when + "." : "; new ones resume when the budget resets."))
    ));
    noticeEl.style.display = "block";
  }

  function safeHttpUrl(u) {
    return typeof u === "string" && /^https?:\/\//i.test(u) ? u : null;
  }

  // The answer stage writes plain paragraphs separated by blank lines -- render each as
  // its own <p> rather than dumping one unbroken block of text.
  function renderAnswer(text) {
    answerEl.textContent = "";
    var paragraphs = String(text == null ? "" : text)
      .split(/\n\s*\n/)
      .filter(function (p) { return p.trim(); });
    for (var i = 0; i < paragraphs.length; i++) {
      var para = document.createElement("p");
      para.textContent = paragraphs[i].trim();
      answerEl.appendChild(para);
    }
  }

  function renderSources(sources) {
    sourcesListEl.textContent = "";
    if (!sources || !sources.length) {
      sourcesEl.style.display = "none";
      return;
    }
    for (var i = 0; i < sources.length; i++) {
      var s = sources[i] || {};
      var row = document.createElement("div");
      row.className = "na-source";

      var name = document.createElement("b");
      var url = safeHttpUrl(s.url);
      if (url) {
        var link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
        link.textContent = String(s.id || "");
        name.appendChild(link);
      } else {
        name.textContent = String(s.id || "");
      }
      row.appendChild(name);
      row.appendChild(document.createTextNode(" \u2014 " + String(s.why || "")));
      sourcesListEl.appendChild(row);
    }
    sourcesEl.style.display = "block";
  }

  // Nothing here is ever sent back to the server -- the API is stateless per question and has
  // no notion of a conversation, so this history is a purely client-side scrollback with no
  // per-turn cost and no cap needed on how many turns it holds.
  function stripIds(el) {
    el.removeAttribute("id");
    var withIds = el.querySelectorAll("[id]");
    for (var i = 0; i < withIds.length; i++) withIds[i].removeAttribute("id");
    return el;
  }

  // Snapshots the answer currently on screen into a collapsed entry before it's overwritten by
  // the next question. Cloned rather than re-rendered from the API response, so the archived
  // copy always matches exactly what the reader saw.
  function archivePreviousAnswer(question) {
    var item = document.createElement("details");
    item.className = "na-history-item";
    var summary = document.createElement("summary");
    summary.textContent = question;
    item.appendChild(summary);
    var body = document.createElement("div");
    body.className = "na-history-body";
    if (warningEl.style.display !== "none") body.appendChild(stripIds(warningEl.cloneNode(true)));
    body.appendChild(stripIds(answerEl.cloneNode(true)));
    if (sourcesEl.style.display !== "none") body.appendChild(stripIds(sourcesEl.cloneNode(true)));
    item.appendChild(body);
    historyEl.insertBefore(item, historyEl.firstChild);
  }

  // Reset the feedback row to its clickable, unanswered state -- called before showing a new
  // answer so a rating given on a prior answer never carries over onto the next one.
  function resetFeedback() {
    currentQueryId = null;
    currentFeedbackToken = null;
    feedbackEl.style.display = "none";
    feedbackThanksEl.style.display = "none";
    for (var i = 0; i < feedbackBtns.length; i++) {
      feedbackBtns[i].disabled = false;
      feedbackBtns[i].classList.remove("na-selected");
      feedbackBtns[i].style.display = "";
    }
  }

  function submitFeedback(rating, clickedBtn) {
    if (!currentQueryId || !currentFeedbackToken || !feedbackApiUrl) return;
    for (var i = 0; i < feedbackBtns.length; i++) feedbackBtns[i].disabled = true;
    clickedBtn.classList.add("na-selected");

    fetch(feedbackApiUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        query_id: currentQueryId,
        rating: rating,
        token: currentFeedbackToken,
      }),
    })
      .then(function (res) {
        if (!res.ok) throw new Error("feedback request failed");
        for (var i = 0; i < feedbackBtns.length; i++) {
          if (feedbackBtns[i] !== clickedBtn) feedbackBtns[i].style.display = "none";
        }
        feedbackThanksEl.style.display = "inline";
      })
      .catch(function () {
        // Quiet failure -- feedback is a nice-to-have, not worth surfacing an error banner
        // over. Just re-enable the buttons so the reader can try again.
        for (var i = 0; i < feedbackBtns.length; i++) feedbackBtns[i].disabled = false;
        clickedBtn.classList.remove("na-selected");
      });
  }

  for (var fbIndex = 0; fbIndex < feedbackBtns.length; fbIndex++) {
    feedbackBtns[fbIndex].addEventListener("click", function (e) {
      submitFeedback(e.currentTarget.getAttribute("data-rating"), e.currentTarget);
    });
  }

  // The loading text has two real states, driven by actual "STAGE:" lines the Worker streams
  // back as the query pipeline genuinely progresses (see lineStreamResponse in
  // worker/src/index.ts) -- not a client-side guess at timing. There's no real signal between
  // "reading the documents" and "writing the answer" since that's one continuous Gemini call
  // on the backend, so those honestly collapse into a single stage rather than faking a third.
  var LOADING_STAGE_TEXT = {
    reading: "Reading the reports and writing your answer",
  };

  var elapsedTimer = null;

  function setLoading(isLoading) {
    loadingEl.style.display = isLoading ? "block" : "none";
    submitBtn.disabled = isLoading;
    input.disabled = isLoading;

    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    if (!elapsedEl) return;

    if (isLoading) {
      loadingTextEl.textContent = "Searching NACFE’s research library";
      elapsedEl.textContent = "";
      // A ticking counter is the part that actually proves the page is alive. At a median of
      // 21s and a p90 of 46s, a reader watching a static indicator concludes it has hung --
      // so show the time passing rather than only animating. Starts at 5s to avoid flashing
      // a counter at anyone whose question hits the cache and returns almost immediately.
      var started = Date.now();
      elapsedTimer = setInterval(function () {
        var secs = Math.round((Date.now() - started) / 1000);
        elapsedEl.textContent = secs >= 5 ? " " + secs + "s elapsed." : "";
      }, 1000);
    } else {
      elapsedEl.textContent = "";
    }
  }

  // Reads the streamed "STAGE:<name>\n" / "DATA:<json>\n" response line by line, invoking
  // onStage for each stage marker as it actually arrives and resolving with the parsed final
  // payload once the terminal DATA line shows up.
  function readLineStream(body, onStage) {
    var reader = body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";

    function pump() {
      return reader.read().then(function (result) {
        if (result.done) {
          throw new Error("response ended unexpectedly");
        }
        buffer += decoder.decode(result.value, { stream: true });
        var lines = buffer.split("\n");
        buffer = lines.pop();
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (line.indexOf("DATA:") === 0) {
            return JSON.parse(line.slice(5));
          }
          if (line.indexOf("STAGE:") === 0) {
            onStage(line.slice(6));
          }
        }
        return pump();
      });
    }

    return pump();
  }

  // Rendered explicitly rather than via the auto-scanning class hook, because this page stays
  // alive across submissions and a Turnstile token is redeemed exactly once. Keeping the
  // widget id lets us reset it after every request so a second question gets a fresh token.
  if (TURNSTILE_SITEKEY) {
    var tsScript = document.createElement("script");
    tsScript.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    tsScript.async = true;
    tsScript.defer = true;
    tsScript.onload = function () {
      if (!window.turnstile) return;
      turnstileWidgetId = window.turnstile.render("#na-turnstile", {
        sitekey: TURNSTILE_SITEKEY,
        action: "query",
      });
    };
    document.head.appendChild(tsScript);
  }

  function turnstileToken() {
    if (!TURNSTILE_SITEKEY || !window.turnstile || turnstileWidgetId === null) return "";
    try {
      return window.turnstile.getResponse(turnstileWidgetId) || "";
    } catch (err) {
      return "";
    }
  }

  function resetTurnstile() {
    if (!TURNSTILE_SITEKEY || !window.turnstile || turnstileWidgetId === null) return;
    try {
      window.turnstile.reset(turnstileWidgetId);
    } catch (err) {
      // a reset failure shouldn't take the widget down; the next submit just re-checks
    }
  }

  // The sponsor block exists only when a sponsor is configured. Built from DOM nodes with
  // textContent rather than markup, so a name or tagline containing markup renders as text.
  if (SPONSOR.name) {
    var sponsorLogoSrc = safeHttpUrl(SPONSOR.logoUrl);
    if (sponsorLogoSrc) {
      var sponsorLogo = document.createElement("img");
      sponsorLogo.setAttribute("src", sponsorLogoSrc);
      sponsorLogo.setAttribute("alt", SPONSOR.name + " logo");
      // A broken/unreachable logo shouldn't leave a broken-image icon sitting in the row --
      // just drop it and keep the text, same as if no logo had been configured at all.
      sponsorLogo.addEventListener("error", function () {
        sponsorLogo.remove();
      });
      sponsorEl.appendChild(sponsorLogo);
    }
    var sponsorText = document.createElement("span");
    var sponsorName = document.createElement("b");
    sponsorName.textContent = SPONSOR.name;
    sponsorText.appendChild(sponsorName);
    if (SPONSOR.tagline) {
      sponsorText.appendChild(document.createTextNode(" \u2014 " + SPONSOR.tagline));
    }
    sponsorEl.appendChild(document.createTextNode("Sponsored by "));
    sponsorEl.appendChild(sponsorText);

    var sponsorHref = safeHttpUrl(SPONSOR.url);
    if (sponsorHref) {
      var sponsorLink = document.createElement("a");
      sponsorLink.setAttribute("href", sponsorHref);
      sponsorLink.setAttribute("target", "_blank");
      // noopener is what stops the sponsor's page reaching back through window.opener;
      // sponsored links also carry rel="sponsored" per Google's link-attribution guidance.
      sponsorLink.setAttribute("rel", "noopener noreferrer sponsored");
      sponsorLink.textContent = "Learn more";
      sponsorLink.addEventListener("click", function () {
        sendEvent("sponsor_click");
      });
      sponsorEl.appendChild(document.createTextNode(" "));
      sponsorEl.appendChild(sponsorLink);
    }
    sponsorEl.style.display = "flex";
  }

  if (infoBtn && infoEl) {
    infoBtn.addEventListener("click", function () {
      var open = infoEl.style.display !== "none";
      infoEl.style.display = open ? "none" : "block";
      infoBtn.setAttribute("aria-expanded", String(!open));
    });
  }

  // One impression per widget load. This is the number a sponsor actually buys, and it is far
  // larger than the question count -- most readers never type anything.
  sendEvent("impression");

  // Ask once on load whether the tool is budget-limited, so a reader finds out before
  // composing a question rather than after submitting one. Deliberately non-blocking and
  // silent on failure: the form stays fully usable either way, because cached questions are
  // still answered when the budget is spent.
  if (statusApiUrl) {
    fetch(statusApiUrl)
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (st) { if (st && st.degraded) showBudgetNotice(null, st.resets_at); })
      .catch(function (err) {
        // Non-fatal: the form works regardless. Logged rather than swallowed, because an
        // empty catch here hid a real bug once -- a redeclaration that reset the notice
        // element to null, so the banner silently never rendered.
        if (window.console && console.warn) console.warn("nacfe-assist: status check failed", err);
      });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var question = input.value.trim();
    if (!question) return;

    if (resultEl.classList.contains("na-visible") && shownQuestion) {
      archivePreviousAnswer(shownQuestion);
    }

    errorEl.style.display = "none";
    resultEl.classList.remove("na-visible");
    resetFeedback();
    setLoading(true);

    // Bound the request. Without this, a stalled backend leaves the widget spinning with its
    // input disabled and no way back other than reloading the host page.
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    var timedOut = false;
    var timer = setTimeout(function () {
      timedOut = true;
      if (controller) controller.abort();
    }, REQUEST_TIMEOUT_MS);

    fetch(apiUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        question: question,
        "cf-turnstile-response": turnstileToken(),
        // Which NACFE page the question came from. Sent without its query string; the Worker
        // re-normalizes anyway, since anything from the browser is untrusted.
        page_url: location.origin + location.pathname,
        history: conversationHistory,
      }),
      signal: controller ? controller.signal : undefined,
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().catch(function () { return {}; }).then(function (body) {
            // The monthly-budget degrade path answers 503 with {answer, degraded:true}. That
            // is a real notice written for the reader, not a failure -- surfacing it as
            // "Something went wrong: request failed (503)" threw the message away.
            // Carry resets_at through: rebuilding the object by hand dropped it, so the
            // refused-query banner fell back to "when the budget resets" while the on-load
            // banner named the date.
            if (body && body.answer) {
              return { answer: body.answer, degraded: true, resets_at: body.resets_at };
            }
            throw new Error(body.error || ("request failed (" + res.status + ")"));
          });
        }
        return readLineStream(res.body, function (stage) {
          if (LOADING_STAGE_TEXT[stage]) loadingTextEl.textContent = LOADING_STAGE_TEXT[stage];
        });
      })
      .then(function (data) {
        if (data.error) throw new Error(data.error);
        clearTimeout(timer);
        setLoading(false);
        resetTurnstile(); // the token just spent is dead; mint a fresh one for the next question

        if (data.degraded) {
          // Not an answer: no sources, nothing to rate. Rendered in the same banner the
          // on-load check uses, above the form, so the reader sees one consistent
          // explanation rather than an error where an answer should be.
          // The widget's own wording, not the server's prose: the API's `answer` field is
          // written for direct API consumers and duplicates this banner's bold prefix, and
          // it carries no reset date. One consistent sentence, wherever the state is learned.
          showBudgetNotice(null, data.resets_at);
          renderAnswer("");
          renderSources(null);
          resultEl.classList.remove("na-visible");
          return;
        }

        renderAnswer(data.answer || "");
        renderSources(data.selected_sources);
        if (data.recency_warning) {
          warningEl.textContent = data.recency_warning;
          warningEl.style.display = "block";
        } else {
          warningEl.style.display = "none";
        }
        if (data.query_id && data.feedback_token) {
          currentQueryId = data.query_id;
          currentFeedbackToken = data.feedback_token;
          feedbackEl.style.display = "flex";
        }
        resultEl.classList.add("na-visible");
        shownQuestion = question;
        conversationHistory.push({ question: question, answer: data.answer || "" });
        if (conversationHistory.length > CONVERSATION_TURNS_SENT) {
          conversationHistory.splice(0, conversationHistory.length - CONVERSATION_TURNS_SENT);
        }
      })
      .catch(function (err) {
        clearTimeout(timer);
        setLoading(false);
        resetTurnstile();
        errorEl.textContent = timedOut
          ? "That took too long to answer. Please try again."
          : "Something went wrong: " + err.message;
        errorEl.style.display = "block";
      });
  });
})();
