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
      "#nacfe-assist-root .na-loading{font-size:14px;color:#6b6b6b;padding:12px 0}",
      "#nacfe-assist-root .na-dots{display:inline-flex;gap:3px;margin-left:6px;",
      "vertical-align:middle}",
      "#nacfe-assist-root .na-dots span{width:5px;height:5px;border-radius:50%;",
      "background:#ab1428;animation:na-pulse 1.2s ease-in-out infinite}",
      "#nacfe-assist-root .na-dots span:nth-child(2){animation-delay:.2s}",
      "#nacfe-assist-root .na-dots span:nth-child(3){animation-delay:.4s}",
      "@keyframes na-pulse{0%,80%,100%{opacity:.25;transform:scale(.8)}",
      "40%{opacity:1;transform:scale(1)}}",
      "#nacfe-assist-root .na-error{font-size:14px;color:#ab1428;padding:12px 0}",
      "#nacfe-assist-root .na-footer{font-size:11px;color:#9a9a9a;margin-top:18px}",
      "#nacfe-assist-root .na-footer a{color:#001961}",
      "#nacfe-assist-root .na-sponsor{display:flex;align-items:center;gap:10px;",
      "font-size:14px;color:#6b6b6b;margin:0 0 6px}",
      "#nacfe-assist-root .na-sponsor img{width:28px;height:28px;border-radius:5px;",
      "flex-shrink:0;object-fit:contain}",
      "#nacfe-assist-root .na-sponsor b{color:#333}",
      "#nacfe-assist-root .na-sponsor a{color:#001961;text-decoration:none}",
      "#nacfe-assist-root .na-sponsor a:hover{text-decoration:underline}",
      // Desktop-width viewports get a wider column and larger type -- the base rules above
      // stay mobile-sized so phones/narrow embeds aren't affected.
      "@media (min-width:640px){",
      "#nacfe-assist-root{max-width:900px}",
      "#nacfe-assist-root .na-label{font-size:15px}",
      "#nacfe-assist-root input[type=text]{font-size:17px;padding:13px 16px}",
      "#nacfe-assist-root button{font-size:17px;padding:13px 24px}",
      "#nacfe-assist-root .na-hint{font-size:14px}",
      "#nacfe-assist-root .na-answer{font-size:18px;line-height:1.7}",
      "#nacfe-assist-root .na-warning{font-size:15px}",
      "#nacfe-assist-root .na-sources-label{font-size:14px}",
      "#nacfe-assist-root .na-source{font-size:15px;padding:10px 12px}",
      "#nacfe-assist-root .na-loading{font-size:16px}",
      "#nacfe-assist-root .na-dots span{width:6px;height:6px}",
      "#nacfe-assist-root .na-error{font-size:16px}",
      "#nacfe-assist-root .na-feedback-label{font-size:15px}",
      "#nacfe-assist-root .na-feedback-btn{font-size:15px;padding:8px 16px}",
      "#nacfe-assist-root .na-feedback-thanks{font-size:15px}",
      "#nacfe-assist-root .na-footer{font-size:13px}",
      "#nacfe-assist-root .na-sponsor{font-size:16px}",
      "#nacfe-assist-root .na-sponsor img{width:32px;height:32px}",
      "}",
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
    '<div class="na-loading" id="na-loading" style="display:none">',
    '<span id="na-loading-text">Searching NACFE’s research library</span>',
    '<span class="na-dots"><span></span><span></span><span></span></span>',
    '</div>',
    '<div class="na-error" id="na-error" style="display:none"></div>',
    '<div class="na-result" id="na-result">',
    '<div class="na-warning" id="na-warning" style="display:none"></div>',
    '<div class="na-answer" id="na-answer"></div>',
    '<div class="na-sources" id="na-sources" style="display:none">',
    '<p class="na-sources-label">Sources</p>',
    '<div id="na-sources-list"></div>',
    '</div>',
    '<div class="na-feedback" id="na-feedback" style="display:none">',
    '<span class="na-feedback-label">Was this answer accurate?</span>',
    '<button type="button" class="na-feedback-btn" data-rating="correct">Correct</button>',
    '<button type="button" class="na-feedback-btn" data-rating="partial">Partially correct</button>',
    '<button type="button" class="na-feedback-btn" data-rating="wrong">Wrong</button>',
    '<span class="na-feedback-thanks" id="na-feedback-thanks" style="display:none">Thanks for the feedback!</span>',
    '</div>',
    '</div>',
    '<p class="na-sponsor" id="na-sponsor">',
    '<img src="data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 40 40\'%3E%3Crect width=\'40\' height=\'40\' rx=\'8\' fill=\'%232563eb\'/%3E%3Ctext x=\'20\' y=\'26\' font-family=\'Arial\' font-size=\'16\' font-weight=\'700\' fill=\'white\' text-anchor=\'middle\'%3EES%3C/text%3E%3C/svg%3E" alt="Example Sponsor Name logo" />',
    '<span><b>Example Sponsor Name</b> — Sponsor tag line. <a href="#">Learn more</a></span>',
    '</p>',
    '<p class="na-footer">Powered by NACFE’s research library. Answers cite specific reports and demonstrations — verify against the original source for critical decisions.</p>',
    '</div>',
  ].join("");

  var form = mount.querySelector("#na-form");
  var input = mount.querySelector("#na-input");
  var submitBtn = mount.querySelector("#na-submit");
  var loadingEl = mount.querySelector("#na-loading");
  var loadingTextEl = mount.querySelector("#na-loading-text");
  var errorEl = mount.querySelector("#na-error");
  var resultEl = mount.querySelector("#na-result");
  var warningEl = mount.querySelector("#na-warning");
  var answerEl = mount.querySelector("#na-answer");
  var sourcesEl = mount.querySelector("#na-sources");
  var sourcesListEl = mount.querySelector("#na-sources-list");
  var feedbackEl = mount.querySelector("#na-feedback");
  var feedbackBtns = mount.querySelectorAll(".na-feedback-btn");
  var feedbackThanksEl = mount.querySelector("#na-feedback-thanks");
  var feedbackApiUrl = apiUrl.replace(/\/query$/, "/feedback");
  var currentQueryId = null;

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
      var label = s.url
        ? '<a href="' + escapeHtml(s.url) + '" target="_blank" rel="noopener">' + escapeHtml(s.id) + "</a>"
        : escapeHtml(s.id);
      return '<div class="na-source"><b>' + label + "</b> — " +
        escapeHtml(s.why || "") + "</div>";
    }).join("");
    sourcesEl.style.display = "block";
  }

  // Reset the feedback row to its clickable, unanswered state -- called before showing a new
  // answer so a rating given on a prior answer never carries over onto the next one.
  function resetFeedback() {
    currentQueryId = null;
    feedbackEl.style.display = "none";
    feedbackThanksEl.style.display = "none";
    for (var i = 0; i < feedbackBtns.length; i++) {
      feedbackBtns[i].disabled = false;
      feedbackBtns[i].classList.remove("na-selected");
      feedbackBtns[i].style.display = "";
    }
  }

  function submitFeedback(rating, clickedBtn) {
    if (!currentQueryId) return;
    for (var i = 0; i < feedbackBtns.length; i++) feedbackBtns[i].disabled = true;
    clickedBtn.classList.add("na-selected");

    fetch(feedbackApiUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query_id: currentQueryId, rating: rating }),
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

  function setLoading(isLoading) {
    loadingEl.style.display = isLoading ? "block" : "none";
    submitBtn.disabled = isLoading;
    input.disabled = isLoading;
    if (isLoading) {
      loadingTextEl.textContent = "Searching NACFE’s research library";
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

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var question = input.value.trim();
    if (!question) return;

    errorEl.style.display = "none";
    resultEl.classList.remove("na-visible");
    resetFeedback();
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
        return readLineStream(res.body, function (stage) {
          if (LOADING_STAGE_TEXT[stage]) loadingTextEl.textContent = LOADING_STAGE_TEXT[stage];
        });
      })
      .then(function (data) {
        if (data.error) throw new Error(data.error);
        setLoading(false);
        renderAnswer(data.answer || "");
        renderSources(data.selected_sources);
        if (data.recency_warning) {
          warningEl.textContent = data.recency_warning;
          warningEl.style.display = "block";
        } else {
          warningEl.style.display = "none";
        }
        if (data.query_id) {
          currentQueryId = data.query_id;
          feedbackEl.style.display = "flex";
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
