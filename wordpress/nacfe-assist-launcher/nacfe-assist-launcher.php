<?php
/**
 * Plugin Name: NACFE Assist Launcher
 * Description: Adds the floating "Ask NACFE" logo button and chat panel to front-end pages
 *              site-wide, with an on/off switch and include/exclude page lists
 *              (Settings > NACFE Assist Launcher). Contains no secrets and no per-user data, so
 *              page caching is safe.
 * Version:     1.1.0
 */

if (!defined('ABSPATH')) {
    exit; // No direct access.
}

/**
 * Where the Worker lives. defined() guard so staging can override it from wp-config.php.
 */
if (!defined('NACFE_ASSIST_BASE')) {
    define('NACFE_ASSIST_BASE', 'https://nacfe-assist.nacfe.workers.dev');
}

const NACFE_ASSIST_LAUNCHER_OPTION = 'nacfe_assist_launcher';

function nacfe_assist_launcher_defaults()
{
    return [
        'enabled'  => 1,
        // One path per line; * is a wildcard. Empty include list = every page.
        'include'  => '',
        'exclude'  => "/wp-login.php\n/cart*\n/checkout*\n/my-account*",
    ];
}

function nacfe_assist_launcher_settings()
{
    $saved = get_option(NACFE_ASSIST_LAUNCHER_OPTION, []);
    $s = array_merge(nacfe_assist_launcher_defaults(), is_array($saved) ? $saved : []);
    // Emergency switch from wp-config.php: define('NACFE_ASSIST_LAUNCHER_ENABLED', false);
    if (defined('NACFE_ASSIST_LAUNCHER_ENABLED')) {
        $s['enabled'] = NACFE_ASSIST_LAUNCHER_ENABLED ? 1 : 0;
    }
    return $s;
}

/** Lowercased request path, query string dropped, trailing slash trimmed ("/" stays "/"). */
function nacfe_assist_launcher_current_path()
{
    $uri  = isset($_SERVER['REQUEST_URI']) ? wp_unslash($_SERVER['REQUEST_URI']) : '/';
    $path = (string) wp_parse_url($uri, PHP_URL_PATH);
    $path = strtolower($path === '' ? '/' : $path);
    return $path === '/' ? '/' : rtrim($path, '/');
}

/** True if $path matches any line of $patterns (one per line, * wildcard, case-insensitive). */
function nacfe_assist_launcher_matches($path, $patterns)
{
    foreach (preg_split('/\R/', (string) $patterns) as $line) {
        $line = strtolower(trim($line));
        if ($line === '') {
            continue;
        }
        $line = $line === '/' ? '/' : rtrim($line, '/');
        $regex = '#^' . str_replace('\*', '.*', preg_quote($line, '#')) . '$#';
        if (preg_match($regex, $path)) {
            return true;
        }
    }
    return false;
}

function nacfe_assist_launcher_should_show()
{
    if (is_admin() || is_feed() || is_embed() || is_customize_preview() || is_404()
        || (function_exists('wp_doing_ajax') && wp_doing_ajax())
        || (defined('REST_REQUEST') && REST_REQUEST)
        || is_robots() || is_trackback()) {
        return false;
    }
    $s = nacfe_assist_launcher_settings();
    if (empty($s['enabled'])) {
        return false;
    }
    $path = nacfe_assist_launcher_current_path();
    if (nacfe_assist_launcher_matches($path, $s['exclude'])) {
        return false;
    }
    if (trim((string) $s['include']) !== '' && !nacfe_assist_launcher_matches($path, $s['include'])) {
        return false;
    }
    return true;
}

add_action('wp_footer', function () {
    if (!nacfe_assist_launcher_should_show()) {
        return;
    }
    $base = esc_url_raw(NACFE_ASSIST_BASE);
    $logo = plugins_url('assets/ask-nacfe-logo.png', __FILE__);
    ?>
<!-- NACFE Assist Launcher -->
<style id="na-launcher-style">
  #na-teaser {
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    box-sizing: border-box; width: 234px; padding: 12px;
    background: #fff; border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,.18); cursor: pointer;
    transition: box-shadow .15s ease, transform .15s ease;
    font-family: Roboto, Helvetica, Arial, sans-serif;
  }
  #na-teaser:hover { box-shadow: 0 6px 20px rgba(0,0,0,.24); transform: translateY(-1px); }
  #na-teaser.na-glow { animation: na-teaser-glow 1.6s ease-in-out 2; }
  @keyframes na-teaser-glow {
    0%, 100% { box-shadow: 0 4px 14px rgba(0,0,0,.18), 0 0 0px rgba(171,20,40,0); }
    50%      { box-shadow: 0 4px 14px rgba(0,0,0,.18), 0 0 16px 4px rgba(171,20,40,.45); }
  }
  #na-teaser-logo { display: block; width: 210px; height: auto; }
  /* Sponsor name only (no logo or tagline while closed); two lines at most. */
  #na-teaser-sponsor {
    font-size: 11px; color: #6b6b6b; margin-top: 6px; line-height: 1.25;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  }
  #na-teaser.na-hidden { display: none; }
  @media (max-width: 640px) {
    #na-teaser {
      right: 16px; bottom: 16px; bottom: calc(16px + env(safe-area-inset-bottom, 0px));
      width: 170px; padding: 10px;
    }
    #na-teaser-logo { width: 150px; }
  }

  #na-launcher-panel {
    position: fixed; bottom: 24px; right: 24px; width: 640px; height: 840px;
    max-width: calc(100vw - 48px); max-height: calc(100vh - 48px);
    background: #fff; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,.22);
    display: flex; flex-direction: column; z-index: 9998;
    opacity: 0; transform: translateY(12px) scale(.98); pointer-events: none;
    transition: opacity .18s ease, transform .18s ease;
  }
  #na-launcher-panel.na-open { opacity: 1; transform: translateY(0) scale(1); pointer-events: auto; }
  #na-launcher-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 16px; border-bottom: 1px solid #e2e2e2; flex-shrink: 0;
    font-family: Roboto, Helvetica, Arial, sans-serif;
  }
  #na-launcher-logo { display: block; height: 56px; width: auto; }
  @media (max-width: 640px) { #na-launcher-logo { height: 44px; } }
  #na-launcher-close { background: none; border: none; cursor: pointer; color: #6b6b6b; font-size: 20px; line-height: 1; padding: 4px; }
  #na-launcher-close:hover { color: #292d2d; }
  #na-launcher-body { padding: 14px 16px 18px; overflow-y: auto; flex: 1; min-height: 0; }
</style>
<script id="na-launcher-script">
(function () {
  // Single-load guard: if an older pasted copy of the launcher is already on the page (a Custom
  // HTML block), leave it alone rather than create duplicate #na-* ids.
  if (document.getElementById("na-teaser") || window.__naLauncherLoaded) { return; }
  window.__naLauncherLoaded = true;

  var BASE = <?php echo wp_json_encode(untrailingslashit($base)); ?>;
  var LOGO = <?php echo wp_json_encode($logo); ?>;

  var wrap = document.createElement("div");
  wrap.innerHTML =
    '<div id="na-teaser" role="button" tabindex="0" aria-label="Ask NACFE’s Research">' +
      '<img id="na-teaser-logo" src="' + LOGO + '" alt="Ask NACFE" width="210" height="54">' +
      '<div id="na-teaser-sponsor" style="display:none"></div>' +
    '</div>' +
    '<div id="na-launcher-panel" role="dialog" aria-label="Ask NACFE’s Research">' +
      '<div id="na-launcher-header">' +
        '<img id="na-launcher-logo" src="' + LOGO + '" alt="Ask NACFE" width="219" height="56">' +
        '<button id="na-launcher-close" aria-label="Close">×</button></div>' +
      '<div id="na-launcher-body"><div id="nacfe-assist"></div></div>' +
    '</div>';
  var teaser = wrap.querySelector("#na-teaser");
  var panel = wrap.querySelector("#na-launcher-panel");
  var teaserSponsor = wrap.querySelector("#na-teaser-sponsor");
  var closeBtn = wrap.querySelector("#na-launcher-close");

  // Both are appended straight to <body>: themes that put a transform on a wrapper element break
  // position:fixed for anything inside it.
  document.body.appendChild(teaser);
  document.body.appendChild(panel);

  // text/plain keeps this a CORS "simple" request: no preflight, and sendBeacon (which always
  // sends in credentialed mode) is not blocked. The Worker parses the body as JSON regardless.
  function naSendEvent(type) {
    try {
      var url = BASE + "/event";
      var payload = JSON.stringify({ type: type, page_url: location.origin + location.pathname });
      if (navigator.sendBeacon) {
        navigator.sendBeacon(url, new Blob([payload], { type: "text/plain;charset=UTF-8" }));
      } else {
        fetch(url, { method: "POST", headers: { "content-type": "text/plain;charset=UTF-8" }, body: payload, keepalive: true }).catch(function () {});
      }
    } catch (e) {}
  }

  function setOpen(open) {
    panel.classList.toggle("na-open", open);
    teaser.classList.toggle("na-hidden", open);
    if (open) { naSendEvent("widget_expand"); }
  }
  teaser.addEventListener("click", function () { setOpen(true); });
  teaser.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen(true); }
  });
  closeBtn.addEventListener("click", function () { setOpen(false); });

  // Show "Sponsored by X" on the teaser once widget.js has rendered its own sponsor line.
  function mirrorSponsor() {
    var sponsorEl = document.getElementById("na-sponsor");
    if (sponsorEl && sponsorEl.style.display !== "none") {
      var nameEl = sponsorEl.querySelector("b");
      var name = nameEl ? nameEl.textContent.trim() : "";
      if (name) {
        teaserSponsor.textContent = "Sponsored by " + name;
        teaserSponsor.style.display = "block";
      }
    }
  }
  var attempts = 0;
  var poll = setInterval(function () {
    attempts++;
    if (document.getElementById("na-sponsor") || attempts > 100) {
      clearInterval(poll);
      mirrorSponsor();
    }
  }, 100);

  // The #nacfe-assist host exists now, so widget.js can mount.
  var s = document.createElement("script");
  s.src = BASE + "/widget.js";
  s.defer = true;
  document.body.appendChild(s);

  setTimeout(function () { teaser.classList.add("na-glow"); }, 900);
})();
</script>
    <?php
}, 99);

// ---- Settings page: Settings > NACFE Assist Launcher ----

add_action('admin_menu', function () {
    add_options_page(
        'NACFE Assist Launcher',
        'NACFE Assist Launcher',
        'manage_options',
        'nacfe-assist-launcher',
        'nacfe_assist_launcher_render_settings'
    );
});

function nacfe_assist_launcher_render_settings()
{
    if (!current_user_can('manage_options')) {
        wp_die('You do not have permission to access this page.');
    }

    $saved_notice = false;
    if (isset($_POST['nacfe_launcher_submit'])) {
        check_admin_referer('nacfe_assist_launcher_save');
        $clean = function ($key) {
            $v = isset($_POST[$key]) ? (string) wp_unslash($_POST[$key]) : '';
            $lines = array_filter(array_map('trim', preg_split('/\R/', $v)), 'strlen');
            return implode("\n", array_map('sanitize_text_field', $lines));
        };
        update_option(NACFE_ASSIST_LAUNCHER_OPTION, [
            'enabled' => empty($_POST['enabled']) ? 0 : 1,
            'include' => $clean('include'),
            'exclude' => $clean('exclude'),
        ]);
        $saved_notice = true;
    }

    $s = nacfe_assist_launcher_settings();
    $locked = defined('NACFE_ASSIST_LAUNCHER_ENABLED');
    ?>
    <div class="wrap">
      <h1>NACFE Assist Launcher</h1>
      <?php if ($saved_notice) : ?>
        <div class="notice notice-success is-dismissible"><p>Saved. If a caching plugin or CDN is in use, clear its cache for the change to appear on cached pages.</p></div>
      <?php endif; ?>
      <?php if ($locked) : ?>
        <div class="notice notice-warning"><p>The on/off switch is forced by <code>NACFE_ASSIST_LAUNCHER_ENABLED</code> in wp-config.php and cannot be changed here.</p></div>
      <?php endif; ?>
      <form method="post">
        <?php wp_nonce_field('nacfe_assist_launcher_save'); ?>
        <table class="form-table" role="presentation">
          <tr>
            <th scope="row">Show launcher</th>
            <td><label><input type="checkbox" name="enabled" value="1" <?php checked(!empty($s['enabled'])); ?> <?php disabled($locked); ?>> Show the "Ask NACFE's Research" button on the front-end of the site</label></td>
          </tr>
          <tr>
            <th scope="row"><label for="nacfe-include">Only on these pages</label></th>
            <td>
              <textarea id="nacfe-include" name="include" rows="5" class="large-text code"><?php echo esc_textarea($s['include']); ?></textarea>
              <p class="description">One path per line, <code>*</code> is a wildcard (for example <code>/research/*</code>). Leave empty to show on every page.</p>
            </td>
          </tr>
          <tr>
            <th scope="row"><label for="nacfe-exclude">Never on these pages</label></th>
            <td>
              <textarea id="nacfe-exclude" name="exclude" rows="5" class="large-text code"><?php echo esc_textarea($s['exclude']); ?></textarea>
              <p class="description">Same format. Exclusions win over the list above. Paths ignore case and trailing slashes; the home page is <code>/</code>.</p>
            </td>
          </tr>
        </table>
        <?php submit_button('Save changes', 'primary', 'nacfe_launcher_submit'); ?>
      </form>
      <p class="description">The launcher never appears in wp-admin, feeds, or the Customizer preview. If the same launcher was pasted into a page as a Custom HTML block, that copy keeps working and this plugin skips the page; remove the pasted block to let the plugin manage it.</p>
    </div>
    <?php
}
