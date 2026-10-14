<?php
/**
 * Plugin Name: NACFE Assist Admin Console Link
 * Description: Adds a WP-admin menu item that signs a short-lived handoff token and sends a
 *              logged-in administrator straight into the NACFE Assist admin console (usage
 *              dashboard, sponsor management, source catalog). Does not store, read, or write
 *              any admin-console data itself -- it only mints a token and redirects.
 * Version:     1.2.0
 */

if (!defined('ABSPATH')) {
    exit; // No direct access.
}

/**
 * Where the admin console lives. Wrapped in a defined() guard so a staging environment can
 * override it from wp-config.php (define('NACFE_ADMIN_CONSOLE_BASE', '...') there) without
 * editing this file.
 */
if (!defined('NACFE_ADMIN_CONSOLE_BASE')) {
    define('NACFE_ADMIN_CONSOLE_BASE', 'https://nacfe-assist.nacfe.workers.dev');
}

/**
 * The shared secret. Must be the EXACT value passed to:
 *   openssl rand -hex 32 | wrangler secret put ADMIN_TOKEN_SECRET
 * Set it in wp-config.php (not here, and not in the options table/database) -- the same way
 * WordPress stores its own AUTH_KEY/SECURE_AUTH_KEY/etc security salts:
 *
 *   define('NACFE_ADMIN_TOKEN_SECRET', 'the exact value you gave wrangler secret put');
 *
 * Add that line to wp-config.php above the "That's all, stop editing!" comment.
 */
// (Intentionally not defined here -- see wp-config.php.)

add_action('admin_menu', function () {
    $hook = add_menu_page(
        'NACFE Assist Admin',
        'NACFE Assist',
        'manage_options',
        'nacfe-assist-admin',
        'nacfe_assist_admin_console_render', // only ever renders the "not configured" notice --
                                              // see the load-{$hook} redirect below for why.
        'dashicons-admin-site-alt3',
        80
    );

    // Redirecting from load-{$hook} rather than from the page's own render callback: by the
    // time add_menu_page()'s callback runs (invoked from wp-admin/admin.php via do_action()),
    // wp-admin/admin-header.php has already emitted the doctype, <head>, and opening <body> --
    // a wp_redirect()/header() call at that point hits "headers already sent" and silently
    // does nothing on a default install (it can appear to work if output_buffering happens to
    // be enabled in php.ini, which is not something to rely on). load-{$hook} fires before any
    // of that output, which is the standard WordPress idiom for "redirect away from this admin
    // page before it renders."
    add_action("load-{$hook}", 'nacfe_assist_admin_console_redirect');
});

/**
 * Opens the console in a new browser tab. WordPress admin menu links can't declare a target, so
 * this marks the "NACFE Assist" menu item target="_blank" once the page has loaded. The new tab
 * still follows the normal flow: it requests the menu page, load-{$hook} (below) mints a fresh
 * single-use handoff token and redirects to the console, and the WordPress tab stays where it
 * was. rel="noopener" so the console tab can't reach back into wp-admin via window.opener.
 */
add_action('admin_footer', function () {
    if (!current_user_can('manage_options')) {
        return;
    }
    ?>
<script>
(function () {
  var link = document.querySelector('#adminmenu a[href*="page=nacfe-assist-admin"]');
  if (link) {
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener');
  }
})();
</script>
    <?php
});

/**
 * Builds and signs a handoff token: base64url(JSON{email, iat, jti}) + "." +
 * hex(HMAC-SHA256(secret, that base64url string)). Must byte-for-byte match verifyHandoffToken
 * in worker/src/index.ts -- the signature covers the literal base64url string, never a
 * re-encoded form of the JSON. hash_hmac()'s default output is lowercase hex, matching the
 * Worker's hmacHex() exactly -- timingSafeEqual there is a plain string compare, so an
 * uppercase change on either side would break auth with no obvious cause; don't pass
 * $raw_output = true or otherwise change this.
 *
 * `jti` makes the token single-use: the Worker records it in KV on first use and rejects any
 * repeat, even one still well within the token's own short TTL. random_bytes() is PHP's CSPRNG
 * (not mt_rand/rand) -- required here since this value is a security-relevant nonce, not just
 * an arbitrary identifier. A fresh token (and therefore a fresh jti) is minted on every
 * invocation, so this never needs to be reused or cached across requests.
 *
 * `typ` must be exactly "handoff" -- the Worker checks this field explicitly so a handoff
 * token can never be mistaken for (or substituted as) the longer-lived session cookie it gets
 * exchanged for at /admin/login. Omitting or misspelling this makes the Worker reject the
 * token outright rather than silently accepting a malformed one.
 */
function nacfe_assist_mint_handoff_token($email, $secret)
{
    $payload = wp_json_encode([
        'typ'   => 'handoff',
        'email' => $email,
        'iat'   => time(),
        'jti'   => bin2hex(random_bytes(16)),
    ]);
    $b64 = rtrim(strtr(base64_encode($payload), '+/', '-_'), '=');
    $sig = hash_hmac('sha256', $b64, $secret);
    return $b64 . '.' . $sig;
}

/**
 * Runs on load-{$hook}, before the admin page template has emitted anything -- see the
 * add_menu_page() call above for why this can't live in the page's own render callback.
 */
function nacfe_assist_admin_console_redirect()
{
    // Defense in depth: WordPress already enforces the manage_options capability declared in
    // add_menu_page() before this page (and this hook) is reachable at all, so in practice this
    // never fails -- but checking costs nothing and keeps the function safe if that ever changes.
    if (!current_user_can('manage_options')) {
        return;
    }
    // Secret not configured: let the page render normally so the notice below is visible,
    // rather than redirecting to a URL the Worker will just reject anyway.
    if (!defined('NACFE_ADMIN_TOKEN_SECRET') || NACFE_ADMIN_TOKEN_SECRET === '') {
        return;
    }

    $user = wp_get_current_user();
    $token = nacfe_assist_mint_handoff_token($user->user_email, NACFE_ADMIN_TOKEN_SECRET);
    $url = NACFE_ADMIN_CONSOLE_BASE . '/admin/login?token=' . rawurlencode($token);

    // wp_redirect(), not wp_safe_redirect(): the destination is deliberately an external host
    // (the Worker), which wp_safe_redirect() would otherwise block via its allowed-hosts list.
    wp_redirect($url);
    exit;
}

/**
 * Reached only when NACFE_ADMIN_TOKEN_SECRET is unset -- nacfe_assist_admin_console_redirect()
 * above redirects away before this ever renders whenever the secret IS configured.
 */
function nacfe_assist_admin_console_render()
{
    if (!current_user_can('manage_options')) {
        wp_die('You do not have permission to access this page.');
    }
    echo '<div class="wrap"><h1>NACFE Assist Admin</h1>';
    echo '<div class="notice notice-error"><p><strong>NACFE_ADMIN_TOKEN_SECRET is not set.</strong> ';
    echo 'Add <code>define(\'NACFE_ADMIN_TOKEN_SECRET\', \'...\');</code> to wp-config.php, ';
    echo 'using the same value you gave <code>wrangler secret put ADMIN_TOKEN_SECRET</code>.</p></div></div>';
}

// A [nacfe_admin_console_link] shortcode existed in an earlier version of this plugin and was
// removed deliberately: a shortcode renders into ordinary post/page content, which -- unlike
// wp-admin, which WordPress never page-caches -- can be served from a full-page-cache plugin or
// an edge/CDN cache (e.g. Cloudflare in front of nacfe.org) independent of any nocache_headers()
// call this plugin makes (that call sets HTTP headers, which do nothing once a cache has
// already stored the rendered HTML, and can itself be a no-op if called after output has
// started). A live, working handoff token embedded in cacheable HTML would reach every visitor
// who loads that cached page while the token's 30-minute TTL hasn't elapsed, not just the
// logged-in admin who generated it. The menu page above doesn't have this problem (wp-admin is
// never page-cached) and is the only entry point this plugin needs. If a front-end entry point
// is ever genuinely required, mint the token server-side from an admin-post.php action gated by
// a nonce, triggered by a click rather than rendered into page content -- never echo a token
// into HTML that any caching layer could persist.
