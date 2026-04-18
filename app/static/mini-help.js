(function () {
  "use strict";

  function getContext() {
    var el = document.getElementById("mini-help-context-data");
    if (!el || !el.textContent) return {};
    try {
      return JSON.parse(el.textContent);
    } catch (e) {
      return {};
    }
  }

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function init() {
    var fab = document.getElementById("mini-help-fab");
    var panel = document.getElementById("mini-help-panel");
    var closeBtn = document.querySelector(".mini-help-close");
    var send = document.getElementById("mini-help-send");
    var input = document.getElementById("mini-help-input");
    var out = document.getElementById("mini-help-out");
    if (!fab || !panel || !send || !input || !out) return;

    function setOpen(open) {
      panel.hidden = !open;
      fab.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) {
        input.focus();
      }
    }

    fab.addEventListener("click", function () {
      setOpen(panel.hidden);
    });
    if (closeBtn) {
      closeBtn.addEventListener("click", function () {
        setOpen(false);
      });
    }
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setOpen(false);
    });

    send.addEventListener("click", function () {
      var msg = (input.value || "").trim();
      if (!msg) return;
      out.innerHTML = '<span class="mini-help-loading">…</span>';
      send.disabled = true;
      fetch("/mini-help/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ message: msg, context: getContext() }),
      })
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          if (data.ok && data.reply) {
            out.innerHTML = '<pre class="mini-help-reply">' + esc(data.reply) + "</pre>";
          } else {
            out.innerHTML =
              '<p class="mini-help-error">' + esc(data.error || "Something went wrong.") + "</p>";
          }
        })
        .catch(function () {
          out.innerHTML = '<p class="mini-help-error">Could not reach the server.</p>';
        })
        .finally(function () {
          send.disabled = false;
        });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
