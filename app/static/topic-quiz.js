(function () {
  "use strict";

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function init() {
    var boot = document.getElementById("tdd-quiz-bootstrap");
    var root = document.getElementById("tdd-quiz-root");
    if (!boot || !root) return;

    var cfg;
    try {
      cfg = JSON.parse(boot.textContent || "{}");
    } catch (e) {
      return;
    }

    var lectureId = cfg.lectureId;
    var topicSlug = cfg.topicSlug;
    var quizzes = cfg.quizzes || {};
    var keys = Object.keys(quizzes);
    if (keys.length === 0) {
      return;
    }

    root.hidden = false;

    var state = {
      difficulty: keys[0],
      qIndex: 0,
      score: 0,
      missedTags: [],
      answeredThis: false,
    };

    var nav = document.createElement("div");
    nav.className = "tdd-iz-nav";
    if (keys.length > 1) {
      keys.forEach(function (k) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "button button-secondary button-small tdd-iz-tab";
        b.textContent = k.charAt(0).toUpperCase() + k.slice(1);
        b.dataset.diff = k;
        if (k === state.difficulty) b.classList.add("tdd-iz-tab--active");
        b.addEventListener("click", function () {
          state.difficulty = k;
          state.qIndex = 0;
          state.answeredThis = false;
          nav.querySelectorAll(".tdd-iz-tab").forEach(function (x) {
            x.classList.toggle("tdd-iz-tab--active", x.dataset.diff === k);
          });
          renderQuestion();
        });
        nav.appendChild(b);
      });
    }

    var stage = document.createElement("div");
    stage.className = "tdd-iz-stage";

    var feedbackEl = document.createElement("div");
    feedbackEl.className = "tdd-iz-feedback";
    feedbackEl.setAttribute("aria-live", "polite");

    var summaryEl = document.createElement("div");
    summaryEl.className = "tdd-iz-summary";
    summaryEl.hidden = true;

    root.appendChild(nav);
    root.appendChild(stage);
    root.appendChild(feedbackEl);
    root.appendChild(summaryEl);

    function currentQuiz() {
      return quizzes[state.difficulty];
    }

    function currentQuestion() {
      var qz = currentQuiz();
      if (!qz || !qz.questions) return null;
      return qz.questions[state.qIndex] || null;
    }

    function renderQuestion() {
      summaryEl.hidden = true;
      feedbackEl.innerHTML = "";
      stage.innerHTML = "";
      state.answeredThis = false;

      var q = currentQuestion();
      var qz = currentQuiz();
      if (!q || !qz) {
        stage.innerHTML = "<p class=\"muted\">No questions for this level.</p>";
        return;
      }

      var total = qz.questions.length;
      var h = document.createElement("h3");
      h.className = "tdd-iz-qtitle";
      h.textContent = "Question " + (state.qIndex + 1) + " / " + total;

      var p = document.createElement("p");
      p.className = "tdd-iz-stem";
      p.textContent = q.stem;

      var form = document.createElement("div");
      form.className = "tdd-iz-options";
      q.options.forEach(function (opt, idx) {
        var id = "tdd-opt-" + state.difficulty + "-" + state.qIndex + "-" + idx;
        var lab = document.createElement("label");
        lab.className = "tdd-iz-opt";
        var inp = document.createElement("input");
        inp.type = "radio";
        inp.name = "tdd-ans";
        inp.value = String(idx);
        inp.id = id;
        lab.appendChild(inp);
        lab.appendChild(document.createTextNode(" " + opt));
        form.appendChild(lab);
      });

      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "button tdd-iz-submit";
      btn.textContent = "Check answer";

      btn.addEventListener("click", function () {
        var sel = form.querySelector('input[name="tdd-ans"]:checked');
        if (!sel) return;
        var idx = parseInt(sel.value, 10);
        if (state.answeredThis) return;
        state.answeredThis = true;
        btn.disabled = true;
        form.querySelectorAll("input").forEach(function (inp) {
          inp.disabled = true;
        });

        fetch(
          "/lectures/" +
            lectureId +
            "/topics/" +
            encodeURIComponent(topicSlug) +
            "/quiz/check",
          {
            method: "POST",
            headers: { "Content-Type": "application/json", Accept: "application/json" },
            body: JSON.stringify({
              difficulty: state.difficulty,
              question_id: q.id,
              selected_index: idx,
            }),
          }
        )
          .then(function (r) {
            return r.json();
          })
          .then(function (data) {
            if (!data.ok) {
              feedbackEl.innerHTML =
                '<p class="tdd-iz-err">' + esc(data.error || "Error") + "</p>";
              return;
            }
            if (data.correct) state.score += 1;
            else {
              var tag = q.concept_tag || (q.stem ? q.stem.slice(0, 72) : "Topic area");
              state.missedTags.push(tag);
            }
            feedbackEl.innerHTML =
              '<pre class="tdd-iz-feedback-pre">' + esc(data.feedback || "") + "</pre>";

            var next = document.createElement("button");
            next.type = "button";
            next.className = "button button-secondary tdd-iz-next";
            next.textContent =
              state.qIndex + 1 >= currentQuiz().questions.length ? "See summary" : "Next question";
            next.addEventListener("click", function () {
              if (state.qIndex + 1 >= currentQuiz().questions.length) {
                showSummary();
              } else {
                state.qIndex += 1;
                renderQuestion();
              }
            });
            feedbackEl.appendChild(next);
          })
          .catch(function () {
            feedbackEl.innerHTML = '<p class="tdd-iz-err">Request failed.</p>';
          });
      });

      stage.appendChild(h);
      stage.appendChild(p);
      stage.appendChild(form);
      stage.appendChild(btn);
    }

    function showSummary() {
      stage.innerHTML = "";
      feedbackEl.innerHTML = "";
      summaryEl.hidden = false;
      var total = currentQuiz().questions.length;
      var h = document.createElement("h3");
      h.textContent = "Quiz complete";
      var sc = document.createElement("p");
      sc.className = "tdd-iz-score";
      sc.textContent = "Score: " + state.score + " / " + total;

      summaryEl.innerHTML = "";
      summaryEl.appendChild(h);
      summaryEl.appendChild(sc);

      if (state.missedTags.length > 0) {
        var ul = document.createElement("ul");
        ul.className = "tdd-iz-missed";
        state.missedTags.forEach(function (t) {
          var li = document.createElement("li");
          li.textContent = t;
          ul.appendChild(li);
        });
        var sub = document.createElement("p");
        sub.className = "muted small-print";
        sub.textContent = "This session — topics to revisit:";
        summaryEl.appendChild(sub);
        summaryEl.appendChild(ul);
      }

      var sm = document.getElementById("tdd-quiz-server-mistakes");
      if (sm && sm.querySelector && sm.querySelector("li")) {
        var cp = sm.cloneNode(true);
        cp.removeAttribute("id");
        cp.className = "tdd-iz-server-mistakes";
        summaryEl.appendChild(cp);
      }

      var again = document.createElement("p");
      again.className = "muted small-print";
      again.textContent =
        "Generate another quiz (same or different difficulty) to practice again — past mistakes inform new questions.";
      summaryEl.appendChild(again);
    }

    renderQuestion();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
