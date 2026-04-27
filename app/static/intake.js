(function () {
  const drop = document.getElementById("intake-drop");
  const fileInput = document.getElementById("intake-file");
  const statusEl = document.getElementById("intake-status");
  const summary = document.getElementById("intake-summary");
  const sumCourse = document.getElementById("sum-course");
  const sumKind = document.getElementById("sum-kind");
  const sumConf = document.getElementById("sum-conf");
  const sumNote = document.getElementById("sum-note");
  const fToken = document.getElementById("f-token");
  const dToken = document.getElementById("d-token");
  const fCourse = document.getElementById("f-course");
  const fNew = document.getElementById("f-newcourse");
  const fKind = document.getElementById("f-kind");
  const confirmForm = document.getElementById("intake-confirm-form");
  const btnReset = document.getElementById("btn-reset");

  const defaultCourseId = window.__INTAKE_DEFAULT_COURSE_ID__;

  function setStatus(msg, isErr) {
    statusEl.textContent = msg || "";
    statusEl.classList.toggle("error", !!isErr);
  }

  function kindLabel(k) {
    if (k === "exercise") return "Aufgabenblatt / exercise sheet";
    if (k === "material") return "Study material";
    return "Lecture / slides";
  }

  function submitCommit(fields) {
    const f = document.createElement("form");
    f.method = "POST";
    f.action = "/intake/commit";
    for (const [k, v] of Object.entries(fields)) {
      const inp = document.createElement("input");
      inp.type = "hidden";
      inp.name = k;
      inp.value = v == null ? "" : String(v);
      f.appendChild(inp);
    }
    document.body.appendChild(f);
    f.submit();
  }

  function resetUi() {
    summary.hidden = true;
    summary.classList.remove("visible");
    fToken.value = "";
    dToken.value = "";
    fileInput.value = "";
    fNew.value = "";
    fKind.value = "lecture";
    fCourse.value = defaultCourseId != null ? String(defaultCourseId) : "";
    setStatus("");
  }

  btnReset.addEventListener("click", function () {
    if (fToken.value) {
      dToken.value = fToken.value;
      document.getElementById("intake-discard-form").submit();
      return;
    }
    resetUi();
  });

  async function previewFile(file) {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      setStatus("Please choose a PDF file.", true);
      return;
    }
    setStatus("Reading PDF…");
    summary.hidden = true;
    summary.classList.remove("visible");
    const fd = new FormData();
    fd.append("file", file);
    let data;
    try {
      const r = await fetch("/intake/preview", { method: "POST", body: fd });
      data = await r.json();
      if (!r.ok) throw new Error(data.error || r.statusText);
    } catch (e) {
      setStatus(e.message || "Preview failed.", true);
      return;
    }

    fToken.value = data.token;
    dToken.value = data.token;

    const top = data.ranked_courses && data.ranked_courses[0];
    if (top && top.score > 0) {
      sumCourse.textContent = `${top.course_name} — match strength ~${Math.round((data.course_confidence || 0) * 100)}%`;
    } else {
      sumCourse.textContent = "Uncertain — please select a course.";
    }

    sumKind.textContent = kindLabel(data.material_kind);
    const kConf = Math.round((data.material_confidence || 0) * 100);
    sumConf.textContent = `${kConf}% · course signals ${Math.round((data.course_confidence || 0) * 100)}%`;
    sumNote.textContent = data.material_note || "";

    fKind.value = data.material_kind === "exercise" || data.material_kind === "material"
      ? data.material_kind
      : "lecture";
    fCourse.value = top && top.course_id ? String(top.course_id) : (defaultCourseId != null ? String(defaultCourseId) : "");
    if (data.auto_commit && top && top.course_id) {
      setStatus("High confidence — saving to your library…");
      submitCommit({
        token: data.token,
        course_id: String(top.course_id),
        new_course_name: "",
        material_kind: data.material_kind,
        lecture_title: "",
      });
      return;
    }

    summary.hidden = false;
    summary.classList.add("visible");
    setStatus("Review the guess below, then save — or adjust course or type first.");
  }

  ["dragenter", "dragover"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault();
      e.stopPropagation();
      drop.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault();
      e.stopPropagation();
      drop.classList.remove("dragover");
    });
  });
  drop.addEventListener("drop", function (e) {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) previewFile(f);
  });
  // Opening the native file dialog twice (drop zone + bubbling from the file input)
  // breaks the flow on macOS/Safari — only trigger programmatic open when the click
  // was not already on the file input.
  drop.addEventListener("click", function (e) {
    if (e.target === fileInput) return;
    if (fileInput.contains(e.target)) return;
    fileInput.click();
  });
  fileInput.addEventListener("click", function (e) {
    e.stopPropagation();
  });
  fileInput.addEventListener("mousedown", function (e) {
    e.stopPropagation();
  });
  drop.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      if (e.target === fileInput) return;
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", function () {
    const f = fileInput.files && fileInput.files[0];
    if (f) previewFile(f);
  });

  if (defaultCourseId != null) {
    fCourse.value = String(defaultCourseId);
  }
})();
