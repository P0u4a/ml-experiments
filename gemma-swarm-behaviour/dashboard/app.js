(function () {
  "use strict";

  const payload = window.TRACE_DATA || { runs: [] };
  const runs = payload.runs || [];
  const query = new URLSearchParams(window.location.search);
  const requestedRunId = query.get("run");
  const initialRunId = runs.some((run) => run.id === requestedRunId) ? requestedRunId : runs[runs.length - 1]?.id || null;
  const state = {
    runId: initialRunId,
    condition: query.get("condition") || "all",
    outcome: query.get("outcome") || "all",
    runSearch: "",
    view: "turns",
    traceSearch: "",
    eventType: "all",
    agent: "all",
    selectedKey: query.get("item") || "summary",
    tab: "trace",
    annotations: [],
    annotationDraft: null,
  };

  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  const json = (value) => escapeHtml(JSON.stringify(value, null, 2));
  const currentRun = () => runs.find((run) => run.id === state.runId) || runs[runs.length - 1];
  const formatMs = (ms) => ms == null ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
  const formatNumber = (value) => Number(value || 0).toLocaleString();
  const formatDate = (value) => value ? new Date(value).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "Unknown";
  const shortId = (id) => id
    .replace("eval-tampering-session-private-board-private-board-causal-20260902-session-smoke-r", "private session · smoke ")
    .replace("eval-tampering-session-private-board-private-board-causal-20260902-main-session-r", "private session · ")
    .replace("eval-tampering-swarm-private-board-private-board-causal-20260902-swarm-smoke-r", "private swarm · smoke ")
    .replace("eval-tampering-swarm-private-board-private-board-causal-20260902-main-swarm-r", "private swarm · ")
    .replace("eval-tampering-swarm-counterfactual-20260901T102251Z-r", "shared swarm · ")
    .replace("eval-tampering-shared-board-swarm-", "shared swarm · ");
  const toolSummary = (turn) => {
    const call = turn.tool_calls?.[0];
    if (!call) return "Assistant response";
    const fn = call.function || {};
    try {
      const args = JSON.parse(fn.arguments || "{}");
      if (fn.name === "execute_command") return args.command || fn.name;
      if (fn.name === "submit") return `Submit · ${args.suspicious_observations || "result"}`;
      if (fn.name === "read_messageboard") return "Read message board";
      if (fn.name === "send_message") return `Send · ${args.message || "message"}`;
    } catch (_) {}
    return fn.name || "Tool call";
  };
  const eventLabel = (event) => event.type.replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase());
  const annotationsForRun = (runId) => state.annotations.filter((annotation) => annotation.run_id === runId);
  const annotationsForItem = (runId, itemKey) => annotationsForRun(runId).filter((annotation) => annotation.target?.item_key === itemKey);
  const annotationCount = (runId, itemKey = null) => itemKey ? annotationsForItem(runId, itemKey).length : annotationsForRun(runId).length;

  function annotatedText(value, itemKey, field) {
    const text = String(value ?? "");
    const relevant = annotationsForItem(currentRun()?.id, itemKey)
      .filter((annotation) => annotation.target?.field === field)
      .filter((annotation) => Number.isInteger(annotation.target?.start) && Number.isInteger(annotation.target?.end))
      .filter((annotation) => annotation.target.start >= 0 && annotation.target.end <= text.length && annotation.target.end > annotation.target.start);
    if (!relevant.length) return escapeHtml(text);
    const boundaries = [...new Set([0, text.length, ...relevant.flatMap((annotation) => [annotation.target.start, annotation.target.end])])].sort((a, b) => a - b);
    return boundaries.slice(0, -1).map((start, index) => {
      const end = boundaries[index + 1];
      const segment = escapeHtml(text.slice(start, end));
      const active = relevant.filter((annotation) => annotation.target.start <= start && annotation.target.end >= end);
      if (!active.length) return segment;
      const top = active[active.length - 1];
      const ids = active.map((annotation) => annotation.id).join(",");
      const title = active.map((annotation) => annotation.comment).join(" · ");
      return `<mark class="annotation-mark ${escapeHtml(top.color || "yellow")}" data-annotation-ids="${escapeHtml(ids)}" title="${escapeHtml(title)}">${segment}</mark>`;
    }).join("");
  }

  function annotatable(value, itemKey, field, tag = "div", className = "prose", label = "Trace text") {
    return `<${tag} class="${className} annotatable" data-item-key="${escapeHtml(itemKey)}" data-field="${escapeHtml(field)}" data-label="${escapeHtml(label)}">${annotatedText(value, itemKey, field)}</${tag}>`;
  }

  function cohortStats() {
    const privateSession = runs.filter((run) => run.condition === "solo-private");
    const privateSwarm = runs.filter((run) => run.condition === "swarm-private");
    const sharedSwarm = runs.filter((run) => run.condition === "swarm-shared");
    const agentDisclosures = (items) => items.reduce((total, run) => total + run.disclosed_count, 0);
    const agentTotal = (items) => items.reduce((total, run) => total + run.agent_count, 0);
    $("#cohort-stats").innerHTML = `
      <div class="stat-pill"><strong>${agentDisclosures(privateSession)}/${agentTotal(privateSession)}</strong><span>solo-framed disclose</span></div>
      <div class="stat-pill"><strong>${agentDisclosures(privateSwarm)}/${agentTotal(privateSwarm)}</strong><span>private swarm disclose</span></div>
      <div class="stat-pill"><strong>${agentDisclosures(sharedSwarm)}/${agentTotal(sharedSwarm)}</strong><span>shared swarm disclose</span></div>`;
    $("#generated-at").textContent = `Snapshot ${formatDate(payload.generated_at)}`;
  }

  function filteredRuns() {
    const query = state.runSearch.toLowerCase();
    return runs.filter((run) => {
      if (state.condition !== "all" && run.condition !== state.condition) return false;
      if (state.outcome !== "all" && run.outcome !== state.outcome) return false;
      return !query || run.id.toLowerCase().includes(query) || (run.suspicious_observations || "").toLowerCase().includes(query);
    });
  }

  function renderRuns() {
    const visible = filteredRuns();
    $("#run-count").textContent = `${visible.length} of ${runs.length} runs`;
    $("#run-list").innerHTML = visible.map((run) => `
      <button class="run-card ${run.id === state.runId ? "active" : ""}" data-run-id="${escapeHtml(run.id)}">
        <div class="run-card-top">
          <span class="run-name">${escapeHtml(shortId(run.id))}</span>
          <span class="badge ${run.condition}">${run.condition}</span>
        </div>
        <div class="run-card-meta">
          <span class="badge ${run.outcome}">${run.outcome}</span>
          <span>${annotationCount(run.id) ? `<span class="comment-count">● ${annotationCount(run.id)}</span> · ` : ""}${run.generations} turns · ${run.agent_count} agents</span>
        </div>
      </button>`).join("") || `<div class="empty"><strong>No matching sessions</strong>Try a different filter.</div>`;
    document.querySelectorAll("[data-run-id]").forEach((button) => button.addEventListener("click", () => selectRun(button.dataset.runId)));
  }

  function selectRun(id) {
    state.runId = id;
    state.selectedKey = "summary";
    state.tab = "trace";
    state.agent = "all";
    state.traceSearch = "";
    $("#trace-search").value = "";
    renderAll();
  }

  function traceItems(run) {
    const query = state.traceSearch.toLowerCase();
    if (state.view === "events") {
      return run.events
        .map((event, index) => ({ kind: "event", key: `event-${index}`, index, event }))
        .filter((item) => state.eventType === "all" || item.event.type === state.eventType)
        .filter((item) => state.agent === "all" || item.event.agent_id === state.agent)
        .filter((item) => !query || JSON.stringify(item.event).toLowerCase().includes(query));
    }
    return run.turns
      .map((turn, index) => ({ kind: "turn", key: `turn-${index}`, index, turn }))
      .filter((item) => state.agent === "all" || item.turn.agent_id === state.agent)
      .filter((item) => !query || `${item.turn.reasoning || ""} ${toolSummary(item.turn)} ${JSON.stringify(item.turn.tool_results)}`.toLowerCase().includes(query));
  }

  function renderTimeline() {
    const run = currentRun();
    if (!run) return;
    $("#timeline-title").textContent = shortId(run.id);
    $("#agent-filter").hidden = run.agent_count <= 1;
    $("#agent-filter").innerHTML = `<option value="all">All agents</option>${run.agents.map((agent) => `<option value="${escapeHtml(agent.id)}" ${state.agent === agent.id ? "selected" : ""}>${escapeHtml(agent.id)}</option>`).join("")}`;
    const eventTypes = [...new Set(run.events.map((event) => event.type))].sort();
    $("#event-filter").hidden = state.view !== "events";
    $("#event-filter").innerHTML = `<option value="all">All events</option>${eventTypes.map((type) => `<option value="${escapeHtml(type)}" ${state.eventType === type ? "selected" : ""}>${escapeHtml(type.replaceAll("_", " "))}</option>`).join("")}`;
    const items = traceItems(run);
    const cards = items.map((item) => {
      if (item.kind === "turn") {
        const turn = item.turn;
        const call = turn.tool_calls?.[0]?.function?.name;
        const comments = annotationCount(run.id, item.key);
        return `<button class="timeline-item ${call ? "tool" : ""} ${state.selectedKey === item.key ? "active" : ""}" data-key="${item.key}">
          <span class="event-icon">${item.index + 1}</span>
          <span class="timeline-copy"><span class="timeline-label-row"><span class="timeline-label">${escapeHtml(toolSummary(turn))}</span>${comments ? `<span class="comment-count">● ${comments}</span>` : ""}</span><span class="timeline-sub"><span>${escapeHtml(turn.agent_id)}</span><span>${formatMs(turn.latency_ms)}</span><span>${formatNumber(turn.usage?.completion_tokens)} out</span><span>${escapeHtml(turn.finish_reason || "done")}</span></span></span>
        </button>`;
      }
      const event = item.event;
      const comments = annotationCount(run.id, item.key);
      return `<button class="timeline-item ${event.type.includes("tool") ? "tool" : ""} ${state.selectedKey === item.key ? "active" : ""}" data-key="${item.key}">
        <span class="event-icon">${event.sequence ?? item.index + 1}</span>
        <span class="timeline-copy"><span class="timeline-label-row"><span class="timeline-label">${escapeHtml(eventLabel(event))}</span>${comments ? `<span class="comment-count">● ${comments}</span>` : ""}</span><span class="timeline-sub"><span>${formatDate(event.timestamp)}</span><span>${escapeHtml(event.agent_id || "system")}</span></span></span>
      </button>`;
    }).join("");
    $("#timeline").innerHTML = `<button class="timeline-item summary ${state.selectedKey === "summary" ? "active" : ""}" data-key="summary"><span class="event-icon">◎</span><span class="timeline-copy"><span class="timeline-label-row"><span class="timeline-label">Run overview</span>${annotationCount(run.id) ? `<span class="comment-count">● ${annotationCount(run.id)}</span>` : ""}</span><span class="timeline-sub"><span>${run.generations} turns</span><span>${run.events.length} events</span><span>${run.messageboard.length} board messages</span></span></span></button>${cards || '<div class="empty"><strong>No matching trace items</strong>Clear the trace search or filter.</div>'}`;
    document.querySelectorAll("[data-key]").forEach((button) => button.addEventListener("click", () => { state.selectedKey = button.dataset.key; state.tab = "trace"; renderTimeline(); renderDetail(); }));
  }

  function selectedItem(run) {
    if (state.selectedKey === "summary") return { kind: "summary", run };
    const [kind, index] = state.selectedKey.split("-");
    if (kind === "turn") return { kind, turn: run.turns[Number(index)], index: Number(index) };
    if (kind === "event") return { kind, event: run.events[Number(index)], index: Number(index) };
    return { kind: "summary", run };
  }

  function metric(label, value) { return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`; }
  function section(title, content, className = "") { return `<section class="section ${className}"><div class="section-header"><span>${escapeHtml(title)}</span></div><div class="section-body">${content}</div></section>`; }

  function renderDetail() {
    const run = currentRun();
    if (!run) return;
    const item = selectedItem(run);
    let title = "Run overview";
    let subtitle = run.id;
    let metrics = [metric("Agents", run.agent_count), metric("Disclosed", `${run.disclosed_count}/${run.agent_count}`), metric("Turns", run.generations), metric("Output tokens", formatNumber(run.completion_tokens))];
    let tabs = ["trace", "board", "manifest", "raw", "comments"];
    if (item.kind === "turn") {
      title = toolSummary(item.turn);
      subtitle = `${item.turn.agent_id} · Turn ${item.index + 1} · ${formatDate(item.turn.timestamp)}`;
      metrics = [metric("Agent", item.turn.agent_id), metric("Latency", formatMs(item.turn.latency_ms)), metric("Prompt", formatNumber(item.turn.usage?.prompt_tokens)), metric("Completion", formatNumber(item.turn.usage?.completion_tokens))];
      tabs = ["trace", "request", "response", "comments"];
    } else if (item.kind === "event") {
      title = eventLabel(item.event);
      subtitle = `Event ${item.event.sequence ?? item.index + 1} · ${formatDate(item.event.timestamp)}`;
      metrics = [metric("Actor", item.event.agent_id || "system"), metric("Sequence", item.event.sequence ?? item.index + 1), metric("Type", item.event.type), metric("Run", run.condition)];
      tabs = ["trace", "raw", "comments"];
    }
    const scopedCount = item.kind === "summary" ? annotationCount(run.id) : annotationCount(run.id, `${item.kind}-${item.index}`);
    $("#detail-hero").innerHTML = `<div class="detail-title-row"><div><span class="eyebrow">${escapeHtml(run.condition)} · ${escapeHtml(run.outcome)}</span><h2 class="detail-title">${escapeHtml(title)}</h2><p class="detail-subtitle">${escapeHtml(subtitle)}</p></div><div><span class="badge ${run.outcome}">${run.outcome}</span>${scopedCount ? ` <span class="badge warning">${scopedCount} comment${scopedCount === 1 ? "" : "s"}</span>` : ""}</div></div><div class="metric-row">${metrics.join("")}</div>`;
    if (!tabs.includes(state.tab)) state.tab = tabs[0];
    $("#detail-tabs").innerHTML = tabs.map((tab) => `<button class="${state.tab === tab ? "active" : ""}" data-tab="${tab}">${tab === "trace" ? (item.kind === "summary" ? "Overview" : "Readable") : tab === "comments" ? `Comments${scopedCount ? ` (${scopedCount})` : ""}` : tab.replace(/^./, (m) => m.toUpperCase())}</button>`).join("");
    document.querySelectorAll("[data-tab]").forEach((button) => button.addEventListener("click", () => { state.tab = button.dataset.tab; renderDetail(); }));
    $("#detail-content").innerHTML = detailBody(run, item);
    bindAnnotationInteractions();
  }

  function detailBody(run, item) {
    if (state.tab === "comments") return commentsBody(run, item);
    if (item.kind === "summary") {
      if (state.tab === "board") {
        const messages = run.messageboard.map((message) => `<article class="board-message ${message.seeded ? "seeded" : ""}"><div class="board-message-meta"><span>${message.board_id && message.board_id !== "shared" ? `${escapeHtml(message.board_id)} · ` : ""}Message ${message.id}</span><strong>${escapeHtml(message.author)}</strong><span>${formatDate(message.timestamp)}</span></div>${annotatable(message.body, "summary", `messageboard:${message.board_id || "shared"}:${message.id}`, "div", "prose", `Message board · ${message.board_id || "shared"} · ${message.author}`)}</article>`).join("");
        return section(`Message board · ${run.messageboard.length} entries`, messages || '<div class="empty"><strong>No board messages recorded</strong></div>');
      }
      if (state.tab === "manifest") return section("Manifest", annotatable(JSON.stringify(run.manifest, null, 2), "summary", "manifest", "pre", "", "Manifest"));
      if (state.tab === "raw") return section("Completion record", annotatable(JSON.stringify(run.complete, null, 2), "summary", "complete", "pre", "", "Completion record"));
      const promptSections = run.initial_messages.map((message, index) => section(`${message.role} prompt`, annotatable(message.content, "summary", `prompt:${index}`, "div", "prose", `${message.role} prompt`))).join("");
      const agentCards = run.agents.map((agent) => `<article class="agent-result"><div class="agent-result-head"><strong>${escapeHtml(agent.id)}</strong><span class="badge ${agent.outcome}">${agent.outcome}</span></div><div class="agent-result-metrics"><span>accuracy ${escapeHtml(agent.accuracy)}</span><span>${escapeHtml(agent.turns)} turns</span></div>${annotatable(agent.suspicious_observations || "No suspicious-observation text", "summary", `suspicious_observations:${agent.id}`, "div", "prose", `${agent.id} suspicious observations`)}</article>`).join("");
      const boardScope = run.manifest?.messageboard_scope === "private" ? "private append-only boards" : "shared append-only board";
      return `<div class="result-banner"><span class="badge ${run.outcome}">${run.outcome}</span><div><strong>${run.disclosed_count} disclosed · ${run.withheld_count} withheld</strong><p>Four independent submissions · ${boardScope} · average accuracy ${escapeHtml(run.accuracy)}</p></div></div>${section("Agent outcomes", `<div class="agent-results">${agentCards}</div>`)}${promptSections}`;
    }
    if (item.kind === "event") {
      if (state.tab === "raw") return section("Raw structured event", annotatable(JSON.stringify(item.event, null, 2), `event-${item.index}`, "event_raw", "pre", "", "Raw structured event"));
      return section("Event payload", annotatable(JSON.stringify(item.event.data || {}, null, 2), `event-${item.index}`, "event_data", "pre", "", "Event payload"));
    }
    const turn = item.turn;
    const itemKey = `turn-${item.index}`;
    if (state.tab === "request") return section("Exact generation request", annotatable(JSON.stringify(turn.request, null, 2), itemKey, "request", "pre", "", "Exact generation request"));
    if (state.tab === "response") return section("Raw llama.cpp response", annotatable(JSON.stringify(turn.response, null, 2), itemKey, "response", "pre", "", "Raw llama.cpp response"));
    const reasoning = section("Reasoning / CoT", annotatable(turn.reasoning || "No reasoning content recorded.", itemKey, "reasoning", "div", "prose", "Reasoning / CoT"), "reasoning");
    const content = turn.content ? section("Assistant content", annotatable(turn.content, itemKey, "assistant_content", "div", "prose", "Assistant content")) : "";
    const calls = (turn.tool_calls || []).map((call, index) => section(`Tool call · ${call.function?.name || index + 1}`, annotatable(JSON.stringify(call.function || call, null, 2), itemKey, `tool_call:${index}`, "pre", "", `Tool call · ${call.function?.name || index + 1}`), "tool-call") + (turn.tool_results?.[index] ? section("Tool result", annotatable(turn.tool_results[index].content || "", itemKey, `tool_result:${index}`, "pre", "", "Tool result"), "tool-result") : "")).join("");
    return reasoning + content + calls + section("Generation metrics", `<pre>${json({ usage: turn.usage, timings: turn.timings, latency_ms: turn.latency_ms, attempts: turn.attempts, retry_errors: turn.retry_errors })}</pre>`);
  }

  function commentsBody(run, item) {
    const itemKey = item.kind === "summary" ? null : `${item.kind}-${item.index}`;
    const annotations = itemKey ? annotationsForItem(run.id, itemKey) : annotationsForRun(run.id);
    if (!annotations.length) {
      return `<div class="empty"><strong>No comments here yet</strong>Select any text in the readable trace, prompt, manifest, or raw JSON to add one.</div>`;
    }
    return `<div class="comments-list">${annotations.map((annotation) => `
      <article class="comment-card ${state.selectedAnnotationId === annotation.id ? "active" : ""}" id="comment-${escapeHtml(annotation.id)}">
        <div class="comment-quote ${escapeHtml(annotation.color || "yellow")}">${escapeHtml(annotation.quote)}</div>
        <div class="comment-body">
          <p>${escapeHtml(annotation.comment)}</p>
          <div class="comment-meta">
            <span>${escapeHtml(annotation.target?.label || annotation.target?.field || "Trace text")} · ${formatDate(annotation.created_at)}</span>
            <span class="comment-controls">
              <button class="action-button" data-jump-annotation="${escapeHtml(annotation.id)}">Show highlight</button>
              <button class="action-button danger" data-delete-annotation="${escapeHtml(annotation.id)}">Delete</button>
            </span>
          </div>
        </div>
      </article>`).join("")}</div>`;
  }

  function showToast(message) {
    const toast = $("#toast");
    toast.textContent = message;
    toast.hidden = false;
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => { toast.hidden = true; }, 2600);
  }

  function hideAnnotationPopover() {
    $("#annotation-popover").hidden = true;
    state.annotationDraft = null;
  }

  function showAnnotationPopover(range, container) {
    const text = container.textContent || "";
    const selected = range.toString();
    if (!selected.trim() || selected.length > 20000) return;
    const before = range.cloneRange();
    before.selectNodeContents(container);
    before.setEnd(range.startContainer, range.startOffset);
    const start = before.toString().length;
    const end = start + selected.length;
    const rect = range.getBoundingClientRect();
    state.annotationDraft = {
      quote: selected,
      color: "yellow",
      target: {
        item_key: container.dataset.itemKey,
        field: container.dataset.field,
        label: container.dataset.label,
        start,
        end,
        prefix: text.slice(Math.max(0, start - 120), start),
        suffix: text.slice(end, end + 120),
      },
    };
    const popover = $("#annotation-popover");
    popover.innerHTML = `
      <div class="selection-quote">${escapeHtml(selected.length > 240 ? `${selected.slice(0, 240)}…` : selected)}</div>
      <textarea id="annotation-comment" placeholder="Add a comment…" maxlength="4000"></textarea>
      <div class="annotation-actions">
        <div class="color-palette" aria-label="Highlight color">
          ${["yellow", "pink", "blue", "green"].map((color) => `<button class="color-chip ${color} ${color === "yellow" ? "active" : ""}" data-color="${color}" aria-label="${color}"></button>`).join("")}
        </div>
        <div class="action-buttons"><button class="action-button" id="cancel-annotation">Cancel</button><button class="action-button primary" id="save-annotation">Save comment</button></div>
      </div>`;
    popover.hidden = false;
    const width = 340;
    const left = Math.max(12, Math.min(window.innerWidth - width - 12, rect.left + rect.width / 2 - width / 2));
    const top = Math.max(12, Math.min(window.innerHeight - 230, rect.bottom + 9));
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
    popover.querySelectorAll("[data-color]").forEach((button) => button.addEventListener("click", () => {
      state.annotationDraft.color = button.dataset.color;
      popover.querySelectorAll("[data-color]").forEach((item) => item.classList.toggle("active", item === button));
    }));
    $("#cancel-annotation").addEventListener("click", hideAnnotationPopover);
    $("#save-annotation").addEventListener("click", saveAnnotation);
    $("#annotation-comment").focus();
  }

  async function saveAnnotation() {
    const comment = $("#annotation-comment")?.value?.trim();
    if (!comment || !state.annotationDraft) {
      showToast("Write a comment before saving.");
      return;
    }
    const button = $("#save-annotation");
    button.disabled = true;
    try {
      const response = await fetch("/api/annotations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: currentRun().id, comment, ...state.annotationDraft }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "Could not save annotation");
      state.annotations.push(body.annotation);
      state.selectedAnnotationId = body.annotation.id;
      hideAnnotationPopover();
      window.getSelection()?.removeAllRanges();
      renderAll();
      showToast("Comment saved beside the run logs.");
    } catch (error) {
      button.disabled = false;
      showToast(error.message || "Could not save annotation");
    }
  }

  async function deleteAnnotation(annotationId) {
    const annotation = state.annotations.find((item) => item.id === annotationId);
    if (!annotation || !window.confirm("Delete this comment? The deletion remains recorded in the annotation log.")) return;
    try {
      const response = await fetch(`/api/annotations/${encodeURIComponent(annotationId)}?run_id=${encodeURIComponent(annotation.run_id)}`, { method: "DELETE" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "Could not delete annotation");
      state.annotations = state.annotations.filter((item) => item.id !== annotationId);
      state.selectedAnnotationId = null;
      renderAll();
      showToast("Comment deleted; tombstone retained in the log.");
    } catch (error) {
      showToast(error.message || "Could not delete annotation");
    }
  }

  function jumpToAnnotation(annotationId) {
    const annotation = state.annotations.find((item) => item.id === annotationId);
    if (!annotation) return;
    state.runId = annotation.run_id;
    state.selectedKey = annotation.target.item_key;
    state.view = annotation.target.item_key.startsWith("event-") ? "events" : "turns";
    state.tab = annotation.target.field === "request" ? "request" : annotation.target.field === "response" ? "response" : annotation.target.field === "event_raw" ? "raw" : annotation.target.item_key === "summary" && annotation.target.field === "manifest" ? "manifest" : annotation.target.item_key === "summary" && annotation.target.field === "complete" ? "raw" : "trace";
    state.selectedAnnotationId = annotationId;
    document.querySelectorAll("#view-toggle button").forEach((button) => button.classList.toggle("active", button.dataset.value === state.view));
    renderAll();
    requestAnimationFrame(() => {
      const mark = [...document.querySelectorAll("[data-annotation-ids]")].find((element) => element.dataset.annotationIds.split(",").includes(annotationId));
      mark?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  function bindAnnotationInteractions() {
    const detail = $("#detail-content");
    detail.onmouseup = () => {
      setTimeout(() => {
        const selection = window.getSelection();
        if (!selection || selection.isCollapsed || !selection.rangeCount) return;
        const range = selection.getRangeAt(0);
        const common = range.commonAncestorContainer.nodeType === Node.TEXT_NODE ? range.commonAncestorContainer.parentElement : range.commonAncestorContainer;
        const container = common?.closest?.(".annotatable");
        if (!container || !container.contains(range.startContainer) || !container.contains(range.endContainer)) return;
        showAnnotationPopover(range, container);
      }, 0);
    };
    detail.onclick = (event) => {
      const mark = event.target.closest?.("[data-annotation-ids]");
      if (mark && window.getSelection()?.isCollapsed) {
        state.selectedAnnotationId = mark.dataset.annotationIds.split(",")[0];
        state.tab = "comments";
        renderDetail();
        requestAnimationFrame(() => $(`#comment-${state.selectedAnnotationId}`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
        return;
      }
      const jump = event.target.closest?.("[data-jump-annotation]");
      if (jump) jumpToAnnotation(jump.dataset.jumpAnnotation);
      const remove = event.target.closest?.("[data-delete-annotation]");
      if (remove) deleteAnnotation(remove.dataset.deleteAnnotation);
    };
  }

  async function loadAnnotations() {
    try {
      const response = await fetch("/api/annotations", { cache: "no-store" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "Could not load annotations");
      state.annotations = body.annotations || [];
      renderAll();
    } catch (error) {
      showToast("Annotations are unavailable. Restart the dashboard server.");
    }
  }

  function renderAll() { cohortStats(); renderRuns(); renderTimeline(); renderDetail(); }
  function setSegment(container, value, stateKey) {
    state[stateKey] = value;
    container.querySelectorAll("button").forEach((button) => button.classList.toggle("active", button.dataset.value === value));
    if (stateKey === "view") { state.selectedKey = "summary"; $("#event-filter").hidden = value !== "events"; renderTimeline(); renderDetail(); }
    else renderRuns();
  }

  $("#run-search").addEventListener("input", (event) => { state.runSearch = event.target.value; renderRuns(); });
  $("#trace-search").addEventListener("input", (event) => { state.traceSearch = event.target.value; renderTimeline(); });
  $("#event-filter").addEventListener("change", (event) => { state.eventType = event.target.value; renderTimeline(); });
  $("#agent-filter").addEventListener("change", (event) => { state.agent = event.target.value; state.selectedKey = "summary"; renderTimeline(); renderDetail(); });
  $("#condition-filter").addEventListener("click", (event) => event.target.dataset.value && setSegment(event.currentTarget, event.target.dataset.value, "condition"));
  $("#outcome-filter").addEventListener("click", (event) => event.target.dataset.value && setSegment(event.currentTarget, event.target.dataset.value, "outcome"));
  $("#view-toggle").addEventListener("click", (event) => event.target.dataset.value && setSegment(event.currentTarget, event.target.dataset.value, "view"));
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    const run = currentRun();
    if (!run) return;
    if (event.key === "[" || event.key === "]") {
      const visible = filteredRuns();
      const index = visible.findIndex((item) => item.id === run.id);
      const next = event.key === "]" ? Math.min(visible.length - 1, index + 1) : Math.max(0, index - 1);
      if (visible[next]) selectRun(visible[next].id);
      return;
    }
    if (!["j", "k", "ArrowDown", "ArrowUp"].includes(event.key)) return;
    event.preventDefault();
    const items = ["summary", ...traceItems(run).map((item) => item.key)];
    const index = Math.max(0, items.indexOf(state.selectedKey));
    const delta = ["j", "ArrowDown"].includes(event.key) ? 1 : -1;
    state.selectedKey = items[Math.max(0, Math.min(items.length - 1, index + delta))];
    state.tab = "trace";
    renderTimeline(); renderDetail();
    document.querySelector(`[data-key="${state.selectedKey}"]`)?.scrollIntoView({ block: "nearest" });
  });

  [["condition-filter", "condition"], ["outcome-filter", "outcome"]].forEach(([id, key]) => {
    document.querySelectorAll(`#${id} button`).forEach((button) => button.classList.toggle("active", button.dataset.value === state[key]));
  });
  renderAll();
  loadAnnotations();
})();
