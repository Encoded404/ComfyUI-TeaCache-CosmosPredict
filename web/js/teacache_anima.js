// TeaCacheAnima — collapsible override section + conditional sub-widgets
//
//  1.  An "overrides" combo (hide/show) toggles four override dropdowns.
//  2.  Each dropdown shows conditional sub-widgets (e.g. residual_blend
//      appears only when residual_strategy="blended").
//  3.  Widget state survives workflow save/reload via onSerialize/onConfigure.

(function () {
  const app = window.comfyAPI?.app;
  if (!app) return;

  // ---------------------------------------------------------------------------
  //  Override dropdowns
  // ---------------------------------------------------------------------------
  const OVERRIDES = {
    residual_strategy: {
      type: "combo",
      default: "auto",
      opts: { values: ["auto", "hard", "blended", "scaled"] },
    },
    block_mode: {
      type: "combo",
      default: "auto",
      opts: {
        values: ["auto", "all_or_nothing", "split_fraction", "split_groups"],
      },
    },
    accumulation_type: {
      type: "combo",
      default: "auto",
      opts: {
        values: [
          "auto",
          "hard_reset",
          "carry_over",
          "leaky",
          "windowed",
        ],
      },
    },
    step_schedule: {
      type: "combo",
      default: "auto",
      opts: {
        values: [
          "auto",
          "constant",
          "cosine",
          "linear_ramp",
          "linear_decay",
          "bell",
        ],
      },
    },
  };

  // ---------------------------------------------------------------------------
  //  Conditional sub-widgets keyed by parent dropdown name
  // ---------------------------------------------------------------------------
  const CONDITIONAL = {
    residual_strategy: {
      blended: {
        name: "residual_blend",
        type: "number",
        default: 0.5,
        opts: { min: 0.0, max: 1.0, step: 0.01 },
      },
      scaled: {
        name: "residual_scale",
        type: "number",
        default: 0.8,
        opts: { min: 0.01, max: 1.0, step: 0.01 },
      },
    },
    accumulation_type: {
      leaky: {
        name: "leak_factor",
        type: "number",
        default: 0.9,
        opts: { min: 0.01, max: 0.999, step: 0.001 },
      },
      windowed: {
        name: "window_size",
        type: "number",
        default: 5,
        opts: { min: 2, max: 50, step: 1 },
      },
    },
    block_mode: {
      split_fraction: {
        name: "always_fraction",
        type: "number",
        default: 0.36,
        opts: { min: 0.01, max: 0.99, step: 0.01 },
      },
    },
  };

  // Set of ALL widget names that may be added/removed dynamically
  const CONDITIONAL_NAMES = new Set();
  for (const m of Object.values(CONDITIONAL))
    for (const c of Object.values(m)) CONDITIONAL_NAMES.add(c.name);
  const ALL_DYNAMIC = new Set([
    ...Object.keys(OVERRIDES),
    ...CONDITIONAL_NAMES,
  ]);

  // ---------------------------------------------------------------------------
  //  Helpers
  // ---------------------------------------------------------------------------
  function _reflow(node) {
    node._setConcreteSlots?.();
    node.arrange?.();
    const canvas = window.comfyAPI?.app?.canvas;
    if (canvas) canvas.setDirty(true, true);
  }

  function _removeWidget(node, name) {
    const w = node.widgets?.find((w) => w.name === name);
    if (w && node.removeWidget) node.removeWidget(w);
  }

  function _addConditional(node, typeName, value, saved) {
    const cfg = CONDITIONAL[typeName]?.[value];
    if (!cfg) return;
    if (node.widgets?.find((w) => w.name === cfg.name)) return;
    node.addWidget(
      cfg.type,
      cfg.name,
      saved?.[cfg.name] ?? cfg.default,
      null,
      cfg.opts,
    );
  }

  function _syncConditionals(node, saved) {
    for (const typeName of Object.keys(CONDITIONAL)) {
      const dropdown = node.widgets?.find((w) => w.name === typeName);
      if (!dropdown) continue;
      const cur = dropdown.value;

      // Remove stale conditionals owned by this type
      const owned = new Set(
        Object.values(CONDITIONAL[typeName]).map((c) => c.name),
      );
      for (const w of [...(node.widgets || [])])
        if (owned.has(w.name)) _removeWidget(node, w.name);

      // Add the matching one
      _addConditional(node, typeName, cur, saved);
    }
  }

  function _addOverrides(node, saved = {}) {
    for (const [name, cfg] of Object.entries(OVERRIDES)) {
      if (node.widgets?.find((w) => w.name === name)) continue;
      const w = node.addWidget(
        cfg.type,
        name,
        saved[name] ?? cfg.default,
        null,
        cfg.opts,
      );
      // Wire parent callback to keep conditionals in sync
      if (CONDITIONAL[name]) {
        const orig = w.callback;
        w.callback = function (v) {
          if (orig) orig.call(this, v);
          _syncConditionals(node, {});
          _reflow(node);
        };
      }
    }
    _syncConditionals(node, saved);
  }

  function _removeOverrides(node) {
    for (const w of [...(node.widgets || [])])
      if (ALL_DYNAMIC.has(w.name)) node.removeWidget(w);
  }

  // ---------------------------------------------------------------------------
  //  Extension registration
  // ---------------------------------------------------------------------------
  app.registerExtension({
    name: "TeaCache.AnimaOverrides",

    beforeRegisterNodeDef(nodeType, nodeData) {
      if (nodeData.name !== "TeaCacheAnima") return;

      // ── onNodeCreated ──────────────────────────────────────────────────
      const origCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        if (origCreated) origCreated.call(this);

        this.addWidget(
          "combo",
          "overrides",
          "hide",
          (value) => {
            if (value === "show") _addOverrides(this);
            else _removeOverrides(this);
            _reflow(this);
          },
          { values: ["hide", "show"] },
        );
      };

      // ── onConfigure (restore after reload) ─────────────────────────────
      const origConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (data) {
        if (origConfigure) origConfigure.call(this, data);
        const saved = data?._teacache_overrides;
        if (saved && Object.keys(saved).length > 0) {
          _addOverrides(this, saved);
          _reflow(this);
        }
      };

      // ── onSerialize (persist -> reload) ────────────────────────────────
      const origSerialize = nodeType.prototype.onSerialize;
      nodeType.prototype.onSerialize = function (data) {
        if (origSerialize) origSerialize.call(this, data);
        const saved = {};
        for (const w of this.widgets || [])
          if (ALL_DYNAMIC.has(w.name)) saved[w.name] = w.value;
        if (Object.keys(saved).length > 0) data._teacache_overrides = saved;
      };
    },
  });
})();
