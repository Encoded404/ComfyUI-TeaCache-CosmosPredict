// TeaCacheAnima — collapsible override section + conditional sub-widgets
//
//  1.  An "overrides" combo (hide/show) toggles four override dropdowns.
//  2.  Each dropdown shows conditional sub-widgets (e.g. residual_blend
//      appears only when residual_strategy="blended").
//  3.  Widget state survives workflow save/reload via onSerialize/onConfigure.

(function () {
  // Resolve the ComfyApp instance from the global API surface.
  //   New frontend (~1.45+): window.comfyAPI.app = module, .app.app = instance
  //   Old frontend:           window.comfyAPI.app = instance directly
  function resolveApp() {
    var m = window.comfyAPI && window.comfyAPI.app;
    return (m && m.registerExtension) ? m : (m && m.app);
  }

  // Debug: confirm the JS file itself loaded
  console.log("[TeaCache] JS file loaded, checking environment...");
  console.log("[TeaCache] window.comfyAPI keys:", window.comfyAPI ? Object.keys(window.comfyAPI) : "N/A");
  console.log("[TeaCache] window.comfyAPI.app =", window.comfyAPI && window.comfyAPI.app);
  if (window.comfyAPI && window.comfyAPI.app) {
    console.log("[TeaCache] window.comfyAPI.app.app =", window.comfyAPI.app.app);
  }
  console.log("[TeaCache] Resolved app =", resolveApp());

  // Wait for comfyAPI before registering (poll up to 6 seconds)
  var attempts = 0;
  function register() {
    var app = resolveApp();
    if (!app || !app.registerExtension) {
      if (attempts === 0) console.log("[TeaCache] comfyAPI.app not ready yet, polling...");
      if (++attempts < 60) { setTimeout(register, 100); }
      else console.log("[TeaCache] GAVE UP after " + attempts + " attempts — app still not ready");
      return;
    }
    console.log("[TeaCache] Found ComfyApp instance, registering extension...");

    // ---------------------------------------------------------------------------
    //  Override dropdown definitions
    // ---------------------------------------------------------------------------
    var OVERRIDES = {
      residual_strategy: {
        type: "combo", default: "auto",
        opts: { values: ["auto", "hard", "blended", "scaled"] },
      },
      block_mode: {
        type: "combo", default: "auto",
        opts: { values: ["auto", "all_or_nothing", "split_fraction", "split_groups"] },
      },
      accumulation_type: {
        type: "combo", default: "auto",
        opts: { values: ["auto", "hard_reset", "carry_over", "leaky", "windowed"] },
      },
      step_schedule: {
        type: "combo", default: "auto",
        opts: { values: ["auto", "constant", "cosine", "linear_ramp", "linear_decay", "bell"] },
      },
    };

    // ---------------------------------------------------------------------------
    //  Conditional sub-widgets keyed by parent dropdown name
    // ---------------------------------------------------------------------------
    var CONDITIONAL = {
      residual_strategy: {
        blended:  { name: "residual_blend",  type: "number", default: 0.5, opts: { min: 0.0, max: 1.0, step: 0.01 } },
        scaled:   { name: "residual_scale",  type: "number", default: 0.8, opts: { min: 0.01, max: 1.0, step: 0.01 } },
      },
      accumulation_type: {
        leaky:    { name: "leak_factor",     type: "number", default: 0.9, opts: { min: 0.01, max: 0.999, step: 0.001 } },
        windowed: { name: "window_size",     type: "number", default: 5,   opts: { min: 2, max: 50, step: 1 } },
      },
      block_mode: {
        split_fraction: { name: "always_fraction", type: "number", default: 0.36, opts: { min: 0.01, max: 0.99, step: 0.01 } },
      },
    };

    var CONDITIONAL_NAMES = new Set();
    for (var k in CONDITIONAL) {
      if (!CONDITIONAL.hasOwnProperty(k)) continue;
      for (var k2 in CONDITIONAL[k]) {
        if (!CONDITIONAL[k].hasOwnProperty(k2)) continue;
        CONDITIONAL_NAMES.add(CONDITIONAL[k][k2].name);
      }
    }
    var ALL_DYNAMIC = new Set();
    for (var n in OVERRIDES) {
      if (OVERRIDES.hasOwnProperty(n)) ALL_DYNAMIC.add(n);
    }
    CONDITIONAL_NAMES.forEach(function(n) { ALL_DYNAMIC.add(n); });
    ALL_DYNAMIC.add("overrides");

    // ---------------------------------------------------------------------------
    //  Helpers
    // ---------------------------------------------------------------------------
    function reflow(node) {
      if (node._setConcreteSlots) node._setConcreteSlots();
      if (node.arrange) node.arrange();
      var app = resolveApp();
      var canvas = app && app.canvas;
      if (canvas) canvas.setDirty(true, true);
    }

    function removeW(node, name) {
      var widgets = node.widgets || [];
      for (var i = 0; i < widgets.length; i++) {
        if (widgets[i].name === name && node.removeWidget) {
          node.removeWidget(widgets[i]);
          return;
        }
      }
    }

    function addCond(node, typeName, value, saved) {
      var cfg = CONDITIONAL[typeName] && CONDITIONAL[typeName][value];
      if (!cfg) return;
      var widgets = node.widgets || [];
      for (var i = 0; i < widgets.length; i++) {
        if (widgets[i].name === cfg.name) return;
      }
      node.addWidget(cfg.type, cfg.name, (saved && saved[cfg.name] !== undefined) ? saved[cfg.name] : cfg.default, null, cfg.opts);
    }

    function syncCond(node, saved) {
      for (var typeName in CONDITIONAL) {
        if (!CONDITIONAL.hasOwnProperty(typeName)) continue;
        var widgets = node.widgets || [];
        var dd = null;
        for (var i = 0; i < widgets.length; i++) {
          if (widgets[i].name === typeName) { dd = widgets[i]; break; }
        }
        if (!dd) continue;
        var owned = {};
        for (var k in CONDITIONAL[typeName]) {
          if (CONDITIONAL[typeName].hasOwnProperty(k))
            owned[CONDITIONAL[typeName][k].name] = true;
        }
        for (var i = (node.widgets || []).length - 1; i >= 0; i--) {
          if (owned[node.widgets[i].name]) removeW(node, node.widgets[i].name);
        }
        addCond(node, typeName, dd.value, saved);
      }
    }

    function showOver(node, saved) {
      for (var name in OVERRIDES) {
        if (!OVERRIDES.hasOwnProperty(name)) continue;
        var exists = false;
        for (var i = 0; i < (node.widgets || []).length; i++) {
          if (node.widgets[i].name === name) { exists = true; break; }
        }
        if (exists) continue;
        var cfg = OVERRIDES[name];
        var w = node.addWidget(cfg.type, name, (saved && saved[name] !== undefined) ? saved[name] : cfg.default, null, cfg.opts);
        if (CONDITIONAL[name]) {
          var orig = w.callback;
          w.callback = function(v, canvas, n) {
            var node_ = n || this; // ComfyUI passes node as 3rd arg
            if (orig) orig.call(w, v, canvas, node_);
            syncCond(node_, {});
            reflow(node_);
          };
        }
      }
      syncCond(node, saved);
    }

    function hideOver(node) {
      var widgets = node.widgets || [];
      for (var i = widgets.length - 1; i >= 0; i--) {
        if (ALL_DYNAMIC.has(widgets[i].name)) {
          if (node.removeWidget) node.removeWidget(widgets[i]);
        }
      }
    }

    // ---------------------------------------------------------------------------
    //  Register extension
    // ---------------------------------------------------------------------------
    app.registerExtension({
      name: "TeaCache.AnimaOverrides",

      beforeRegisterNodeDef: function(nodeType, nodeData) {
        if (nodeData.name !== "TeaCacheAnima") return;

        var origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
          if (origCreated) origCreated.call(this);
          this.addWidget("combo", "overrides", "hide", function(v, canvas, node) {
            var n = node || this;
            if (v === "show") showOver(n, {});
            else hideOver(n);
            reflow(n);
          }, { values: ["hide", "show"] });
        };

        var origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
          if (origConfigure) origConfigure.call(this, data);
          var saved = data && data._teacache_overrides;
          if (saved && Object.keys(saved).length > 0) {
            showOver(this, saved);
            reflow(this);
          }
        };

        var origSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
          if (origSerialize) origSerialize.call(this, data);
          var saved = {};
          var widgets = this.widgets || [];
          for (var i = 0; i < widgets.length; i++) {
            if (ALL_DYNAMIC.has(widgets[i].name))
              saved[widgets[i].name] = widgets[i].value;
          }
          if (Object.keys(saved).length > 0) data._teacache_overrides = saved;
        };
      },
    });

    console.log("TeaCacheAnima: override extension loaded");
  }

  register();
})();
