// TeaCacheAnima — collapsible override section + conditional sub-widgets
//
//  Uses widgets defined in Python INPUT_TYPES['optional'] and toggles
//  their visibility (widget.hidden) rather than add/removing them.
//  This ensures the new ComfyUI frontend includes their values in the
//  prompt payload.
//
//  1.  An "overrides" combo (hide/show) toggles the override widgets.
//  2.  Each dropdown shows conditional sub-widgets (e.g. residual_scale
//      appears only when residual_strategy="scaled").
//  3.  Widget state survives workflow save/reload via onSerialize/onConfigure.

(function () {
  function resolveApp() {
    var m = window.comfyAPI && window.comfyAPI.app;
    return (m && m.registerExtension) ? m : (m && m.app);
  }

  console.log("[TeaCache] JS file loaded, checking environment...");
  if (window.comfyAPI && window.comfyAPI.app) {
    console.log("[TeaCache] window.comfyAPI.app.app =", window.comfyAPI.app.app);
  }

  var attempts = 0;
  function register() {
    var app = resolveApp();
    if (!app || !app.registerExtension) {
      if (attempts === 0) console.log("[TeaCache] comfyAPI.app not ready yet, polling...");
      if (++attempts < 60) { setTimeout(register, 100); }
      else console.log("[TeaCache] GAVE UP after " + attempts + " attempts");
      return;
    }

    // ---- Widget names (must match Python INPUT_TYPES['optional'] keys) ----
    var MAIN_OVERRIDES = ["residual_strategy", "block_mode", "accumulation_type", "step_schedule"];
    var CONDITIONALS = {
      residual_strategy: {
        blended:  "residual_blend",
        scaled:   "residual_scale",
      },
      accumulation_type: {
        leaky:    "leak_factor",
        windowed: "window_size",
      },
      block_mode: {
        split_fraction: "always_fraction",
      },
    };
    var ALL_NAMES = new Set(MAIN_OVERRIDES.concat("overrides"));
    for (var p in CONDITIONALS) {
      if (!CONDITIONALS.hasOwnProperty(p)) continue;
      for (var v in CONDITIONALS[p]) {
        if (CONDITIONALS[p].hasOwnProperty(v)) ALL_NAMES.add(CONDITIONALS[p][v]);
      }
    }

    // ---- Helpers ----
    function getWidget(node, name) {
      var w = node.widgets || [];
      for (var i = 0; i < w.length; i++) { if (w[i].name === name) return w[i]; }
      return null;
    }

    function setHidden(node, name, hidden) {
      var w = getWidget(node, name);
      if (w) w.hidden = hidden;
    }

    function reflow(node) {
      if (node._setConcreteSlots) node._setConcreteSlots();
      if (node.graph && node.arrange) node.arrange();
      var a = resolveApp();
      var c = a && a.canvas;
      if (c) c.setDirty(true, true);
    }

    function syncCond(node, saved) {
      for (var typeName in CONDITIONALS) {
        if (!CONDITIONALS.hasOwnProperty(typeName)) continue;
        var dd = getWidget(node, typeName);
        if (!dd) continue;
        // Hide all children of this type
        for (var k in CONDITIONALS[typeName]) {
          if (CONDITIONALS[typeName].hasOwnProperty(k))
            setHidden(node, CONDITIONALS[typeName][k], true);
        }
        // Show the matching child (if we know it)
        var child = CONDITIONALS[typeName][dd.value];
        if (child) {
          var savedVal = saved && saved[child] !== undefined ? saved[child] : undefined;
          setHidden(node, child, false);
          var cw = getWidget(node, child);
          if (cw && savedVal !== undefined) cw.value = savedVal;
        }
      }
    }

    function showOver(node, saved) {
      for (var i = 0; i < MAIN_OVERRIDES.length; i++) {
        setHidden(node, MAIN_OVERRIDES[i], false);
        var w = getWidget(node, MAIN_OVERRIDES[i]);
        if (w && saved && saved[MAIN_OVERRIDES[i]] !== undefined)
          w.value = saved[MAIN_OVERRIDES[i]];
      }
      syncCond(node, saved);
    }

    function hideOver(node) {
      ALL_NAMES.forEach(function(n) {
        if (n !== "overrides") setHidden(node, n, true);
      });
    }

    // ---- Register extension ----
    app.registerExtension({
      name: "TeaCache.AnimaOverrides",

      beforeRegisterNodeDef: function(nodeType, nodeData) {
        if (nodeData.name !== "TeaCacheAnima") return;

        var origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
          if (origCreated) origCreated.call(this);

          // Override dropdown callbacks so conditional widgets respond
          var attachCb = function(n, name) {
            var w = getWidget(n, name);
            if (!w || w._tcHooked) return;
            w._tcHooked = true;
            var orig = w.callback;
            w.callback = function(v, canvas, node_) {
              var nn = node_ || n;
              if (orig) orig.call(w, v, canvas, nn);
              syncCond(nn, {});
              reflow(nn);
            };
          };
          for (var i = 0; i < MAIN_OVERRIDES.length; i++) attachCb(this, MAIN_OVERRIDES[i]);

          // Add the "overrides" toggle at the end
          this.addWidget("combo", "overrides", "hide", function(v, canvas, node) {
            var n = node || this;
            if (v === "show") showOver(n, {});
            else hideOver(n);
            reflow(n);
          }, { values: ["hide", "show"] });

          // Start with all override widgets hidden
          hideOver(this);
        };

        var origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
          if (origConfigure) origConfigure.call(this, data);
          var saved = data && data._teacache_overrides;
          if (saved && Object.keys(saved).length > 0) {
            var overW = getWidget(this, "overrides");
            var overrideVal = saved.overrides !== undefined ? saved.overrides : "hide";
            if (overW) {
              overW.value = overrideVal;
              if (overrideVal === "show") showOver(this, saved);
              else hideOver(this);
            } else {
              hideOver(this);
            }
            reflow(this);
          }
        };

        var origSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
          if (origSerialize) origSerialize.call(this, data);
          var saved = {};
          var ws = this.widgets || [];
          for (var i = 0; i < ws.length; i++) {
            if (ALL_NAMES.has(ws[i].name))
              saved[ws[i].name] = ws[i].value;
          }
          if (Object.keys(saved).length > 0) data._teacache_overrides = saved;
        };
      },
    });

    console.log("TeaCacheAnima: override extension loaded (hidden-based)");
  }

  register();
})();
