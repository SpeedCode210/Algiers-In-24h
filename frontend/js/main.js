/* ─── DATA ─── */

var CAT_COLORS = ['#c47a2e', '#7eb8f7', '#226d68', '#78adb5', '#a78bfa'];

var MAP_STYLES = { dark: 'mapbox://styles/mapbox/navigation-night-v1', light: 'mapbox://styles/mapbox/streets-v12' };

const MAPBOX_API_KEY = "pk.eyJ1IjoibW9oYW1lZGVnaGJvdWRqIiwiYSI6ImNtb24ybnZsaTBrOXMycHF5MGc3d2dmcXAifQ.0QWaEMordbNjiOwJD7ZqMA";
const API_LINK = "http://127.0.0.1:5000/api/";

const SOLVERS = new Map([
  ["greedy", ["Greedy (Score Priority)", "Piplup", "Prioritizes and visits landmarks with the highest immediate interest score within your schedule."]],
  ["greedy_ratio", ["Greedy (Score/Time Ratio)", "Bunnelby", "Optimizes efficiency by selecting locations that offer the maximum score for the least amount of travel and visit time."]],
  ["greedy_nearest", ["Greedy (Nearest Neighbor)", "Shroomish", "Always moves to the closest unvisited landmark to drastically cut down driving distances."]],
  ["greedy_random", ["Greedy (Random)", "Froakie", "Keeps adding landmarks randomly until filling the time window. Suitable for users who wanna be surprised and try unexpected experiences."]],
  ["sa", ["Simulated Annealing", "Tinkaton", "A metaheuristic that avoids getting stuck in local routing loops by occasionally accepting worse paths early on, refining into a near-optimal tour."]],
  ["grasp", ["GRASP", "Rayquaza", "Combines randomized greedy start phases with local search enhancements to repeatedly construct high-quality alternatives."]],
  ["tabu", ["Tabu Search", "Mewtwo", "Uses a mathematical memory log to explicitly ban recent route modifications, forcing exploration of unexplored, highly efficient tour variations."]],
  ["ga", ["Genetic Algorithm", "Eevee", "Simulates biological evolution over hundreds of generations by pairing up and mutating routes to find hidden optimization paths."]],
  ["ga_tailored", ["Tailored Genetic Algorithm", "Jolteon", "An advanced structural mutation build customized specifically for the unique terrain layout and categories of Algiers."]],
  ["cplex", ["CPLEX (Exact)", "Magnemite", "Solves the tour mathematically down to absolute global perfection. Yields the definitive highest score, but computation scales sharply with stops."]]
]);

/* ─── STATE ─── */
var map, popup, hotelMarker;
var markers = {};
var selectedMarkerId = null;
var routeLayerIds = [];
var selectedRouteIds = {};
var activeCategories = {};
var currentTheme = 'dark';
var selectedAlgo = 'greedy';
var timeLimitEnabled = false;
var timeLimitMs = 1000;
var lastRoute = [], lastHotel = null, lastGeometry = null;
var routeHistory = []; // ranking history
var runningOptimizer = false;

/* Event Listeners */
document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
document.getElementById('optimize-tab').addEventListener('click', () => switchTab('optimize'));
document.getElementById('ranking-tab').addEventListener('click', () => switchTab('ranking'));
document.getElementById('clear-all-rankings').addEventListener('click', clearAllRankings);
document.getElementById('tour-guide-btn').addEventListener('click', openTourGuide);
document.getElementById('lm-drawer-close').addEventListener('click', closeLmDrawer);
document.getElementById('tg-close-btn').addEventListener('click', closeTourGuide);

/* LOAD INITIAL DATA */
const { hotels } = await fetch(API_LINK + 'hotels').then(r => r.json());
const { landmarks } = await fetch(API_LINK + 'landmarks').then(r => r.json());
const { categories } = await fetch(API_LINK + 'categories').then(r => r.json());

const categoryMap = categories.reduce((map, c, index) => {
  c.color = CAT_COLORS[index];
  map[c.id] = c;
  return map;
}, {});

var categoryOrder = Object.keys(categoryMap); // order array
Object.keys(categoryMap).forEach(function (k) { activeCategories[k] = true; });

/* ─── TABS ─── */
function switchTab(name) {
  document.querySelectorAll('.ptab').forEach(function (t) { t.classList.remove('active'); });
  document.querySelectorAll('.tab-pane').forEach(function (p) { p.classList.remove('active'); });
  document.getElementById(name + '-tab').classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'ranking') renderRankingList();
}

/* ─── TOAST ─── */
function showToast(msg, duration) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(function () { t.classList.remove('show'); }, duration || 2200);
}

/* ─── MAP ─── */
loadMapbox();
function loadMapbox() {
  var token = MAPBOX_API_KEY;
  var errEl = document.getElementById('ts-err');
  if (!token.startsWith('pk.')) { errEl.textContent = 'Token must start with "pk." — get one free at mapbox.com'; return; }
  errEl.textContent = 'Loading Mapbox…';
  var link = document.createElement('link');
  link.rel = 'stylesheet'; link.href = 'https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css';
  document.head.appendChild(link);
  var script = document.createElement('script');
  script.src = 'https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js';
  script.onload = function () {
    mapboxgl.accessToken = token;
    document.getElementById('token-screen').style.display = 'none';
    document.getElementById('panel').style.display = 'flex';
    var ptBtn = document.getElementById('panel-toggle');
    ptBtn.classList.add('visible');
    ptBtn.classList.add('panel-open');
    ptBtn.innerHTML = '&#9664;';
    document.getElementById('stats').style.display = 'flex';
    document.getElementById('algo-indicator').style.display = 'block';
    document.getElementById('theme-toggle').classList.add('visible');
    injectPopupStyles();
    initMap();
  };
  script.onerror = function () { errEl.textContent = 'Failed to load Mapbox. Check your connection.'; };
  document.head.appendChild(script);
}

/* ─── PANEL TOGGLE ─── */
var panelVisible = true;
document.getElementById('panel-toggle').addEventListener('click', togglePanel);
function togglePanel() {
  panelVisible = !panelVisible;
  var panel = document.getElementById('panel');
  var btn = document.getElementById('panel-toggle');
  if (panelVisible) {
    panel.classList.remove('collapsed');
    btn.classList.add('panel-open');
    btn.innerHTML = '&#9664;';
    document.body.classList.remove('panel-collapsed');
    document.getElementById('stats').style.left = '410px';
    document.getElementById('algo-indicator').style.left = '410px';
    document.getElementById('lm-drawer').style.left = '410px';
  } else {
    panel.classList.add('collapsed');
    btn.classList.remove('panel-open');
    btn.innerHTML = '&#9654;';
    document.body.classList.add('panel-collapsed');
    document.getElementById('stats').style.left = '20px';
    document.getElementById('algo-indicator').style.left = '20px';
    document.getElementById('lm-drawer').style.left = '20px';
  }
}

/* ─── POPUP STYLES ─── */
var popupStyleEl = null;
function injectPopupStyles() {
  popupStyleEl = document.createElement('style');
  document.head.appendChild(popupStyleEl);
  updatePopupStyles();
}
function updatePopupStyles() {
  if (!popupStyleEl) return;
  var isDark = currentTheme === 'dark';
  var bg = isDark ? 'rgba(8,12,20,.96)' : 'rgba(195,160,108,.97)';
  var border = isDark ? 'rgba(255,255,255,.12)' : 'rgba(101,67,33,.20)';
  var textMain = isDark ? '#f0ece4' : '#2c1e0f';
  var textMuted = isDark ? 'rgba(240,236,228,.52)' : 'rgba(44,30,15,.52)';
  var nameColor = isDark ? '#e8c96a' : '#7a4e10';
  popupStyleEl.textContent =
    '.mapboxgl-popup-content{background:' + bg + '!important;backdrop-filter:blur(20px)!important;border:1px solid ' + border + '!important;border-radius:12px!important;padding:14px 16px!important;color:' + textMain + '!important;font-family:\'DM Sans\',sans-serif!important;min-width:200px;box-shadow:0 8px 40px rgba(0,0,0,.3)!important;}' +
    '.mapboxgl-popup-tip{display:none!important;}' +
    '.mapboxgl-popup-close-button{color:' + textMuted + '!important;font-size:18px!important;right:8px!important;top:6px!important;background:none!important;}' +
    '.popup-name{font-family:\'Cinzel\',serif;font-size:13px;font-weight:600;color:' + nameColor + ';margin-bottom:8px;line-height:1.4;padding-right:16px;}' +
    '.popup-row{font-size:11px;color:' + textMuted + ';margin-top:4px;}' +
    '.popup-row strong{color:' + textMain + ';}';
}

/* ─── THEME ─── */
function toggleTheme() {
  currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.body.setAttribute('data-theme', currentTheme);
  updatePopupStyles();
  if (map) {
    map.setStyle(MAP_STYLES[currentTheme]);
    map.once('styledata', function () { routeLayerIds = []; if (lastRoute.length && lastHotel) drawRoute(lastRoute, lastHotel, lastGeometry); });
  }
}

/* ─── MAP INIT ─── */
function fmtHour(h) {
  var hh = Math.floor(h), mm = (h % 1 === 0.5) ? '30' : '00';
  return (hh < 10 ? '0' : '') + hh + ':' + mm;
}
function initMap() {
  map = new mapboxgl.Map({
    container: 'map', style: MAP_STYLES[currentTheme],
    center: [3.058, 36.765], zoom: 12.2, pitch: 45, bearing: -8, antialias: true
  });
  map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), 'top-right');
  popup = new mapboxgl.Popup({ closeOnClick: true, maxWidth: '300px', offset: 14 });
  map.on('click', function (e) {
    // close drawer if clicking empty map
    if (!e.originalEvent.target.closest('#lm-drawer') && !e.originalEvent.target.classList.contains('mk-dot')) {
      closeLmDrawer();
    }
  });
  map.on('load', function () {
    setupHotelSelector();
    setupCategoryFilter();
    setupAlgoButtons();
    renderMarkers();
    renderList();
  });
  /* ─── BUDGET RANGE SLIDER ─── */
  function updateRangeSlider() {
    var s = parseFloat(document.getElementById('budget-start').value);
    var e = parseFloat(document.getElementById('budget-end').value);
    if (s >= e - 0.5) {
      if (document.activeElement === document.getElementById('budget-start')) s = e - 0.5;
      else e = s + 0.5;
      document.getElementById('budget-start').value = s;
      document.getElementById('budget-end').value = e;
    }
    var pct1 = (s / 24) * 100;
    var pct2 = (e / 24) * 100;
    document.getElementById('range-fill').style.left = pct1 + '%';
    document.getElementById('range-fill').style.width = (pct2 - pct1) + '%';
    document.getElementById('budget-val').textContent = fmtHour(s) + ' – ' + fmtHour(e);
  }
  document.getElementById('budget-start').addEventListener('input', updateRangeSlider);
  document.getElementById('budget-end').addEventListener('input', updateRangeSlider);
  updateRangeSlider();
  document.getElementById('optimize-btn').addEventListener('click', runOptimizer);
}

/* ─── HOTEL ─── */
function setupHotelSelector() {
  var sel = document.getElementById('hotel-select');
  hotels.forEach(function (h, i) { var o = document.createElement('option'); o.value = i; o.textContent = h.name; sel.appendChild(o); });
  sel.addEventListener('change', placeHotelMarker);
  placeHotelMarker();
}
function placeHotelMarker() {
  var hotel = hotels[+document.getElementById('hotel-select').value];
  if (hotelMarker) hotelMarker.remove();
  var wrap = document.createElement('div');
  wrap.style.cssText = 'width:28px;height:36px;display:flex;align-items:flex-start;justify-content:center;cursor:pointer;';
  var dot = document.createElement('div');
  dot.style.cssText = 'width:22px;height:22px;background:#fff;border:2.5px solid #c9a84c;border-radius:50% 50% 50% 0;transform:rotate(-45deg);box-shadow:0 0 0 4px rgba(196,122,46,.25),0 4px 12px rgba(0,0,0,.5);';
  var inner = document.createElement('div');
  inner.style.cssText = 'position:absolute;width:8px;height:8px;background:#c9a84c;border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);';
  dot.style.position = 'relative';
  dot.appendChild(inner);
  wrap.appendChild(dot);
  hotelMarker = new mapboxgl.Marker({ element: wrap, anchor: 'bottom' })
    .setLngLat([hotel.longitude, hotel.latitude])
    .setPopup(new mapboxgl.Popup({ offset: 14 }).setHTML('<div class="popup-name">' + hotel.name + '</div><div class="popup-row">Starting hotel</div>'))
    .addTo(map);
}

/* ─── ALGO BUTTONS & HOVER TOOLTIPS ─── */
function setupAlgoButtons() {
  let algosContainer = document.getElementsByClassName('algo-row')[0];
  let flag = true;
  
  // Render buttons
  SOLVERS.forEach((value, key, map) => {
    algosContainer.innerHTML += `<button class="algo-btn${flag ? ' active' : ''}" data-algo="${key}">${value[1]}</button>`;
    flag = false;
  });

  const tooltip = document.getElementById('solver-tooltip');
  const tlImg = document.getElementById('tl-img');
  const tlName = document.getElementById('tl-name');
  const tlAlgo = document.getElementById('tl-algo');
  const tlDesc = document.getElementById('tl-desc');

  // Attach event handlers
  document.querySelectorAll('.algo-btn').forEach(function (btn) {
    // Click active behavior
    btn.addEventListener('click', function () {
      document.querySelectorAll('.algo-btn').forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      selectedAlgo = btn.dataset.algo;
    });

    // Hover Enter
    btn.addEventListener('mouseenter', function () {
      const algoKey = btn.dataset.algo;
      const data = SOLVERS.get(algoKey);
      
      if (data) {
        const [algoFullName, mascotName, description] = data;
        
        // Populate Tooltip Data
        tlName.textContent = mascotName;
        tlAlgo.textContent = algoFullName;
        tlDesc.textContent = description;
        tlImg.src = `img/${mascotName}.png`; // Accesses files from your existing ./img directory
        
        // Show tooltip card
        tooltip.classList.add('visible');
      }
    });

    // Hover Exit
    btn.addEventListener('mouseleave', function () {
      tooltip.classList.remove('visible');
    });
  });
}

/* ─── TIME LIMIT ─── */
function toggleTimeLimit() {
  timeLimitEnabled = !timeLimitEnabled;
  timeLimitMs = 10000; // max 10s
  var tog = document.getElementById('tl-toggle');
  var lbl = document.getElementById('tl-label');
  if (timeLimitEnabled) { tog.classList.add('on'); lbl.classList.add('active'); }
  else { tog.classList.remove('on'); lbl.classList.remove('active'); }
}

/* ─── CATEGORY FILTER — DRAGGABLE ─── */
function setupCategoryFilter() {
  renderCategoryRows();
}
function renderCategoryRows() {
  var cont = document.getElementById('cat-filter');
  cont.innerHTML = '';
  categoryOrder.forEach(function (cat, idx) {
    var color = categoryMap[cat].color;
    var row = document.createElement('div');
    row.className = 'cat-row';
    row.draggable = true;
    row.dataset.cat = cat;

    var priorityLabels = ['1st', '2nd', '3rd', '4th', '5th'];
    row.innerHTML =
      '<span class="cat-drag-handle">⠿</span>' +
      '<span class="cat-order-num">' + (idx + 1) + '</span>' +
      '<span class="cat-chip-inline" data-cat="' + cat + '" style="border-color:' + color + ';color:' + color + ';background:' + color + (activeCategories[cat] ? '33' : '10') + ';">' + categoryMap[cat].label + '</span>' +
      '<span class="cat-priority-badge">' + priorityLabels[idx] + '</span>';

    // Chip click = toggle
    row.querySelector('.cat-chip-inline').addEventListener('click', function (e) {
      e.stopPropagation();
      if (activeCategories[cat]) {
        delete activeCategories[cat];
        this.style.background = color + '10'; this.style.opacity = '0.5';
      } else {
        activeCategories[cat] = true;
        this.style.background = color + '33'; this.style.opacity = '1';
      }
      Object.keys(markers).forEach(function (id) {
        var m = markers[id];
        m.wrap.style.display = activeCategories[m.lm.category] ? 'flex' : 'none';
      });
      renderList();
    });
    if (!activeCategories[cat]) { row.querySelector('.cat-chip-inline').style.opacity = '0.5'; }

    // Drag events
    row.addEventListener('dragstart', function (e) { e.dataTransfer.setData('text/plain', cat); row.classList.add('dragging'); });
    row.addEventListener('dragend', function () { row.classList.remove('dragging'); cont.querySelectorAll('.cat-row').forEach(function (r) { r.classList.remove('drag-over'); }); });
    row.addEventListener('dragover', function (e) { e.preventDefault(); row.classList.add('drag-over'); });
    row.addEventListener('dragleave', function () { row.classList.remove('drag-over'); });
    row.addEventListener('drop', function (e) {
      e.preventDefault();
      var dragCat = e.dataTransfer.getData('text/plain');
      var fromIdx = categoryOrder.indexOf(dragCat);
      var toIdx = categoryOrder.indexOf(cat);
      if (fromIdx !== toIdx) {
        categoryOrder.splice(fromIdx, 1);
        categoryOrder.splice(toIdx, 0, dragCat);
        renderCategoryRows();
        renderList();
      }
    });

    cont.appendChild(row);
  });
}

/* ─── MARKERS ─── */
function renderMarkers() {
  landmarks.forEach(function (lm) {
    var color = categoryMap[lm.category].color || '#888';
    var size = Math.round(2 + lm.interest_score * 1.5);
    var padded = size + 12;
    var wrap = document.createElement('div');
    wrap.style.cssText = 'width:' + padded + 'px;height:' + padded + 'px;display:flex;align-items:center;justify-content:center;cursor:pointer;';
    var dot = document.createElement('div');
    dot.className = 'mk-dot';
    dot.style.cssText = 'width:' + size + 'px;height:' + size + 'px;background:' + color + ';border-radius:50%;border:1.5px solid rgba(255,255,255,0.35);box-shadow:0 0 ' + size + 'px ' + color + '66;transition:transform .18s cubic-bezier(.34,1.56,.64,1),box-shadow .18s;';
    wrap.appendChild(dot);
    wrap.addEventListener('mouseenter', function () { dot.style.transform = 'scale(2)'; dot.style.boxShadow = '0 0 ' + (size + 8) + 'px ' + color; });
    wrap.addEventListener('mouseleave', function () { dot.style.transform = 'scale(1)'; dot.style.boxShadow = '0 0 ' + size + 'px ' + color + '66'; });
    wrap.addEventListener('click', function () { showPopup(lm); });
    new mapboxgl.Marker({ element: wrap, anchor: 'center' }).setLngLat([lm.longitude, lm.latitude]).addTo(map);
    markers[lm.id] = { wrap: wrap, dot: dot, lm: lm };
  });
}

/* ─── LANDMARK DRAWER ─── */
function openLmDrawer(lm) {
  var color = categoryMap[lm.category].color || '#888';
  document.getElementById('lm-stripe').style.background = color;
  document.getElementById('lm-eyebrow').textContent = lm.category;
  document.getElementById('lm-name').textContent = lm.name;
  document.getElementById('lm-score').textContent = lm.interest_score;
  document.getElementById('lm-rating').textContent = '★ ' + Math.round(lm.interest_score / 2);
  document.getElementById('lm-dur').textContent = Math.floor(lm.visit_duration_minutes / 60) + 'h' + (lm.visit_duration_minutes % 60 < 10 ? '0' : '') + (lm.visit_duration_minutes % 60);

  var hotel = lastHotel || hotels[+document.getElementById('hotel-select').value];
  document.getElementById('lm-dist').textContent = haversine(hotel, lm).toFixed(1);
  var badge = document.getElementById('lm-route-badge');
  badge.innerHTML = '';
  if (selectedRouteIds[lm.id]) {
    var rIdx = 0;
    for (var i = 0; i < lastRoute.length; i++) { if (lastRoute[i].id === lm.id) { rIdx = i + 1; break; } }
    badge.innerHTML = '<div class="lm-in-route-badge"><span class="lm-route-num">' + rIdx + '</span>&nbsp;Stop #' + rIdx + ' in current route</div>';
  }
  document.getElementById('lm-drawer').classList.add('open');
  var stats = document.getElementById('stats');
  var algo = document.getElementById('algo-indicator');
  if (stats) { stats.style.opacity = '0'; stats.style.pointerEvents = 'none'; }
  if (algo) { algo.style.opacity = '0'; algo.style.pointerEvents = 'none'; }
}
function closeLmDrawer() {
  document.getElementById('lm-drawer').classList.remove('open');
  var stats = document.getElementById('stats');
  var algo = document.getElementById('algo-indicator');
  if (stats) { stats.style.opacity = '1'; stats.style.pointerEvents = ''; }
  if (algo) { algo.style.opacity = '1'; algo.style.pointerEvents = ''; }
  // Reset selected marker
  if (selectedMarkerId !== null && markers[selectedMarkerId]) {
    var prev = markers[selectedMarkerId];
    var prevColor = categoryMap[prev.lm.category].color || '#888';
    var prevSize = Math.round(2 + prev.lm.score * 1.5);
    prev.dot.style.transform = 'scale(1)';
    prev.dot.style.boxShadow = '0 0 ' + prevSize + 'px ' + prevColor + '66';
    prev.dot.style.border = '1.5px solid rgba(255,255,255,0.35)';
  }
  selectedMarkerId = null;
}
function showPopup(lm) {
  map.flyTo({ center: [lm.longitude, lm.latitude], zoom: 15, pitch: 50, duration: 900 });
  // Reset previously selected marker
  if (selectedMarkerId !== null && markers[selectedMarkerId]) {
    var prev = markers[selectedMarkerId];
    var prevColor = categoryMap[prev.lm.category].color || '#888';
    var prevSize = Math.round(2 + prev.lm.interest_score * 1.5);
    prev.dot.style.transform = 'scale(1)';
    prev.dot.style.boxShadow = '0 0 ' + prevSize + 'px ' + prevColor + '66';
    prev.dot.style.border = '1.5px solid rgba(255,255,255,0.35)';
  }
  // Highlight newly selected marker
  selectedMarkerId = lm.id;
  if (markers[lm.id]) {
    var m = markers[lm.id];
    var color = '#888';
    var size = Math.round(2 + lm.score * 1.5);
    m.dot.style.transform = 'scale(2.8)';
    m.dot.style.boxShadow = '0 0 ' + (size * 2.5) + 'px ' + color + ', 0 0 ' + (size * 4) + 'px ' + color + '55';
    m.dot.style.border = '2px solid rgba(255,255,255,0.85)';
  }
  openLmDrawer(lm);
}
function renderList() { } // list removed — info shown via drawer on click

/* ─── OPTIMIZER ─── */
function haversine(a, b) {
  var R = 6371, r = Math.PI / 180;
  var dLat = (b.latitude - a.latitude) * r, dLng = (b.longitude - a.longitude) * r;
  var s = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos(a.latitude * r) * Math.cos(b.latitude * r) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
  return R * 2 * Math.atan2(Math.sqrt(s), Math.sqrt(1 - s));
}
function travelH(a, b) { return haversine(a, b) / 28; }



/* ── MAIN RUN ── */
async function runOptimizer() {
  if (runningOptimizer) return;

  var budgetStart = parseFloat(document.getElementById('budget-start').value);
  var budgetEnd = parseFloat(document.getElementById('budget-end').value);
  var budget = budgetEnd - budgetStart; // duration in hours
  var hotel = hotels[+document.getElementById('hotel-select').value];

  // Build category weights from active categories + priority order
  var categoryWeights = {};
  var numCats = categoryOrder.filter(c => activeCategories[c]).length;

  categoryOrder.forEach(function (cat, idx) {
    if (activeCategories[cat]) {
      // Map priority index to range [1.6 (highest) → 0.4 (lowest)]
      // e.g. 5 active cats → 1.6, 1.36, 1.12, 0.88, 0.64, 0.4
      var normalized = numCats > 1
        ? 1.6 - ((categoryOrder.indexOf(cat)) / (numCats - 1)) * 1.2
        : 1.0; // only one category → neutral weight
      categoryWeights[cat] = parseFloat(normalized.toFixed(2));
    }
  });


  runningOptimizer = true;
  document.getElementById('optimize-btn').disabled = true;
  document.getElementById('progress-bar-wrap').style.display = 'block';

  // SOLVERS is a Map: get([displayName, mascot])
  var solverEntry = SOLVERS.get(selectedAlgo);
  var algoName = solverEntry ? solverEntry[0] : selectedAlgo;

  document.getElementById('progress-label').textContent = 'Running ' + algoName + '…';
  document.getElementById('progress-fill').style.width = '20%';

  var t0 = Date.now();

  try {
    // Build request body
    var requestBody = {
      algorithm: selectedAlgo,
      hotel_id: hotel.id,
      time_budget: Math.round(budget * 60),
      tour_day: getTourDay(),   // implement as needed, e.g. "2025-01-01" or today
      start_time: fmtHour(budgetStart), // e.g. "09:00"
      category_weights: categoryWeights,
      algorithm_params: {}
    };

    if (timeLimitEnabled) {
      requestBody.algorithm_params.time_limit_ms = timeLimitMs;
    }

    var response = await fetch(API_LINK + 'solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      var errText = await response.text();
      throw new Error('API error ' + response.status + ': ' + errText);
    }

    var data = await response.json(); // shape matches your example response
    var elapsed = Date.now() - t0;

    // Map API stops back to local landmark objects (preserving local metadata)
    // Falls back to stop data itself if landmark not found locally
    var route = data.stops.map(function (stop) {
      var local = landmarks.find(function (l) { return l.id === stop.id; });
      return local ? Object.assign({}, local, stop) : stop;
    });

    // ── UI updates ────────────────────────────────────────────────────────────
    document.getElementById('progress-fill').style.width = '100%';
    document.getElementById('progress-label').textContent =
      'Done in ' + elapsed + 'ms · ' + route.length + ' stops';

    setTimeout(function () {
      document.getElementById('progress-bar-wrap').style.display = 'none';
      document.getElementById('progress-fill').style.width = '0%';
      runningOptimizer = false;
      document.getElementById('optimize-btn').disabled = false;
    }, 800);

    // Build selectedRouteIds lookup
    selectedRouteIds = {};
    route.forEach(function (l) { selectedRouteIds[l.id] = true; });

    // Prefer API-provided totals; fall back to local calculation
    var totalScore = data.total_score !== undefined
      ? data.total_score
      : route.reduce(function (s, l) { return s + (l.interest_score || l.score || 0); }, 0);

    var totalTime = data.total_duration_minutes !== undefined
      ? (data.total_duration_minutes / 60)
      : (function () {
        var t = 0, p = hotel;
        route.forEach(function (l) { t += travelH(p, l) + (l.dur || 0); p = l; });
        t += travelH(p, hotel);
        return t;
      })();

    var totalKm = data.total_distance_km !== undefined
      ? data.total_distance_km
      : (function () {
        var km = 0, p = hotel;
        route.forEach(function (l) { km += haversine(p, l); p = l; });
        km += haversine(p, hotel);
        return km;
      })();

    document.getElementById('s-count').textContent = route.length;
    var totalMins = Math.round(totalTime * 60);
    var hh = Math.floor(totalMins / 60);
    var mm = totalMins % 60;
    document.getElementById('s-time').textContent = hh + 'h ' + (mm ? mm + 'min' : '');
    document.getElementById('s-score').textContent = totalScore.toFixed(1);
    document.getElementById('s-km').textContent = totalKm.toFixed(1);

    document.getElementById('algo-ind-name').textContent = algoName;
    document.getElementById('algo-ind-time').textContent = elapsed + 'ms';

    renderList();
    lastRoute = route;
    lastHotel = hotel;
    lastGeometry = data.road_geometry;
    drawRoute(route, hotel, data.road_geometry);

    document.getElementById('tour-guide-btn').classList.add('visible');

    Object.keys(markers).forEach(function (id) {
      var m = markers[id];
      m.dot.style.opacity = selectedRouteIds[id] ? '1' : '0.2';
    });

    addToRanking({
      id: Date.now(),
      algo: algoName,
      algoKey: selectedAlgo,
      hotel: hotel.name,
      hotelObj: hotel,
      stops: route.length,
      score: parseFloat(totalScore.toFixed(1)),
      time: parseFloat(totalTime.toFixed(1)),
      km: parseFloat(totalKm.toFixed(1)),
      budget: budget,
      budgetRange: fmtHour(budgetStart) + '–' + fmtHour(budgetEnd),
      elapsed: elapsed,
      timeLimitUsed: timeLimitEnabled,
      route: route,
      timestamp: Date.now(),
      geometry: data.road_geometry
    });

  } catch (err) {
    var elapsed = Date.now() - t0;
    console.error('runOptimizer error:', err);

    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-label').textContent = 'Error: ' + err.message;

    setTimeout(function () {
      document.getElementById('progress-bar-wrap').style.display = 'none';
      document.getElementById('progress-fill').style.width = '0%';
      runningOptimizer = false;
      document.getElementById('optimize-btn').disabled = false;
    }, 2000);
  }
}

// ── Helper: get today as YYYY-MM-DD (or adapt to your date-picker) ──────────
function getTourDay() {
  var days = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  return days[new Date().getDay()];
}



/* ─── RANKING ─── */
function addToRanking(entry) {
  routeHistory.push(entry);
  routeHistory.sort(function (a, b) { return b.score - a.score; });
  showToast('Route saved to rankings (#' + (routeHistory.indexOf(entry) + 1) + ')');
  // If ranking tab is active, refresh
  if (document.getElementById('tab-ranking').classList.contains('active')) renderRankingList();
}

function deleteRanking(id) {
  routeHistory = routeHistory.filter(function (r) { return r.id !== id; });
  renderRankingList();
}

function clearAllRankings() {
  if (!routeHistory.length) return;
  routeHistory = [];
  renderRankingList();
  showToast('Rankings cleared');
}

function renderRankingList() {
  var cont = document.getElementById('ranking-list');
  cont.innerHTML = '';
  if (!routeHistory.length) {
    cont.innerHTML = '<div class="rank-empty"><div class="rank-empty-icon" style="font-size:36px;margin-bottom:12px;opacity:.4;">—</div>No routes yet.<br>Run an optimization to see rankings here.</div>';
    return;
  }
  var medals = ['🥇', '🥈', '🥉'];
  var medalColors = ['#c9a84c', '#9eaabb', '#a07850'];
  routeHistory.forEach(function (entry, idx) {
    var card = document.createElement('div');
    card.className = 'rank-card rank-' + (idx + 1);

    var timeAgo = getTimeAgo(entry.timestamp);
    var stopsHtml = entry.route.slice(0, 8).map(function (l) {
      return '<span class="rank-stop-chip" style="border-color:' + CAT_COLORS[l.cat] + '44;">' + l.name.split(' ')[0] + '…</span>';
    }).join('') + (entry.route.length > 8 ? '<span class="rank-stop-chip">+' + (entry.route.length - 8) + '</span>' : '');

    var acol = 'var(--gold)';
    var posColor = medalColors[idx] || 'var(--text-muted)';

    card.innerHTML =
      '<button class="rank-del-btn" title="Remove">✕</button>' +
      '<div class="rank-card-header">' +
      '<div class="rank-pos" style="color:' + posColor + ';border:1.5px solid ' + posColor + '44;border-radius:6px;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-size:13px;">' + (idx + 1) + '</div>' +
      '<div class="rank-info">' +
      '<div class="rank-algo" style="color:' + acol + '">' + entry.algo + '</div>' +
      '<div class="rank-hotel">' + entry.hotel + '</div>' +
      '</div>' +
      '<div class="rank-time-ago">' + timeAgo + '</div>' +
      '</div>' +
      '<div style="display:flex;align-items:baseline;gap:6px;margin-bottom:6px;">' +
      '<div class="rank-score-big">' + entry.score + '</div>' +
      '<div style="font-size:10px;color:var(--text-muted);">pts · ' + (entry.budgetRange || entry.budget + 'h') + (entry.timeLimitUsed ? ' · limit' : '') + '</div>' +
      '</div>' +
      '<div class="rank-stats-row">' +
      '<div class="rank-stat"><div class="rank-stat-val">' + entry.stops + '</div><div class="rank-stat-lbl">Stops</div></div>' +
      '<div class="rank-stat"><div class="rank-stat-val">' + entry.time + 'h</div><div class="rank-stat-lbl">Time</div></div>' +
      '<div class="rank-stat"><div class="rank-stat-val">' + entry.km + '</div><div class="rank-stat-lbl">km</div></div>' +
      '<div class="rank-stat"><div class="rank-stat-val">' + entry.elapsed + 'ms</div><div class="rank-stat-lbl">Computed</div></div>' +
      '</div>' +
      '<div class="rank-stops">' + stopsHtml + '</div>' +
      '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--glass-border);display:flex;justify-content:flex-end;">' +
      '<button class="rank-tg-btn" style="background:rgba(34,109,104,.1);border:1px solid rgba(34,109,104,.3);color:var(--teal);border-radius:8px;padding:4px 12px;font-size:10px;font-weight:600;letter-spacing:.5px;cursor:pointer;font-family:\'Outfit\',sans-serif;transition:all .15s;" title="Open Tour Guide">🗺 View Tour Guide</button>' +
      '</div>';

    card.querySelector('.rank-del-btn').addEventListener('click', function (e) {
      e.stopPropagation();
      deleteRanking(entry.id);
    });

    card.querySelector('.rank-tg-btn').addEventListener('click', function (e) {
      e.stopPropagation();
      lastRoute = entry.route; lastHotel = entry.hotelObj;
      document.getElementById('s-count').textContent = entry.stops;
      var totalMins = Math.round(entry.time * 60);
      var hh = Math.floor(totalMins / 60);
      var mm = totalMins % 60;
      document.getElementById('s-time').textContent = hh + 'h ' + (mm ? mm + 'min' : '');
      document.getElementById('s-score').textContent = entry.score;
      document.getElementById('s-km').textContent = entry.km;
      document.getElementById('algo-ind-name').textContent = entry.algo;
      document.getElementById('tour-guide-btn').classList.add('visible');
      openTourGuide();
    });

    // Click to restore route on map
    card.addEventListener('click', function () {
      selectedRouteIds = {};
      entry.route.forEach(function (l) { selectedRouteIds[l.id] = true; });
      lastRoute = entry.route; lastHotel = entry.hotelObj;
      drawRoute(entry.route, entry.hotelObj, entry.geometry);
      Object.keys(markers).forEach(function (id) {
        var m = markers[id]; m.dot.style.opacity = selectedRouteIds[+id] ? '1' : '0.2';
      });
      switchTab('optimize');
      renderList();
      document.getElementById('s-count').textContent = entry.stops;
      var totalMins = Math.round(entry.time * 60);
      var hh = Math.floor(totalMins / 60);
      var mm = totalMins % 60;
      document.getElementById('s-time').textContent = hh + 'h ' + (mm ? mm + 'min' : '');
      document.getElementById('s-score').textContent = entry.score;
      document.getElementById('s-km').textContent = entry.km;
      showToast('Route #' + (idx + 1) + ' restored on map');
      document.getElementById('tour-guide-btn').classList.add('visible');
    });

    cont.appendChild(card);
  });
}

function getTimeAgo(ts) {
  var diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60) return diff + 's ago';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  return Math.floor(diff / 3600) + 'h ago';
}

/* ─── TOUR GUIDE ─── */
var CAT_EMOJI = {
  'Historical': '🏛️', 'Religious': '🕌', 'Attraction': '🎡', 'Tradition&art': '🎨', 'Shopping': '🛍️'
};

function openTourGuide() {
  if (!lastRoute || !lastRoute.length) { showToast('Run an optimization first'); return; }
  var overlay = document.getElementById('tour-guide-overlay');
  overlay.classList.add('open');

  // Summary bar
  document.getElementById('tg-s-stops').textContent = document.getElementById('s-count').textContent;
  document.getElementById('tg-s-time').textContent = document.getElementById('s-time').textContent;
  document.getElementById('tg-s-score').textContent = document.getElementById('s-score').textContent;
  document.getElementById('tg-s-km').textContent = document.getElementById('s-km').textContent;
  document.getElementById('tg-s-algo').textContent = document.getElementById('algo-ind-name').textContent;

  var hotel = lastHotel;
  document.getElementById('tg-subtitle').textContent =
    'Starting from ' + (hotel ? hotel.name : 'your hotel') + ' · ' + lastRoute.length + ' stops';

  renderTourGuideStops(lastRoute, hotel);
}

function closeTourGuide() {
  document.getElementById('tour-guide-overlay').classList.remove('open');
}

function renderTourGuideStops(route, hotel) {
  var list = document.getElementById('tg-stops-list');
  list.innerHTML = '';

  // Hotel start chip
  if (hotel) {
    var startEl = document.createElement('div');
    startEl.className = 'tg-hotel-cap';
    startEl.innerHTML = '<span style="font-size:18px;">🏨</span><span><b>Start:</b> ' + hotel.name + '</span>';
    list.appendChild(startEl);
  }

  route.forEach(function (lm, idx) {
    var catColor = 'var(--gold)';
    var isLast = idx === route.length - 1;

    var card = document.createElement('div');
    card.className = 'tg-stop-card';
    card.style.animationDelay = (idx * 0.045) + 's';

    // Stars (filled using colored spans)
    var starsHtml = '';
    var ldm;
    landmarks.forEach(l=>{
      if(l.id == lm.id){
        ldm = l;
      }
    })
    var r = Math.round(ldm.interest_score/2);
    for (var s = 1; s <= 5; s++) {
      starsHtml += '<span style="color:' + (s <= Math.round(r) ? catColor : 'rgba(150,150,150,.3)') + ';font-size:12px;">★</span>';
    }

    card.innerHTML =
      '<div class="tg-stop-num-col">' +
      '<div class="tg-stop-num" style="color:' + catColor + ';border-color:' + catColor + '55;background:' + catColor + '18;">' + (idx + 1) + '</div>' +
      (!isLast ? '<div class="tg-connector" style="border-color:' + catColor + ';"></div>' : '') +
      '</div>' +
      '<div class="tg-stop-body">' +
      '<div class="tg-stop-cat" style="color:' + catColor + '">' + (categoryMap[lm.category].label.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;")) + '</div>' +
      '<div class="tg-stop-name">' + lm.name + '</div>' +
      '<div class="tg-stop-rating" style="margin-bottom:7px;">' + starsHtml + ' <span style="font-size:11px;color:var(--text-muted);margin-left:3px;">' + '</span></div>' +
      '<div class="tg-stop-grid">' +
      '<div class="tg-stop-chip" style="border-color:' + catColor + '44;"><b style="color:' + catColor + '">' + lm.interest_score + '</b> pts</div>' +
      '<div class="tg-stop-chip"><b>' + Math.floor(lm.visit_duration_minutes/60) + 'h' + (lm.visit_duration_minutes%60 < 10 ? '0' : '') + (lm.visit_duration_minutes%60) + 'min</b> visit</div>' +
      '</div>' +
      '<div class="tg-stop-hours">🕐 ' + Math.floor(lm.visit_duration_minutes/60) + 'h' + (lm.visit_duration_minutes%60 < 10 ? '0' : '') + (lm.visit_duration_minutes%60) + 'min</div>' +
      '</div>';

    list.appendChild(card);
  });

  // Hotel end chip
  if (hotel) {
    var endEl = document.createElement('div');
    endEl.className = 'tg-hotel-cap';
    endEl.innerHTML = '<span style="font-size:18px;">🏨</span><span><b>Return:</b> ' + hotel.name + '</span>';
    list.appendChild(endEl);
  }
}

// Close overlay on background click
document.getElementById('tour-guide-overlay').addEventListener('click', function (e) {
  if (e.target === this) closeTourGuide();
});

/* ─── DRAW ROUTE ─── */
function drawRoute(route, hotel, roadGeometry) {
  routeLayerIds.forEach(function (id) {
    try { if (map.getLayer(id)) map.removeLayer(id); } catch (e) { }
    try { if (map.getSource(id)) map.removeSource(id); } catch (e) { }
  });
  routeLayerIds = [];
  if (!route.length) return;

  // Use real road geometry from API if available, else fall back to straight lines
  var geo;
  if (roadGeometry && roadGeometry.coordinates && roadGeometry.coordinates.length) {
    geo = { type: 'Feature', geometry: roadGeometry };
  } else {
    var coords = [[hotel.longitude, hotel.latitude]]
      .concat(route.map(function (l) { return [l.longitude ?? l.lng, l.latitude ?? l.lat]; }))
      .concat([[hotel.longitude, hotel.latitude]]);
    geo = { type: 'Feature', geometry: { type: 'LineString', coordinates: coords } };
  }

  map.addSource('route', { type: 'geojson', data: geo });
  map.addLayer({
    id: 'route-glow', type: 'line', source: 'route',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: { 'line-color': '#4ecdc4', 'line-width': 22, 'line-opacity': 0.18, 'line-blur': 10 }
  });
  map.addLayer({
    id: 'route-line', type: 'line', source: 'route',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: { 'line-color': '#7ffff4', 'line-width': 4, 'line-opacity': 1 }
  });
  routeLayerIds = ['route-glow', 'route-line', 'route'];

  // Fit bounds around the geometry coordinates
  var allCoords = geo.geometry.coordinates;
  var bounds = new mapboxgl.LngLatBounds(allCoords[0], allCoords[0]);
  allCoords.forEach(function (c) { bounds.extend(c); });
  map.fitBounds(bounds, {
    padding: { top: 60, bottom: 100, left: 420, right: 40 },
    duration: 1100,
    pitch: 45
  });
}
