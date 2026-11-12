(function () {
  // Chart.js loads in the dashboard's extra_js block, after this file,
  // so wait for DOMContentLoaded: by then every script has executed.
  const initChart = () => {
  const el = document.getElementById("importanceChart");
  if (!el) return;

  const data = window.__IMPORTANCE__;
  if (!data || !data.labels || !data.values) return;

  if (typeof Chart === "undefined") {
    // The Chart.js CDN failed (offline, blocked): without this guard
    // the ReferenceError below leaves a blank card with no message.
    const note = document.createElement("div");
    note.className = "alert alert-warning mb-0";
    note.textContent = "Chart library could not be loaded — check your connection and reload.";
    el.replaceWith(note);
    return;
  }

  new Chart(el, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [{
        label: "Importance",
        data: data.values
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: { beginAtZero: true }
      }
    }
  });
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initChart);
  } else {
    initChart();
  }
})();

(function () {
  // The server keeps only the first two selected types, so a third
  // ticked box used to be silently dropped. Prevent selecting a third.
  const boxes = document.querySelectorAll('input[type="checkbox"][name="types"]');
  if (!boxes.length) return;
  boxes.forEach((box) => {
    box.addEventListener("change", () => {
      const checked = document.querySelectorAll('input[type="checkbox"][name="types"]:checked');
      if (checked.length > 2) box.checked = false;
    });
  });
})();
(function () {
  // Total is the sum of the six base stats by definition, but the form
  // used to accept any Total, so a self-contradictory vector (e.g.
  // Total 900 with stats summing to 300) reached the model. Auto-fill
  // Total from the stats until the user edits Total by hand.
  const total = document.getElementById("Total");
  if (!total) return;
  const stats = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]
    .map((name) => document.getElementById(name))
    .filter(Boolean);
  if (stats.length !== 6) return;
  let manualTotal = false;
  total.addEventListener("input", () => { manualTotal = true; });
  const syncTotal = () => {
    if (manualTotal) return;
    total.value = stats.reduce((sum, el) => sum + (Number(el.value) || 0), 0);
  };
  stats.forEach((el) => el.addEventListener("input", syncTotal));
  syncTotal();
})();
