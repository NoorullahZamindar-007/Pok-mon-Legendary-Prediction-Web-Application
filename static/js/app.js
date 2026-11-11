(function () {
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
