(function () {
  const el = document.getElementById("importanceChart");
  if (!el) return;

  const data = window.__IMPORTANCE__;
  if (!data || !data.labels || !data.values) return;

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
