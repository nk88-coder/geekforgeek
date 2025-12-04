// Client-side JS to POST form data to /predict and show the result
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('txnForm');
  const shapSection = document.getElementById('shapSection');
  const shapList = document.getElementById('shapList');
  const resultCard = document.getElementById('resultCard');
  const txInfoEl = document.getElementById('txInfo');
  const computedEl = document.getElementById('computedFeatures');
  const predEl = document.getElementById('predictionInfo');
  const rawJsonPre = document.getElementById('rawJson');
  const txnTableBody = document.querySelector('#txnTable tbody');
  const userIdInput = document.querySelector('input[name="user_id"]');
  const chartCanvas = document.getElementById('txnChart');
  let txnChart = null;

  function initChart() {
    const ctx = chartCanvas.getContext('2d');
    // create gradient for the line - bluish theme
    const grad = ctx.createLinearGradient(0, 0, 0, chartCanvas.height);
    grad.addColorStop(0, 'rgba(0,212,255,0.25)');
    grad.addColorStop(1, 'rgba(0,212,255,0)');

    txnChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Transaction Amount',
          data: [],
          fill: true,
          backgroundColor: grad,
          borderColor: '#00d4ff',
          tension: 0.2,
          pointRadius: 6,
          pointHoverRadius: 8,
          pointBackgroundColor: '#00d4ff'
        }]
      },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: { 
            mode: 'index', 
            intersect: false,
            backgroundColor: 'rgba(10,14,39,0.9)',
            borderColor: 'rgba(0,212,255,0.5)',
            borderWidth: 1,
            titleColor: '#00d4ff',
            bodyColor: '#c0cce0',
            padding: 10
          }
        },
        interaction: { intersect: false, mode: 'index' },
        scales: {
          x: { 
            display: true, 
            grid: { color: 'rgba(0,212,255,0.1)' }, 
            ticks: { color: '#a0adc0' }
          },
          y: { 
            beginAtZero: false, 
            grid: { color: 'rgba(0,212,255,0.1)' }, 
            ticks: { color: '#a0adc0' }
          }
        }
      }
    });
  }

  function updateChart(history) {
    if (!txnChart) initChart();
    const labels = history.map(h => h.timestamp || h.transaction_id || '');
    const data = history.map(h => Number(h.amount));
    // point colors: green if increased from prev, red if decreased
    const colors = [];
    for (let i = 0; i < data.length; i++) {
      if (i === 0) colors.push('#6b7280');
      else colors.push(data[i] >= data[i-1] ? '#16a34a' : '#ef4444');
    }
    txnChart.data.labels = labels;
    txnChart.data.datasets[0].data = data;
    txnChart.data.datasets[0].pointBackgroundColor = colors;
    txnChart.update();
  }

  function clearResult() {
    if (!txInfoEl || !computedEl || !predEl || !rawJsonPre) return;
    txInfoEl.innerHTML = '';
    computedEl.innerHTML = '';
    predEl.innerHTML = '';
    rawJsonPre.textContent = '';
  }

  function addKV(container, key, value) {
    const kv = document.createElement('div');
    kv.className = 'kv';
    const k = document.createElement('div'); k.className = 'k'; k.textContent = key;
    const v = document.createElement('div'); v.className = 'v'; v.textContent = value;
    kv.appendChild(k); kv.appendChild(v);
    container.appendChild(kv);
  }

  function buildResult(json) {
    clearResult();
    if (!json) return;
    if (txInfoEl.children.length === 0) {
      const h = document.createElement('h3'); h.textContent = 'Transaction Info'; h.style.margin = '6px 8px'; txInfoEl.appendChild(h);
    }
    if (json.transaction_id) addKV(txInfoEl, 'Transaction ID', json.transaction_id);
    if (json.user_id) addKV(txInfoEl, 'User ID', json.user_id);
    if (json.amount !== undefined) addKV(txInfoEl, 'Amount', Number(json.amount).toFixed(2));
    if (json.transaction_type) addKV(txInfoEl, 'Type', json.transaction_type);
    if (json.merchant_category) addKV(txInfoEl, 'Merchant', json.merchant_category);
    if (json.timestamp) addKV(txInfoEl, 'Timestamp', json.timestamp);

    if (computedEl.children.length === 0) { const h = document.createElement('h3'); h.textContent = 'Computed Features'; h.style.margin = '6px 8px'; computedEl.appendChild(h); }
    if (json.avg_last_3_txns !== undefined) addKV(computedEl, 'Avg (last 3)', Number(json.avg_last_3_txns).toFixed(2));
    if (json.avg_last_30_txns !== undefined) addKV(computedEl, 'Avg (last 30)', Number(json.avg_last_30_txns).toFixed(2));
    if (json.amount_deviation_score !== undefined) addKV(computedEl, 'Deviation score', Number(json.amount_deviation_score).toFixed(4));
    if (json.transactions_today !== undefined) addKV(computedEl, 'Transactions today', json.transactions_today);
    if (json.avg_daily_transactions_30d !== undefined) addKV(computedEl, 'Avg/day (30d)', Number(json.avg_daily_transactions_30d).toFixed(2));
    if (json.is_new_device !== undefined) addKV(computedEl, 'Is new device', json.is_new_device);
    if (json.location_change_flag !== undefined) addKV(computedEl, 'Location change', json.location_change_flag);
    if (json.feature_deviations && Array.isArray(json.feature_deviations)) {
      json.feature_deviations.forEach((f) => addKV(computedEl, f.feature, `z=${Number(f.z).toFixed(3)} v=${Number(f.value).toFixed(3)}`));
    }

    if (predEl.children.length === 0) { const h = document.createElement('h3'); h.textContent = 'Prediction'; h.style.margin = '6px 8px'; predEl.appendChild(h); }
    if (json.predicted_class !== undefined) {
      const classMap = { '0': 'No Fraud', '1': 'Fraud', '2': 'Risky' };
      const classLabel = classMap[String(json.predicted_class)] || json.predicted_class;
      addKV(predEl, 'Predicted class', classLabel);
    }
    if (Array.isArray(json.probabilities)) json.probabilities.forEach((p, i) => addKV(predEl, `Prob Class ${i}`, `${(p * 100).toFixed(2)}%`));
    if (json.shap && Array.isArray(json.shap) && Number(json.predicted_class) === 1) addKV(predEl, 'SHAP', json.shap.map(s => `${s.feature}=${Number(s.value).toFixed(3)}`).join(', '));

    rawJsonPre.textContent = JSON.stringify(json, null, 2);
  }

  async function fetchHistory(userId) {
    if (!userId) return;
    try {
      const res = await fetch(`/history?user_id=${encodeURIComponent(userId)}`);
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      if (Array.isArray(json.history)) {
        updateChart(json.history);
        // populate table with recent history (reverse chronological to keep most recent at top)
        txnTableBody.innerHTML = '';
        json.history.forEach((h) => {
          const tr = document.createElement('tr');
          const addCell = (txt) => { const td = document.createElement('td'); td.textContent = (txt !== undefined && txt !== null) ? String(txt) : ''; return td; };
          tr.appendChild(addCell(h.transaction_id));
          tr.appendChild(addCell(Number(h.amount).toFixed(2)));
          tr.appendChild(addCell(h.timestamp));
          tr.appendChild(addCell(h.transaction_type || ''));
          tr.appendChild(addCell(h.merchant_category || ''));
          tr.appendChild(addCell(h.is_new_device !== undefined ? String(h.is_new_device) : ''));
          tr.appendChild(addCell(h.predicted_class !== undefined ? String(h.predicted_class) : ''));
          txnTableBody.appendChild(tr);
        });
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearResult();
    rawJsonPre.textContent = 'Sending request...';

    const formData = new FormData(form);
    const payload = {};
    for (const [k, v] of formData.entries()) payload[k] = v;

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || res.statusText);
      }

      const json = await res.json();

      // Use the result card to present data in a clean way
      buildResult(json);
      // Refresh chart with the (new) history for this user
      try {
        const effectiveUserId = json.user_id || payload.user_id || userIdInput.value;
        if (effectiveUserId) await fetchHistory(effectiveUserId);
      } catch (err) {
        console.error(err);
      }

      // Show SHAP explanation only if returned and predicted_class == 1
      try {
        if (json.shap && Array.isArray(json.shap) && Number(json.predicted_class) === 1) {
          shapList.innerHTML = '';
          json.shap.forEach((s) => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${s.feature}</strong> ${Number(s.value).toFixed(4)}`;
            shapList.appendChild(li);
          });
          shapSection.style.display = 'block';
        } else {
          shapSection.style.display = 'none';
        }
      } catch (err) {
        console.error('Error rendering SHAP:', err);
        shapSection.style.display = 'none';
      }

      // feature deviations are handled above and included in the computed group
    } catch (err) {
      clearResult();
      rawJsonPre.textContent = 'Error: ' + err.message;
      console.error(err);
    }
  });

  // When page loads, initialize chart and fetch history for current user id (if provided)
  initChart();
  const initialUser = (userIdInput && userIdInput.value) ? userIdInput.value : 'user_1';
  if (userIdInput) userIdInput.value = userIdInput.value || 'user_1';
  fetchHistory(initialUser);

  // Fetch history when user id input changes
  userIdInput.addEventListener('change', (e) => fetchHistory(e.target.value));

  // No clear button - keep form behavior simple
});
