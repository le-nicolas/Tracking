(function () {
  const STORAGE_KEY = "tracking_entries_v1";

  const state = {
    entries: loadEntries(),
    filterCategory: "all",
    search: "",
  };

  const entryForm = document.getElementById("entryForm");
  const dateInput = document.getElementById("dateInput");
  const categoryInput = document.getElementById("categoryInput");
  const labelInput = document.getElementById("labelInput");
  const valueInput = document.getElementById("valueInput");
  const unitInput = document.getElementById("unitInput");
  const notesInput = document.getElementById("notesInput");
  const filterCategory = document.getElementById("filterCategory");
  const searchInput = document.getElementById("searchInput");
  const clearAllBtn = document.getElementById("clearAllBtn");
  const exportCsvBtn = document.getElementById("exportCsvBtn");
  const entryTableBody = document.getElementById("entryTableBody");
  const emptyState = document.getElementById("emptyState");

  const entriesStat = document.getElementById("entriesStat");
  const valueStat = document.getElementById("valueStat");
  const categoryStat = document.getElementById("categoryStat");
  const streakStat = document.getElementById("streakStat");

  init();

  function init() {
    dateInput.valueAsDate = new Date();
    entryForm.addEventListener("submit", onSubmit);
    filterCategory.addEventListener("change", onFilterChange);
    searchInput.addEventListener("input", onSearchInput);
    clearAllBtn.addEventListener("click", onClearAll);
    exportCsvBtn.addEventListener("click", onExportCsv);
    entryTableBody.addEventListener("click", onTableAction);
    render();
  }

  function onSubmit(event) {
    event.preventDefault();

    const value = Number(valueInput.value);
    const label = labelInput.value.trim();

    if (!label || Number.isNaN(value) || value < 0) {
      return;
    }

    const entry = {
      id: makeId(),
      date: dateInput.value,
      category: categoryInput.value,
      label: label,
      value: value,
      unit: unitInput.value.trim(),
      notes: notesInput.value.trim(),
      createdAt: new Date().toISOString(),
    };

    state.entries.push(entry);
    state.entries.sort(compareByDateDesc);
    persist();
    entryForm.reset();
    dateInput.valueAsDate = new Date();
    render();
  }

  function onFilterChange(event) {
    state.filterCategory = event.target.value;
    render();
  }

  function onSearchInput(event) {
    state.search = event.target.value.trim().toLowerCase();
    render();
  }

  function onClearAll() {
    if (!state.entries.length) {
      return;
    }
    if (!window.confirm("Delete all tracked entries? This cannot be undone.")) {
      return;
    }
    state.entries = [];
    persist();
    render();
  }

  function onExportCsv() {
    if (!state.entries.length) {
      return;
    }

    const rows = [
      ["id", "date", "category", "label", "value", "unit", "notes", "createdAt"],
      ...state.entries.map((entry) => [
        entry.id,
        entry.date,
        entry.category,
        entry.label,
        String(entry.value),
        entry.unit,
        entry.notes,
        entry.createdAt,
      ]),
    ];

    const csv = rows.map((row) => row.map(escapeCsvCell).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "tracking-export.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function onTableAction(event) {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    if (target.dataset.action !== "delete") {
      return;
    }

    const id = target.dataset.id;
    state.entries = state.entries.filter((entry) => entry.id !== id);
    persist();
    render();
  }

  function filteredEntries() {
    return state.entries.filter((entry) => {
      const matchesCategory =
        state.filterCategory === "all" || entry.category === state.filterCategory;
      const haystack = `${entry.label} ${entry.notes}`.toLowerCase();
      const matchesSearch = !state.search || haystack.includes(state.search);
      return matchesCategory && matchesSearch;
    });
  }

  function render() {
    const visible = filteredEntries();
    renderStats(state.entries);
    renderTable(visible);
  }

  function renderStats(entries) {
    entriesStat.textContent = String(entries.length);

    const total = entries.reduce((sum, entry) => sum + Number(entry.value || 0), 0);
    valueStat.textContent = compactNumber(total);

    const frequency = {};
    for (const entry of entries) {
      frequency[entry.category] = (frequency[entry.category] || 0) + 1;
    }

    let topCategory = "-";
    let maxCount = 0;
    for (const [name, count] of Object.entries(frequency)) {
      if (count > maxCount) {
        topCategory = name;
        maxCount = count;
      }
    }
    categoryStat.textContent = topCategory;

    streakStat.textContent = `${calculateStreak(entries)} days`;
  }

  function renderTable(entries) {
    entryTableBody.innerHTML = "";

    if (!entries.length) {
      emptyState.hidden = false;
      return;
    }

    emptyState.hidden = true;

    const fragment = document.createDocumentFragment();
    for (const entry of entries.sort(compareByDateDesc)) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${escapeHtml(entry.date)}</td>
        <td>${escapeHtml(entry.category)}</td>
        <td>${escapeHtml(entry.label)}</td>
        <td>${escapeHtml(formatValue(entry.value, entry.unit))}</td>
        <td>${escapeHtml(entry.notes || "-")}</td>
        <td>
          <button class="row-delete" data-action="delete" data-id="${escapeHtml(entry.id)}" type="button">
            Delete
          </button>
        </td>
      `;
      fragment.appendChild(row);
    }

    entryTableBody.appendChild(fragment);
  }

  function calculateStreak(entries) {
    if (!entries.length) {
      return 0;
    }

    const uniqueDates = [...new Set(entries.map((entry) => entry.date))]
      .filter(Boolean)
      .sort((a, b) => (a > b ? -1 : 1));

    if (!uniqueDates.length) {
      return 0;
    }

    let streak = 1;
    let cursor = new Date(uniqueDates[0] + "T00:00:00");

    for (let i = 1; i < uniqueDates.length; i += 1) {
      const nextDate = new Date(uniqueDates[i] + "T00:00:00");
      const dayDiff = Math.round((cursor.getTime() - nextDate.getTime()) / 86400000);
      if (dayDiff === 1) {
        streak += 1;
        cursor = nextDate;
      } else {
        break;
      }
    }

    return streak;
  }

  function loadEntries() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  function persist() {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state.entries));
  }

  function makeId() {
    if (window.crypto && window.crypto.randomUUID) {
      return window.crypto.randomUUID();
    }
    return `entry_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
  }

  function compareByDateDesc(a, b) {
    if (a.date === b.date) {
      return b.createdAt.localeCompare(a.createdAt);
    }
    return b.date.localeCompare(a.date);
  }

  function compactNumber(value) {
    return new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 2,
    }).format(value);
  }

  function formatValue(value, unit) {
    const formatted = compactNumber(Number(value || 0));
    return unit ? `${formatted} ${unit}` : formatted;
  }

  function escapeCsvCell(value) {
    const raw = String(value ?? "");
    if (/[",\n]/.test(raw)) {
      return `"${raw.replace(/"/g, "\"\"")}"`;
    }
    return raw;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
})();
