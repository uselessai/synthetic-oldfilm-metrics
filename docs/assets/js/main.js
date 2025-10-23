/* Build the gallery and small behaviors */
(function () {
  // Prefer comparative view if multiple model folders exist.
  const groupsContainer = document.getElementById('video-groups');
  if (!groupsContainer) return;

  function renderCard(item, modelLabel) {
    const card = document.createElement('article');
    card.className = 'video-card';
    const header = document.createElement('header');
    header.textContent = (modelLabel ? modelLabel + ' — ' : '') + (item.title || item.file);
    const footer = document.createElement('footer');
    footer.textContent = item.caption || '';

    const lower = (item.file || '').toLowerCase();
    const isInlinePlayable = /\.(mp4|webm|ogv)$/i.test(lower);
    if (isInlinePlayable) {
      const video = document.createElement('video');
      video.src = item.file;
      video.controls = true;
      video.preload = 'metadata';
      card.appendChild(header);
      card.appendChild(video);
      card.appendChild(footer);
    } else {
      const linkWrap = document.createElement('div');
      linkWrap.style.padding = '12px';
      const a = document.createElement('a');
      a.href = item.file;
      a.textContent = 'Open / download video';
      a.target = '_blank';
      a.rel = 'noopener';
      const hint = document.createElement('div');
      hint.style.color = 'var(--muted)';
      hint.style.fontSize = '13px';
      hint.style.marginTop = '6px';
      hint.textContent = 'Note: Some browsers do not play AVI inline.';
      linkWrap.appendChild(a);
      linkWrap.appendChild(hint);
      card.appendChild(header);
      card.appendChild(linkWrap);
      card.appendChild(footer);
    }
    return card;
  }

  function renderGroup(title, items) {
    const section = document.createElement('section');
    const h3 = document.createElement('h3');
    h3.textContent = title;
    const grid = document.createElement('div');
    grid.className = 'video-grid';
    const frag = document.createDocumentFragment();
    items.forEach((it) => frag.appendChild(renderCard(it)));
    grid.appendChild(frag);
    section.appendChild(h3);
    section.appendChild(grid);
    return section;
  }

  function renderComparative(data) {
    const container = document.createElement('section');
    const h3 = document.createElement('h3');
    h3.textContent = 'Model Comparison';
    container.appendChild(h3);

    const table = document.createElement('div');
    table.className = 'compare-table';

    // Header row
    const headerRow = document.createElement('div');
    headerRow.className = 'compare-row compare-header';
    const headFirst = document.createElement('div');
    headFirst.className = 'compare-cell compare-label';
    headFirst.textContent = 'Scene';
    headerRow.appendChild(headFirst);
    data.models.forEach((m) => {
      const c = document.createElement('div');
      c.className = 'compare-cell';
      c.textContent = m;
      headerRow.appendChild(c);
    });
    table.appendChild(headerRow);

    // Data rows
    data.rows.forEach((row) => {
      const rowEl = document.createElement('div');
      rowEl.className = 'compare-row';
      const label = document.createElement('div');
      label.className = 'compare-cell compare-label';
      label.textContent = row.name;
      rowEl.appendChild(label);
      data.models.forEach((m) => {
        const cell = document.createElement('div');
        cell.className = 'compare-cell';
        const file = row.items[m];
        if (file) {
          const item = { file: file, title: row.name };
          cell.appendChild(renderCard(item, m));
        } else {
          const miss = document.createElement('div');
          miss.className = 'missing';
          miss.textContent = '—';
          cell.appendChild(miss);
        }
        rowEl.appendChild(cell);
      });
      table.appendChild(rowEl);
    });

    container.appendChild(table);
    return container;
  }

  function buildComparativeFromGroups(groups) {
    const models = groups.map((g) => g.title);
    const byName = new Map(); // name -> { model -> file }
    const playableOrder = ['.mp4', '.webm', '.ogv', '.avi'];

    function stem(path) {
      try {
        const parts = path.split('/');
        const fn = parts[parts.length - 1];
        return fn.replace(/\.[^.]+$/, '');
      } catch (_) {
        return path;
      }
    }

    function chooseBest(files) {
      // files: array of file paths with different extensions; pick best by playableOrder
      let best = files[0];
      let bestRank = 999;
      files.forEach((f) => {
        const m = f.toLowerCase().match(/\.[^.]+$/);
        const ext = m ? m[0] : '';
        const rank = Math.max(0, playableOrder.indexOf(ext));
        if (rank !== -1 && rank < bestRank) { bestRank = rank; best = f; }
      });
      return best;
    }

    const tmpMap = {}; // name -> model -> [files]
    groups.forEach((g) => {
      g.items.forEach((it) => {
        const name = stem(it.file);
        tmpMap[name] = tmpMap[name] || {};
        tmpMap[name][g.title] = tmpMap[name][g.title] || [];
        tmpMap[name][g.title].push(it.file);
      });
    });

    Object.keys(tmpMap).sort().forEach((name) => {
      const modelFiles = {};
      models.forEach((m) => {
        const files = tmpMap[name][m];
        if (files && files.length) {
          modelFiles[m] = chooseBest(files);
        }
      });
      byName.set(name, modelFiles);
    });

    const rows = Array.from(byName.keys()).map((name) => ({ name, items: byName.get(name) || {} }));
    return { models, rows };
  }

  // 1) If precomputed comparisons exist, render them.
  if (window.videoComparisons && window.videoComparisons.models && window.videoComparisons.rows) {
    groupsContainer.appendChild(renderComparative(window.videoComparisons));
    return;
  }

  // 2) Else if we have groups, build comparative view automatically.
  if (Array.isArray(window.videoGroups) && window.videoGroups.length) {
    const data = buildComparativeFromGroups(window.videoGroups);
    groupsContainer.appendChild(renderComparative(data));
    return;
  }

  if (Array.isArray(window.videoGroups) && window.videoGroups.length) {
    const frag = document.createDocumentFragment();
    window.videoGroups.forEach((g) => {
      frag.appendChild(renderGroup(g.title || 'Videos', g.items || []));
    });
    groupsContainer.appendChild(frag);
    return;
  }

  // Fallback (legacy): single flat list
  const items = (window.videosData || []).map((v) => ({
    title: v.title || v.file,
    file: v.file,
    caption: v.caption || '',
  }));
  groupsContainer.appendChild(renderGroup('Videos', items));
})();
