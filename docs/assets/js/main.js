/* Build the gallery and small behaviors */
(function () {
  // Support grouped videos via window.videoGroups; fallback to flat window.videosData
  const groupsContainer = document.getElementById('video-groups');
  if (!groupsContainer) return;

  function renderCard(item) {
    const card = document.createElement('article');
    card.className = 'video-card';
    const header = document.createElement('header');
    header.textContent = item.title || item.file;
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
