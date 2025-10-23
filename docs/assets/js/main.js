/* Build the gallery and small behaviors */
(function () {
  const grid = document.getElementById('video-grid');
  if (!grid) return;

  // videosData is injected from assets/data/videos.js
  const items = (window.videosData || []).map((v) => ({
    title: v.title || v.file,
    file: v.file,
    caption: v.caption || '',
  }));

  const frag = document.createDocumentFragment();
  items.forEach((item) => {
    const card = document.createElement('article');
    card.className = 'video-card';
    const header = document.createElement('header');
    header.textContent = item.title;
    const video = document.createElement('video');
    video.src = item.file;
    video.controls = true;
    video.preload = 'metadata';
    const footer = document.createElement('footer');
    footer.textContent = item.caption;
    card.appendChild(header);
    card.appendChild(video);
    card.appendChild(footer);
    frag.appendChild(card);
  });

  grid.appendChild(frag);
})();

