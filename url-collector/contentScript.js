// contentScript.js
(function() {
  let isRunning = false;
  let observer = null;

  // --- LOGIC NHẬN DIỆN (GIỮ NGUYÊN TỪ CODE CỦA BẠN VÌ RẤT TỐT) ---
  const URL_REGEX = /\b((?:https?:\/\/|www\.)[^\s"'<>]+)/gi;

  function isElementVisible(el) {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return false;
    return true;
  }

  function isNodeInViewport(node) {
    const el = (node.nodeType === Node.ELEMENT_NODE) ? node : node.parentElement;
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    return rect.bottom >= 0 && rect.right >= 0 &&
           rect.top <= (window.innerHeight || document.documentElement.clientHeight) &&
           rect.left <= (window.innerWidth || document.documentElement.clientWidth);
  }

  function extractLinks() {
    let results = [];

    // 1. Quét thẻ A
    const anchors = document.querySelectorAll('a[href]');
    anchors.forEach(a => {
      if (isElementVisible(a) && isNodeInViewport(a)) {
        let href = a.href;
        if (href.startsWith('http')) {
          results.push({ url: href, source: 'anchor', text: (a.innerText || '').trim() });
        }
      }
    });

    // 2. Quét Text Nodes (Regex)
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    let node;
    while (node = walker.nextNode()) {
      if (node.parentElement && isElementVisible(node.parentElement) && isNodeInViewport(node)) {
        let match;
        while ((match = URL_REGEX.exec(node.nodeValue)) !== null) {
          let raw = match[1];
          // Clean URL
          raw = raw.replace(/[,\.\)\]\:;]+$/,''); 
          if (!raw.startsWith('http')) raw = 'http://' + raw;
          results.push({ url: raw, source: 'text', text: 'Text Node' });
        }
      }
    }

    return results;
  }
  // --- HẾT PHẦN LOGIC CŨ ---

  function scanAndSend() {
    if (!isRunning) return;
    const links = extractLinks();
    if (links.length > 0) {
      // Gửi về Background để xử lý trùng lặp và lưu trữ
      chrome.runtime.sendMessage({ action: "save_links", data: links });
    }
  }

  // Khởi động
  function init() {
    // Kiểm tra trạng thái từ Storage (người dùng có đang Bật không?)
    chrome.storage.local.get(['isScanning'], (res) => {
      if (res.isScanning) {
        isRunning = true;
        scanAndSend(); // Quét ngay lập tức
        
        // Theo dõi cuộn trang và thay đổi DOM
        if (!observer) {
          observer = new MutationObserver(() => {
             // Dùng debounce để tránh spam quá nhiều khi cuộn
             clearTimeout(window._scanTimeout);
             window._scanTimeout = setTimeout(scanAndSend, 500); 
          });
          observer.observe(document.body, { childList: true, subtree: true });
        }
      }
    });
  }

  // Lắng nghe lệnh thay đổi trạng thái từ Popup
  chrome.storage.onChanged.addListener((changes, namespace) => {
    if (changes.isScanning) {
      isRunning = changes.isScanning.newValue;
      if (isRunning) init();
      else if (observer) {
        observer.disconnect();
        observer = null;
      }
    }
  });

  // Chạy lần đầu khi load trang
  init();

})();