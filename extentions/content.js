// content.js (OOP - Boundary)
// Trách nhiệm: giám sát DOM, thu thập URL, hiển thị cảnh báo thị giác.
// KHÔNG xử lý policy/ngưỡng -> việc đó thuộc Background/Logic.

(() => {
  // Chống inject nhiều lần
  if (window.__PhishingScannerBoundaryInstance__) return;

  class PhishingScannerBoundary {
    constructor(config = {}) {
      this.config = {
        rootMargin: "50px",
        scanSelector: "a[href]",
        mutationDebounceMs: 200,
        warningClass: "ps-warning-link",
        iconClass: "ps-warning-icon",
        storageKey: "scannerEnabled", // nếu bạn dùng popup bật/tắt
        ...config
      };

      this._enabled = true;
      this._mutationTimer = null;

      // SVG icon cảnh báo (nhỏ gọn)
      this.WARNING_ICON_SVG = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
          viewBox="0 0 24 24" fill="none" stroke="red" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round"
          style="margin-left:6px; vertical-align:middle; cursor:help;">
          <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
          <line x1="12" y1="9" x2="12" y2="13"></line>
          <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
      `;

      // Observer
      this.viewportObserver = null;
      this.domMutationObserver = null;

      // Inject CSS 1 lần
      this._injectStyles();
    }

    // ===== Public lifecycle =====
    async bootstrap() {
      // Nếu bạn có popup bật/tắt: chỉ chạy khi storage cho phép
      const enabled = await this._readEnabledFromStorage();
      if (!enabled) {
        this._enabled = false;
        return;
      }

      this.start();
      console.log("Phishing Scanner: Boundary started.");
    }

    start() {
      if (this._enabled === false) this._enabled = true;

      if (!this.viewportObserver) this.viewportObserver = this._createViewportObserver();
      if (!this.domMutationObserver) this.domMutationObserver = this._createMutationObserver();

      // scan lần đầu
      this.scan();
    }

    stop() {
      this._enabled = false;

      if (this.viewportObserver) {
        this.viewportObserver.disconnect();
        this.viewportObserver = null;
      }
      if (this.domMutationObserver) {
        this.domMutationObserver.disconnect();
        this.domMutationObserver = null;
      }
      if (this._mutationTimer) {
        clearTimeout(this._mutationTimer);
        this._mutationTimer = null;
      }

      console.log("Phishing Scanner: Boundary stopped.");
    }

    // ===== Core boundary responsibilities =====
    scan() {
      if (!this._enabled) return;

      const links = document.querySelectorAll(this.config.scanSelector);
      links.forEach((link) => this._observeLink(link));
    }

    // ===== Internal helpers =====
    _observeLink(linkEl) {
      // Dùng dataset để chống lặp (bền vững kể cả inject lại)
      if (linkEl.dataset.psObserved === "1") return;
      linkEl.dataset.psObserved = "1";

      // chỉ observe khi có observer
      if (this.viewportObserver) this.viewportObserver.observe(linkEl);
    }

    _createViewportObserver() {
      return new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!this._enabled) return;
            if (!entry.isIntersecting) return;

            const linkEl = entry.target;
            this._processLink(linkEl);

            // xử lý xong thì bỏ theo dõi
            this.viewportObserver.unobserve(linkEl);
          });
        },
        { rootMargin: this.config.rootMargin }
      );
    }

    _createMutationObserver() {
      const observer = new MutationObserver(() => {
        if (!this._enabled) return;

        // debounce để tránh scan quá dày trên trang dynamic
        if (this._mutationTimer) clearTimeout(this._mutationTimer);
        this._mutationTimer = setTimeout(() => this.scan(), this.config.mutationDebounceMs);
      });

      observer.observe(document.body, { childList: true, subtree: true });
      return observer;
    }

    _processLink(linkEl) {
      // chống check lại
      if (linkEl.dataset.psChecked === "1") return;
      linkEl.dataset.psChecked = "1";

      const url = linkEl.href || "";
      if (!this._isCandidateUrl(url)) return;

      this._requestCheck(url)
        .then((res) => {
          if (!this._enabled) return;
          if (!res || !res.success) return;

          // Boundary chỉ hiển thị theo “kết quả logic” trả về
          if (res.is_phishing) {
            this._renderWarning(linkEl, res.score);
          }
        })
        .catch(() => {
          // im lặng để không phá UX
        });
    }

    _isCandidateUrl(url) {
      // Boundary chỉ lọc cơ bản (không phải policy)
      return typeof url === "string" && url.startsWith("http");
    }

    _requestCheck(url) {
      return new Promise((resolve) => {
        try {
          chrome.runtime.sendMessage({ action: "checkUrl", url }, (response) => {
            if (chrome.runtime.lastError) return resolve(null);
            resolve(response || null);
          });
        } catch {
          resolve(null);
        }
      });
    }

    _renderWarning(linkEl, score) {
      // chống render lại
      if (linkEl.dataset.psFlagged === "1") return;
      linkEl.dataset.psFlagged = "1";

      linkEl.classList.add(this.config.warningClass);

      const icon = document.createElement("span");
      icon.className = this.config.iconClass;
      icon.innerHTML = this.WARNING_ICON_SVG;

      const pct = typeof score === "number" ? (score * 100).toFixed(1) : "?";
      icon.title = `CẢNH BÁO: nghi ngờ lừa đảo (${pct}%)`;

      linkEl.appendChild(icon);
    }

    _injectStyles() {
      if (document.getElementById("ps-boundary-style")) return;

      const style = document.createElement("style");
      style.id = "ps-boundary-style";
      style.textContent = `
        .${this.config.warningClass} {
          border-bottom: 2px solid red !important;
          background-color: rgba(255,0,0,0.10) !important;
        }
      `;
      document.head.appendChild(style);
    }

    _readEnabledFromStorage() {
      return new Promise((resolve) => {
        // Nếu manifest không có permission "storage" thì vẫn chạy mặc định
        if (!chrome?.storage?.local) return resolve(true);

        chrome.storage.local.get(this.config.storageKey, (data) => {
          const enabled = data?.[this.config.storageKey] !== false;
          resolve(enabled);
        });
      });
    }
  }

  // Khởi chạy 1 instance duy nhất
  const instance = new PhishingScannerBoundary();
  window.__PhishingScannerBoundaryInstance__ = instance;

  // Nếu DOM chưa sẵn sàng thì đợi
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => instance.bootstrap());
  } else {
    instance.bootstrap();
  }
})();
