'use strict';

/**
 * background.js (MV3 service worker) - OOP / Control-Logic + Phishing Cache
 * - Nhận message từ content/popup
 * - Gọi server /predict để phân loại URL
 * - Cache các URL đã bị gán nhãn phishing để tránh tra lại
 */

const CONFIG = Object.freeze({
  SERVER_URL: 'https://api.phishingbkfm.id.vn/predict',
  STORAGE_KEY: 'scannerEnabled',
  CONTENT_SCRIPT_FILE: 'content.js',
  FETCH_TIMEOUT_MS: 20000,

  // Cache phishing
  PHISHING_CACHE_KEY: 'phishingUrlCache',   // lưu object {normalizedUrl: {score, ts}}
  PHISHING_CACHE_TTL_MS: 30 * 24 * 60 * 60 * 1000, // 30 ngày
  PHISHING_CACHE_MAX_ENTRIES: 2000
});

/** ========== Small async wrappers for Chrome callback APIs ========== */
class ChromeAsync {
  static storageGet(key) {
    return new Promise((resolve) => {
      chrome.storage.local.get(key, (data) => resolve(data || {}));
    });
  }

  static storageSet(obj) {
    return new Promise((resolve) => {
      chrome.storage.local.set(obj, () => resolve());
    });
  }

  static tabsQuery(queryInfo) {
    return new Promise((resolve) => {
      chrome.tabs.query(queryInfo, (tabs) => resolve(tabs || []));
    });
  }

  static executeScript(tabId, files) {
    return new Promise((resolve, reject) => {
      chrome.scripting.executeScript({ target: { tabId }, files }, () => {
        const err = chrome.runtime.lastError;
        if (err) return reject(err);
        resolve();
      });
    });
  }

  static tabsSendMessage(tabId, message) {
    return new Promise((resolve) => {
      chrome.tabs.sendMessage(tabId, message, (resp) => resolve(resp));
    });
  }
}

/** ========== Repository: đọc trạng thái bật/tắt scanner ========== */
class SettingsRepository {
  constructor(storageKey = CONFIG.STORAGE_KEY) {
    this.storageKey = storageKey;
  }

  async isScannerEnabled() {
    const data = await ChromeAsync.storageGet(this.storageKey);
    return data[this.storageKey] !== false; // default ON
  }
}

/** ========== Cache Repo: chỉ lưu URL phishing ========== */
class PhishingCacheRepository {
  constructor({
    storageKey = CONFIG.PHISHING_CACHE_KEY,
    ttlMs = CONFIG.PHISHING_CACHE_TTL_MS,
    maxEntries = CONFIG.PHISHING_CACHE_MAX_ENTRIES
  } = {}) {
    this.storageKey = storageKey;
    this.ttlMs = ttlMs;
    this.maxEntries = maxEntries;

    this._loaded = false;
    this._map = new Map(); // normalizedUrl -> { score, ts }
  }

  _normalizeUrl(rawUrl) {
    try {
      const u = new URL(rawUrl);
      // bỏ #fragment để cùng 1 trang không bị coi là khác
      u.hash = '';
      // chuẩn hoá host (URL object tự lower-case host)
      return u.toString();
    } catch {
      return null;
    }
  }

  async _ensureLoaded() {
    if (this._loaded) return;

    const data = await ChromeAsync.storageGet(this.storageKey);
    const obj = data[this.storageKey] || {};

    // đổ vào map
    for (const [k, v] of Object.entries(obj)) {
      if (v && typeof v.ts === 'number') this._map.set(k, v);
    }

    // prune ngay sau khi load
    await this._pruneAndPersist();

    this._loaded = true;
  }

  async get(rawUrl) {
    await this._ensureLoaded();
    const key = this._normalizeUrl(rawUrl);
    if (!key) return null;

    const v = this._map.get(key);
    if (!v) return null;

    // TTL
    if (Date.now() - v.ts > this.ttlMs) {
      this._map.delete(key);
      await this._persist(); // cập nhật storage
      return null;
    }

    return { url: key, score: v.score ?? null, ts: v.ts };
  }

  async put(rawUrl, score) {
    await this._ensureLoaded();
    const key = this._normalizeUrl(rawUrl);
    if (!key) return;

    this._map.set(key, { score: typeof score === 'number' ? score : null, ts: Date.now() });

    await this._pruneAndPersist();
  }

  async _pruneAndPersist() {
    // 1) xoá hết hạn
    const now = Date.now();
    for (const [k, v] of this._map.entries()) {
      if (!v || typeof v.ts !== 'number' || (now - v.ts > this.ttlMs)) {
        this._map.delete(k);
      }
    }

    // 2) giới hạn dung lượng: nếu vượt maxEntries, xoá bản ghi cũ nhất
    if (this._map.size > this.maxEntries) {
      const arr = Array.from(this._map.entries()); // [key, {ts,...}]
      arr.sort((a, b) => (a[1].ts || 0) - (b[1].ts || 0)); // cũ -> mới
      const toRemove = this._map.size - this.maxEntries;
      for (let i = 0; i < toRemove; i++) {
        this._map.delete(arr[i][0]);
      }
    }

    await this._persist();
  }

  async _persist() {
    // convert map -> object
    const obj = Object.fromEntries(this._map.entries());
    await ChromeAsync.storageSet({ [this.storageKey]: obj });
  }
}

/** ========== API Client: gọi server dự đoán URL ========== */
class PhishingApiClient {
  constructor(serverUrl = CONFIG.SERVER_URL, timeoutMs = CONFIG.FETCH_TIMEOUT_MS) {
    this.serverUrl = serverUrl;
    this.timeoutMs = timeoutMs;
  }

  async predict(url) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch(this.serverUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
        signal: controller.signal
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status} ${res.statusText} ${text}`.trim());
      }

      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }
}

/** ========== Quản lý content script (inject / stop) ========== */
class ContentScriptManager {
  constructor(contentFile = CONFIG.CONTENT_SCRIPT_FILE) {
    this.contentFile = contentFile;
  }

  async inject(tabId) {
    await ChromeAsync.executeScript(tabId, [this.contentFile]);
  }

  async requestStop(tabId) {
    await ChromeAsync.tabsSendMessage(tabId, { action: 'stopScanner' });
  }
}

/** ========== Controller: điều phối logic background ========== */
class BackgroundController {
  constructor(settingsRepo, apiClient, contentManager, phishingCacheRepo) {
    this.settingsRepo = settingsRepo;
    this.apiClient = apiClient;
    this.contentManager = contentManager;
    this.phishingCacheRepo = phishingCacheRepo;

    this.onMessage = this.onMessage.bind(this);
    this.onTabUpdated = this.onTabUpdated.bind(this);
  }

  register() {
    chrome.runtime.onMessage.addListener(this.onMessage);
    chrome.tabs.onUpdated.addListener(this.onTabUpdated);
  }

  onMessage(request, sender, sendResponse) {
    const action = request?.action;

    if (action === 'checkUrl') {
      this.handleCheckUrl(request, sendResponse);
      return true; // async
    }

    if (action === 'toggleScanner') {
      this.handleToggleScanner(request).finally(() => {});
      return;
    }

    return;
  }

  async handleCheckUrl(request, sendResponse) {
    const url = request?.url;

    if (typeof url !== 'string' || url.length === 0) {
      sendResponse({ success: false, error: 'Invalid url' });
      return;
    }

    try {
      // 1) CHECK CACHE trước: nếu đã từng phishing thì trả ngay
      const cached = await this.phishingCacheRepo.get(url);
      if (cached) {
        sendResponse({
          success: true,
          is_phishing: true,
          score: cached.score
        });
        return;
      }

      // 2) Không có cache -> gọi server
      const data = await this.apiClient.predict(url);

      const isPhishing = !!data.is_phishing;
      const score = typeof data.score === 'number' ? data.score : null;

      // 3) Nếu phishing -> lưu cache để lần sau khỏi hỏi server
      if (isPhishing) {
        await this.phishingCacheRepo.put(url, score);
      }

      sendResponse({
        success: true,
        is_phishing: isPhishing,
        score
      });
    } catch (err) {
      console.error('Server error detail:', err.name, err.message); // In rõ tên lỗi
      sendResponse({ 
      success: false, 
     error: `${err.name}: ${err.message}` // Trả về thông tin chi tiết thay vì [object]
    });
  }
  }

  async handleToggleScanner(request) {
    const enabled = !!request?.enabled;

    const tabs = await ChromeAsync.tabsQuery({ active: true, currentWindow: true });
    const tabId = tabs?.[0]?.id;
    if (!tabId) return;

    if (enabled) {
      try {
        await this.contentManager.inject(tabId);
      } catch (err) {
        console.error('Inject content.js failed:', err);
      }
    } else {
      await this.contentManager.requestStop(tabId);
    }
  }

  async onTabUpdated(tabId, changeInfo, tab) {
    if (changeInfo.status !== 'complete') return;
    if (!tab?.active) return;

    const enabled = await this.settingsRepo.isScannerEnabled();
    if (!enabled) return;

    try {
      await this.contentManager.inject(tabId);
    } catch {
      // ignore (chrome://, webstore, ...)
    }
  }
}

/** ========== Bootstrap ========== */
(() => {
  const repo = new SettingsRepository(CONFIG.STORAGE_KEY);
  const api = new PhishingApiClient(CONFIG.SERVER_URL, CONFIG.FETCH_TIMEOUT_MS);
  const contentMgr = new ContentScriptManager(CONFIG.CONTENT_SCRIPT_FILE);
  const phishingCache = new PhishingCacheRepository({
    storageKey: CONFIG.PHISHING_CACHE_KEY,
    ttlMs: CONFIG.PHISHING_CACHE_TTL_MS,
    maxEntries: CONFIG.PHISHING_CACHE_MAX_ENTRIES
  });

  const controller = new BackgroundController(repo, api, contentMgr, phishingCache);
  controller.register();
})();
