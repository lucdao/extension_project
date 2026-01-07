// =========================
// Helpers: promisify chrome APIs (an toàn cho MV3 callback-based)
// =========================
function storageGet(key) {
  return new Promise((resolve) => {
    chrome.storage.local.get(key, (data) => resolve(data));
  });
}

function storageSet(obj) {
  return new Promise((resolve) => {
    chrome.storage.local.set(obj, () => resolve());
  });
}

// =========================
// Repository: lưu/đọc cấu hình
// =========================
class SettingsRepository {
  constructor(storageKey = "scannerEnabled") {
    this.storageKey = storageKey;
  }

  async getScannerEnabled() {
    const data = await storageGet(this.storageKey);
    // mặc định ON nếu chưa có giá trị
    return data[this.storageKey] !== false;
  }

  async setScannerEnabled(enabled) {
    await storageSet({ [this.storageKey]: enabled });
  }
}

// =========================
// Gateway: giao tiếp background
// =========================
class BackgroundGateway {
  async notifyToggle(enabled) {
    // sendMessage là async theo callback; ta không cần await kết quả ở đây
    chrome.runtime.sendMessage({
      action: "toggleScanner",
      enabled
    });
  }
}

// =========================
// View: chỉ lo DOM + render + bắt sự kiện UI
// =========================
class PopupView {
  constructor() {
    this.toggleSwitch = document.getElementById("toggleSwitch");
    this.statusText = document.getElementById("statusText");

    if (!this.toggleSwitch || !this.statusText) {
      throw new Error("PopupView: Không tìm thấy phần tử DOM cần thiết.");
    }
  }

  render(isEnabled) {
    this.toggleSwitch.checked = isEnabled;
    this.statusText.textContent = isEnabled ? "Scanner ON" : "Scanner OFF";
  }

  renderInstant(isEnabled) {
    const styleOverride = document.createElement("style");
    styleOverride.textContent = `.slider, .slider:before { transition: none !important; }`;
    document.head.appendChild(styleOverride);

    this.render(isEnabled);

    setTimeout(() => {
      styleOverride.remove();
    }, 50);
  }

  setBusy(isBusy) {
    this.toggleSwitch.disabled = isBusy;
  }

  onToggleChange(handler) {
    this.toggleSwitch.addEventListener("change", () => {
      handler(this.toggleSwitch.checked);
    });
  }
}

// =========================
// Controller: điều phối luồng nghiệp vụ
// =========================
class PopupController {
  constructor(view, repo, gateway) {
    this.view = view;
    this.repo = repo;
    this.gateway = gateway;

    this.isInitialized = false;
  }

  async init() {
    // 1) Load trạng thái ban đầu
    const enabled = await this.repo.getScannerEnabled();
    this.view.renderInstant(enabled);

    // 2) Bind event
    this.view.onToggleChange((newState) => this.handleToggleChange(newState));

    this.isInitialized = true;
  }

  async handleToggleChange(newState) {
    // Tránh xử lý khi chưa init (phòng trường hợp hiếm)
    if (!this.isInitialized) return;

    this.view.setBusy(true);

    try {
      // 1) Lưu cấu hình
      await this.repo.setScannerEnabled(newState);

      // 2) Cập nhật UI
      this.view.render(newState);

      // 3) Thông báo background
      await this.gateway.notifyToggle(newState);
    } catch (err) {
      // Nếu có lỗi: rollback UI theo trạng thái đã lưu (an toàn)
      const enabled = await this.repo.getScannerEnabled();
      this.view.render(enabled);
      console.error("PopupController error:", err);
    } finally {
      this.view.setBusy(false);
    }
  }
}

// =========================
// Bootstrap
// =========================
document.addEventListener("DOMContentLoaded", () => {
  const view = new PopupView();
  const repo = new SettingsRepository("scannerEnabled");
  const gateway = new BackgroundGateway();

  const controller = new PopupController(view, repo, gateway);
  controller.init();
});
