// popup.js

function updateUI() {
  chrome.storage.local.get(['collectedLinks', 'isScanning'], (result) => {
    const links = result.collectedLinks || [];
    const isScanning = result.isScanning || false;
    
    // Cập nhật số lượng
    document.getElementById('totalCount').innerText = links.length;

    // Cập nhật nút Bật/Tắt
    const btn = document.getElementById('btnToggle');
    if (isScanning) {
      btn.innerText = "DỪNG QUÉT";
      btn.classList.add('stop');
    } else {
      btn.innerText = "BẮT ĐẦU QUÉT";
      btn.classList.remove('stop');
    }
  });
}

// 1. Nút Bật/Tắt
document.getElementById('btnToggle').addEventListener('click', () => {
  chrome.storage.local.get(['isScanning'], (result) => {
    const newState = !result.isScanning;
    chrome.storage.local.set({ isScanning: newState });
    updateUI();
    
    // Reload tab hiện tại để script nhận trạng thái mới ngay lập tức
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
        if(tabs[0]) chrome.tabs.reload(tabs[0].id);
    });
  });
});

// 2. Nút Tải CSV
document.getElementById('btnDownload').addEventListener('click', () => {
  chrome.storage.local.get(['collectedLinks'], (result) => {
    const links = result.collectedLinks || [];
    if (links.length === 0) {
      alert("Chưa có link nào được lưu!");
      return;
    }

    // Tạo nội dung CSV
    // Thêm BOM \uFEFF để Excel hiển thị đúng Tiếng Việt
    let csvContent = "\uFEFFUrl,Text,Source,Time\n"; 
    
    links.forEach(item => {
      // Xử lý các dấu phẩy hoặc xuống dòng trong dữ liệu để không vỡ CSV
      const safeUrl = `"${(item.url || '').replace(/"/g, '""')}"`;
      const safeText = `"${(item.text || '').replace(/"/g, '""')}"`;
      const safeSource = item.source;
      const safeTime = item.time;
      
      csvContent += `${safeUrl},${safeText},${safeSource},${safeTime}\n`;
    });

    // Tải xuống
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `All_Links_${new Date().getTime()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });
});

// 3. Nút Xóa dữ liệu
document.getElementById('btnClear').addEventListener('click', () => {
  if (confirm("Bạn có chắc muốn xóa toàn bộ link đã lưu không?")) {
    chrome.storage.local.set({ collectedLinks: [] });
    updateUI();
  }
});

// Tự động cập nhật UI khi mở popup
updateUI();

// Lắng nghe thay đổi để cập nhật số lượng realtime khi popup đang mở
chrome.storage.onChanged.addListener((changes) => {
  if (changes.collectedLinks) {
    document.getElementById('totalCount').innerText = changes.collectedLinks.newValue.length;
  }
});