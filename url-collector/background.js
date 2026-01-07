// background.js

// Lắng nghe tin nhắn từ Content Script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "save_links") {
    const newLinks = request.data; // Danh sách link mới tìm được
    
    if (!newLinks || newLinks.length === 0) return;

    // Lấy dữ liệu cũ từ bộ nhớ
    chrome.storage.local.get(['collectedLinks', 'isScanning'], (result) => {
      // Nếu đang tắt chế độ quét thì bỏ qua
      if (result.isScanning === false) return;

      let currentLinks = result.collectedLinks || [];
      
      // Tạo Set chứa các URL đã có để check trùng cho nhanh
      const existingUrls = new Set(currentLinks.map(item => item.url));
      let countAdded = 0;

      newLinks.forEach(linkObj => {
        if (!existingUrls.has(linkObj.url)) {
          // Thêm timestamp để biết lấy lúc nào
          linkObj.time = new Date().toLocaleString('vi-VN');
          currentLinks.push(linkObj);
          existingUrls.add(linkObj.url);
          countAdded++;
        }
      });

      // Nếu có link mới thì lưu lại vào bộ nhớ
      if (countAdded > 0) {
        chrome.storage.local.set({ collectedLinks: currentLinks }, () => {
          console.log(`Đã lưu thêm ${countAdded} link mới.`);
        });
      }
    });
  }
});