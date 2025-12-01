// === CẤU HÌNH ===
// Thay link Render của bạn vào đây
// const API_URL = "https://extension-project-ul62.onrender.com/predict";
const API_URL = "http://localhost:5000/predict"; // Dùng cái này nếu test local

// === BIẾN TOÀN CỤC ===
let visibleLinksQueue = new Set(); // Hàng đợi các link đang hiện trên màn hình
const scannedUrls = new Set();     // Danh sách các link đã từng kiểm tra (để không kiểm tra lại)
let isProcessing = false;          // Cờ đánh dấu đang gửi request

// 1. CẤU HÌNH BỘ QUAN SÁT (INTERSECTION OBSERVER)
// Công dụng: Phát hiện xem thẻ <a> có đang nằm trong màn hình hay không
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        // Nếu link lọt vào màn hình (isIntersecting)
        if (entry.isIntersecting) {
            const link = entry.target;
            const href = link.href;

            // Chỉ lấy link http/https và CHƯA từng kiểm tra
            if (href && href.startsWith("http") && !scannedUrls.has(href)) {
                visibleLinksQueue.add(href); // Thêm vào hàng đợi xử lý
            }
        }
    });
}, {
    root: null, // Quan sát viewport trình duyệt
    rootMargin: "0px", // Không cần lề
    threshold: 0.1 // Chỉ cần hiện 10% là bắt đầu tính
});

// 2. HÀM QUÉT DOM
// Tìm tất cả thẻ a và gắn bộ quan sát vào
function observeAllLinks() {
    const links = document.querySelectorAll("a");
    links.forEach(link => {
        // Chỉ quan sát những link chưa có attribute đã xử lý
        if (!link.hasAttribute("data-ai-observed")) {
            observer.observe(link);
            link.setAttribute("data-ai-observed", "true");
        }
    });
}

// 3. HÀM GỬI BATCH (GỬI THEO LÔ)
// Cứ mỗi 1 giây sẽ kiểm tra hàng đợi và gửi đi 1 lần
async function processQueue() {
    if (isProcessing || visibleLinksQueue.size === 0) return;

    isProcessing = true; // Khóa lại để không chạy chồng chéo

    // Lấy tối đa 20 link từ hàng đợi để gửi đi (Tránh quá tải server)
    const batch = Array.from(visibleLinksQueue).slice(0, 20);
    
    // Xóa các link sắp gửi khỏi hàng đợi và đánh dấu là đã quét
    batch.forEach(url => {
        visibleLinksQueue.delete(url);
        scannedUrls.add(url);
    });

    console.log(`AI Phishing: Đang kiểm tra ${batch.length} link mới xuất hiện...`);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ urls: batch })
        });

        if (response.ok) {
            const data = await response.json();
            // Xử lý kết quả trả về
            for (const [url, status] of Object.entries(data)) {
                if (status === "PHISHING") {
                    highlightLink(url);
                }
            }
        } else {
            console.warn("Server quá tải, sẽ thử lại sau.");
            // Nếu lỗi, xóa khỏi scannedUrls để lần sau cuộn tới sẽ check lại
            batch.forEach(url => scannedUrls.delete(url));
        }
    } catch (error) {
        console.error("Lỗi mạng:", error);
    } finally {
        isProcessing = false; // Mở khóa
    }
}

// 4. HÀM TÔ ĐỎ LINK (HIỂN THỊ)
function highlightLink(targetUrl) {
    // Tìm lại các thẻ a khớp với URL này trên màn hình
    // (Vì document.querySelectorAll tốn kém nên ta tìm lại thủ công)
    const links = document.querySelectorAll(`a[href="${targetUrl}"]`);
    
    links.forEach(el => {
        if (el.classList.contains("ai-phishing-detected")) return;

        el.classList.add("ai-phishing-detected");
        
        // Icon
        const warningSpan = document.createElement("span");
        warningSpan.innerHTML = "⚠️";
        warningSpan.className = "ai-phishing-icon";
        
        // Tooltip
        const tooltip = document.createElement("span");
        tooltip.innerText = "Cảnh báo: AI phát hiện lừa đảo!";
        tooltip.className = "ai-phishing-tooltip";
        
        el.appendChild(warningSpan);
        el.appendChild(tooltip);

        el.addEventListener("click", (e) => {
            if(!confirm("⚠️ CẢNH BÁO: Link này có dấu hiệu lừa đảo. Bạn có chắc muốn vào?")) {
                e.preventDefault();
                e.stopPropagation();
            }
        });
    });
}

// === KHỞI CHẠY ===

// 1. Chạy ngay khi load xong
window.addEventListener("load", () => {
    observeAllLinks();
    // Cài đặt chu kỳ quét hàng đợi: 1 giây / lần
    setInterval(processQueue, 1000);
});

// 2. Chạy lại khi DOM thay đổi (Ví dụ Youtube, Facebook load thêm bài viết mới)
const mutationObserver = new MutationObserver(() => {
    observeAllLinks();
});
mutationObserver.observe(document.body, { childList: true, subtree: true });