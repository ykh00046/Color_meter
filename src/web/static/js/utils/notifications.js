/**
 * Notification Utility
 *
 * 사용자 알림(토스트) 표시 유틸리티
 */

/**
 * 사용자 알림 표시
 * @param {string} title - 알림 제목
 * @param {string} message - 알림 메시지
 * @param {string} type - 'success' | 'error' | 'warning' | 'info'
 * @param {number} duration - 표시 시간 (ms), 0이면 수동으로 닫기
 */
export function showNotification(title, message, type = 'info', duration = 5000) {
  const container = document.getElementById('notificationContainer') || createNotificationContainer();

  const notification = document.createElement('div');
  notification.className = `notification notification-${type} animate-slide-in`;

  notification.innerHTML = `
    <div class="flex items-start gap-3">
      <div class="notification-icon">${getIcon(type)}</div>
      <div class="flex-1">
        <div class="notification-title">${escapeHtml(title)}</div>
        <div class="notification-message">${escapeHtml(message)}</div>
      </div>
      <button class="notification-close" aria-label="Close notification">
        <i class="fa-solid fa-times"></i>
      </button>
    </div>
  `;

  // 닫기 버튼 이벤트
  notification.querySelector('.notification-close').addEventListener('click', () => {
    closeNotification(notification);
  });

  container.appendChild(notification);

  // 자동 제거
  if (duration > 0) {
    setTimeout(() => {
      closeNotification(notification);
    }, duration);
  }

  return notification;
}

/**
 * 알림 닫기
 * @param {HTMLElement} notification - 알림 요소
 */
function closeNotification(notification) {
  notification.classList.add('animate-slide-out');
  setTimeout(() => {
    notification.remove();
  }, 300);
}

/**
 * 알림 컨테이너 생성
 * @returns {HTMLDivElement}
 */
function createNotificationContainer() {
  const container = document.createElement('div');
  container.id = 'notificationContainer';
  container.className = 'fixed top-4 right-4 z-50 space-y-2';
  document.body.appendChild(container);
  return container;
}

/**
 * 타입별 아이콘 가져오기
 * @param {string} type - 알림 타입
 * @returns {string} HTML 아이콘 문자열
 */
function getIcon(type) {
  const icons = {
    success: '<i class="fa-solid fa-circle-check text-success"></i>',
    error: '<i class="fa-solid fa-circle-xmark text-error"></i>',
    warning: '<i class="fa-solid fa-triangle-exclamation text-warning"></i>',
    info: '<i class="fa-solid fa-circle-info text-info"></i>',
  };
  return icons[type] || icons.info;
}

/**
 * HTML 이스케이프
 * @param {string} text - 이스케이프할 텍스트
 * @returns {string}
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * 성공 알림 (단축)
 */
export function showSuccess(message, title = 'Success') {
  return showNotification(title, message, 'success');
}

/**
 * 에러 알림 (단축)
 */
export function showError(message, title = 'Error') {
  return showNotification(title, message, 'error');
}

/**
 * 경고 알림 (단축)
 */
export function showWarning(message, title = 'Warning') {
  return showNotification(title, message, 'warning');
}

/**
 * 정보 알림 (단축)
 */
export function showInfo(message, title = 'Info') {
  return showNotification(title, message, 'info');
}

// Global 함수로도 사용 가능하도록 (레거시 지원)
if (typeof window !== 'undefined') {
  window.showNotification = showNotification;
  window.showSuccess = showSuccess;
  window.showError = showError;
  window.showWarning = showWarning;
  window.showInfo = showInfo;
}
