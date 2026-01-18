/**
 * UI Components Library
 *
 * 재사용 가능한 UI 컴포넌트를 생성하는 모듈
 */

export const Components = {
  /**
   * 버튼 생성
   * @param {string} text - 버튼 텍스트
   * @param {string} variant - 'primary' | 'secondary' | 'success' | 'warning' | 'error'
   * @param {Function} onClick - 클릭 핸들러
   * @returns {HTMLButtonElement}
   */
  createButton(text, variant = 'primary', onClick = null) {
    const btn = document.createElement('button');
    btn.className = `btn btn-${variant}`;
    btn.textContent = text;

    if (onClick) {
      btn.addEventListener('click', onClick);
    }

    return btn;
  },

  /**
   * 카드 생성
   * @param {string} title - 카드 제목
   * @param {HTMLElement|string} content - 카드 내용
   * @param {Object} options - 추가 옵션 {className, headerClassName}
   * @returns {HTMLDivElement}
   */
  createCard(title, content, options = {}) {
    const card = document.createElement('div');
    card.className = options.className || 'card';

    if (title) {
      const header = document.createElement('div');
      header.className = options.headerClassName || 'card-header';
      header.textContent = title;
      card.appendChild(header);
    }

    const body = document.createElement('div');
    if (typeof content === 'string') {
      body.innerHTML = content;
    } else if (content instanceof HTMLElement) {
      body.appendChild(content);
    }
    card.appendChild(body);

    return card;
  },

  /**
   * 메트릭 카드 생성
   * @param {string} label - 메트릭 레이블
   * @param {string|number} value - 메트릭 값
   * @param {string} unit - 단위 (선택)
   * @param {string} status - 'success' | 'warning' | 'error' (선택)
   * @returns {HTMLDivElement}
   */
  createMetricCard(label, value, unit = '', status = null) {
    const card = document.createElement('div');
    card.className = 'metric-card';

    const labelEl = document.createElement('div');
    labelEl.className = 'metric-label';
    labelEl.textContent = label;

    const valueEl = document.createElement('div');
    valueEl.className = 'metric-value';
    if (status) {
      valueEl.classList.add(`text-${status}`);
    }

    if (unit) {
      valueEl.innerHTML = `${value}<span class="text-sm ml-1">${unit}</span>`;
    } else {
      valueEl.textContent = value;
    }

    card.appendChild(labelEl);
    card.appendChild(valueEl);

    return card;
  },

  /**
   * 탭 시스템 생성
   * @param {Array<{id: string, label: string, content: HTMLElement, icon?: string}>} tabs - 탭 목록
   * @param {Function} onTabChange - 탭 변경 콜백
   * @returns {HTMLDivElement}
   */
  createTabs(tabs, onTabChange = null) {
    const container = document.createElement('div');
    container.className = 'tabs-container';

    // 탭 헤더
    const tabHeaders = document.createElement('div');
    tabHeaders.className = 'flex gap-2 border-b border-default mb-6';

    // 탭 컨텐츠 컨테이너
    const tabContents = document.createElement('div');
    tabContents.className = 'tab-contents';

    tabs.forEach((tab, index) => {
      // 헤더 버튼
      const btn = document.createElement('button');
      btn.className = `tab-btn ${index === 0 ? 'active' : ''}`;
      btn.dataset.tabId = tab.id;

      if (tab.icon) {
        const icon = document.createElement('i');
        icon.className = tab.icon;
        btn.appendChild(icon);
      }

      const label = document.createElement('span');
      label.textContent = tab.label;
      btn.appendChild(label);

      btn.addEventListener('click', () => {
        // 모든 탭 비활성화
        tabHeaders.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        tabContents.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));

        // 현재 탭 활성화
        btn.classList.add('active');
        const contentEl = document.getElementById(`tab-content-${tab.id}`);
        contentEl.classList.remove('hidden');
        contentEl.classList.add('active');

        if (onTabChange) {
          onTabChange(tab.id);
        }
      });

      tabHeaders.appendChild(btn);

      // 컨텐츠
      const content = document.createElement('div');
      content.id = `tab-content-${tab.id}`;
      content.className = `tab-content ${index === 0 ? 'active' : 'hidden'}`;
      content.appendChild(tab.content);

      tabContents.appendChild(content);
    });

    container.appendChild(tabHeaders);
    container.appendChild(tabContents);

    return container;
  },

  /**
   * 데이터 테이블 생성
   * @param {Array<Object>} data - 데이터 배열
   * @param {Array<{key: string, label: string, formatter?: Function}>} columns - 컬럼 정의
   * @returns {HTMLTableElement}
   */
  createDataTable(data, columns) {
    const table = document.createElement('table');
    table.className = 'w-full text-sm';

    // 헤더
    const thead = document.createElement('thead');
    thead.className = 'text-dim uppercase border-b border-default';
    const headerRow = document.createElement('tr');

    columns.forEach(col => {
      const th = document.createElement('th');
      th.className = 'py-3 px-4 text-left';
      th.textContent = col.label;
      headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // 바디
    const tbody = document.createElement('tbody');
    tbody.className = 'divide-y divide-border-default';

    data.forEach(row => {
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-surface transition-colors';

      columns.forEach(col => {
        const td = document.createElement('td');
        td.className = 'py-3 px-4';
        const value = row[col.key];
        td.textContent = col.formatter ? col.formatter(value, row) : value;
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });

    table.appendChild(tbody);

    return table;
  },

  /**
   * 로딩 스피너 생성
   * @param {string} size - 'sm' | 'md' | 'lg'
   * @returns {HTMLDivElement}
   */
  createSpinner(size = 'md') {
    const spinner = document.createElement('div');
    spinner.className = `spinner spinner-${size}`;
    spinner.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
    return spinner;
  },

  /**
   * 빈 상태 메시지 생성
   * @param {string} message - 메시지
   * @param {string} icon - Font Awesome 아이콘 클래스
   * @returns {HTMLDivElement}
   */
  createEmptyState(message, icon = 'fa-inbox') {
    const container = document.createElement('div');
    container.className = 'flex flex-col items-center justify-center py-12 text-center';

    const iconEl = document.createElement('i');
    iconEl.className = `fa-solid ${icon} text-6xl text-dim mb-4`;

    const messageEl = document.createElement('p');
    messageEl.className = 'text-lg text-secondary';
    messageEl.textContent = message;

    container.appendChild(iconEl);
    container.appendChild(messageEl);

    return container;
  }
};

// Export for ES6 modules
export default Components;
