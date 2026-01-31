/**
 * Application State Management
 *
 * 중앙화된 애플리케이션 상태 관리 with Observer Pattern
 */

class AppState {
  constructor() {
    this.state = {
      // User & Auth
      user: null,

      // Current Context
      currentSku: null,
      currentInk: null,

      // Inspection State
      inspection: {
        selectedProduct: null,
        lastResult: null,
        isProcessing: false,
        uploadedFile: null,
        // v7 Legacy State
        currentArtifacts: { overlay: '', heatmap: '' },
        worstCase: null,
        worstCfg: null,
        worstHotspot: null,
        inspectMode: 'signature',
        hotspotBound: false,
      },

      // Single Analysis State
      analysis: {
        uploadedFile: null,
        uploadedBlackFile: null,
        result: null,
      },

      // Products (v7 Legacy)
      productStatus: {},

      // History & Stats
      history: {
        filters: {},
        currentPage: 1,
      },

      // UI State
      ui: {
        theme: 'dark',
        sidebarCollapsed: false,
      },
    };

    // Listeners: Map<path, Array<callback>>
    this.listeners = new Map();

    // Load persisted state
    this.loadPersistedState();
  }

  /**
   * 상태 구독
   * @param {string} path - 상태 경로 (예: 'inspection.isProcessing')
   * @param {Function} callback - 상태 변경 시 호출될 콜백
   * @returns {Function} 구독 해제 함수
   */
  subscribe(path, callback) {
    if (!this.listeners.has(path)) {
      this.listeners.set(path, []);
    }

    this.listeners.get(path).push(callback);

    // 현재 값으로 즉시 호출
    const currentValue = this.getState(path);
    if (currentValue !== undefined) {
      callback(currentValue);
    }

    // Unsubscribe function
    return () => {
      const callbacks = this.listeners.get(path);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }

  /**
   * 상태 업데이트
   * @param {string} path - 상태 경로
   * @param {any} value - 새 값
   */
  setState(path, value) {
    const keys = path.split('.');
    let current = this.state;

    // Navigate to parent
    for (let i = 0; i < keys.length - 1; i++) {
      if (!(keys[i] in current)) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }

    // Set value
    const lastKey = keys[keys.length - 1];
    const oldValue = current[lastKey];
    current[lastKey] = value;

    // Notify listeners if value changed
    if (oldValue !== value) {
      this.notifyListeners(path, value);

      // Persist certain states
      if (path.startsWith('ui.')) {
        this.persistState();
      }
    }
  }

  /**
   * 상태 조회
   * @param {string} path - 상태 경로
   * @returns {any} 상태 값
   */
  getState(path) {
    const keys = path.split('.');
    let current = this.state;

    for (const key of keys) {
      if (current === null || current === undefined) {
        return undefined;
      }
      current = current[key];
    }

    return current;
  }

  /**
   * 상태 병합
   * @param {string} path - 상태 경로
   * @param {Object} updates - 업데이트할 객체
   */
  mergeState(path, updates) {
    const current = this.getState(path);
    if (typeof current === 'object' && current !== null) {
      this.setState(path, { ...current, ...updates });
    } else {
      this.setState(path, updates);
    }
  }

  /**
   * 리스너에게 알림
   * @private
   */
  notifyListeners(path, value) {
    // Exact path listeners
    if (this.listeners.has(path)) {
      this.listeners.get(path).forEach(callback => {
        try {
          callback(value);
        } catch (error) {
          console.error(`Error in listener for ${path}:`, error);
        }
      });
    }

    // Parent path listeners (예: 'inspection' 경로는 'inspection.isProcessing' 변경 시에도 알림)
    const parts = path.split('.');
    for (let i = parts.length - 1; i > 0; i--) {
      const parentPath = parts.slice(0, i).join('.');
      if (this.listeners.has(parentPath)) {
        const parentValue = this.getState(parentPath);
        this.listeners.get(parentPath).forEach(callback => {
          try {
            callback(parentValue);
          } catch (error) {
            console.error(`Error in parent listener for ${parentPath}:`, error);
          }
        });
      }
    }
  }

  /**
   * 상태 초기화
   * @param {string} path - 초기화할 경로
   */
  resetState(path) {
    const defaultStates = {
      'inspection': {
        selectedProduct: null,
        lastResult: null,
        isProcessing: false,
        uploadedFile: null,
      },
      'analysis': {
        uploadedFile: null,
        uploadedBlackFile: null,
        result: null,
      },
    };

    if (defaultStates[path]) {
      this.setState(path, defaultStates[path]);
    }
  }

  /**
   * 로컬 스토리지에서 상태 로드
   * @private
   */
  loadPersistedState() {
    try {
      const persistedUI = localStorage.getItem('appState.ui');
      if (persistedUI) {
        this.state.ui = JSON.parse(persistedUI);
      }
    } catch (error) {
      console.warn('Failed to load persisted state:', error);
    }
  }

  /**
   * 로컬 스토리지에 상태 저장
   * @private
   */
  persistState() {
    try {
      localStorage.setItem('appState.ui', JSON.stringify(this.state.ui));
    } catch (error) {
      console.warn('Failed to persist state:', error);
    }
  }

  /**
   * 전체 상태 덤프 (디버깅용)
   */
  dump() {
    return JSON.parse(JSON.stringify(this.state));
  }
}

// Singleton instance
export const appState = new AppState();

// Global access (for debugging)
if (typeof window !== 'undefined') {
  window.appState = appState;
}

export default appState;
