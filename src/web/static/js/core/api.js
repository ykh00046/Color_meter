/**
 * API Client
 *
 * 백엔드 API 호출을 위한 클라이언트 레이어
 */

class ApiClient {
  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * HTTP GET 요청
   * @param {string} endpoint - API 엔드포인트
   * @param {Object} params - 쿼리 파라미터
   * @returns {Promise<any>}
   */
  async get(endpoint, params = {}) {
    const url = new URL(`${this.baseUrl}${endpoint}`, window.location.origin);

    // Add query parameters
    Object.keys(params).forEach(key => {
      if (params[key] !== null && params[key] !== undefined) {
        url.searchParams.append(key, params[key]);
      }
    });

    return this.request(url.toString(), { method: 'GET' });
  }

  /**
   * HTTP POST 요청
   * @param {string} endpoint - API 엔드포인트
   * @param {Object|FormData} data - 요청 데이터
   * @param {string} contentType - 'json' | 'multipart'
   * @returns {Promise<any>}
   */
  async post(endpoint, data, contentType = 'json') {
    const url = `${this.baseUrl}${endpoint}`;
    const options = { method: 'POST' };

    if (contentType === 'json') {
      options.headers = this.defaultHeaders;
      options.body = JSON.stringify(data);
    } else if (contentType === 'multipart') {
      // FormData를 사용하면 Content-Type이 자동 설정됨
      options.body = data;
    }

    return this.request(url, options);
  }

  /**
   * HTTP PUT 요청
   * @param {string} endpoint - API 엔드포인트
   * @param {Object} data - 요청 데이터
   * @returns {Promise<any>}
   */
  async put(endpoint, data) {
    const url = `${this.baseUrl}${endpoint}`;
    const options = {
      method: 'PUT',
      headers: this.defaultHeaders,
      body: JSON.stringify(data),
    };

    return this.request(url, options);
  }

  /**
   * HTTP DELETE 요청
   * @param {string} endpoint - API 엔드포인트
   * @returns {Promise<any>}
   */
  async delete(endpoint) {
    const url = `${this.baseUrl}${endpoint}`;
    const options = { method: 'DELETE' };

    return this.request(url, options);
  }

  /**
   * 공통 요청 핸들러
   * @private
   */
  async request(url, options) {
    try {
      const headers = new Headers(options.headers || {});
      if (url.includes('/api/v7/') && !headers.has('X-User-Role')) {
        headers.set('X-User-Role', 'operator');
      }
      options.headers = headers;

      const response = await fetch(url, options);

      // Handle non-OK responses
      if (!response.ok) {
        const error = await this.parseError(response);
        throw error;
      }

      // Parse response
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      } else {
        return await response.text();
      }
    } catch (error) {
      // Re-throw with better error info
      if (error.name === 'TypeError' && error.message === 'Failed to fetch') {
        throw new Error('Network error: Unable to connect to server');
      }
      throw error;
    }
  }

  /**
   * 에러 응답 파싱
   * @private
   */
  async parseError(response) {
    const error = new Error();
    error.status = response.status;
    error.statusText = response.statusText;

    try {
      const body = await response.json();
      error.message = body.detail || body.message || response.statusText;
      error.data = body;
    } catch {
      error.message = response.statusText;
    }

    return error;
  }
}

// Create singleton instance
export const apiClient = new ApiClient();

// V7 API specific methods
export const v7Api = {
  /**
   * STD 등록
   */
  async register(formData) {
    return apiClient.post('/v7/register', formData, 'multipart');
  },

  /**
   * 검사 실행
   */
  async inspect(formData) {
    return apiClient.post('/v7/inspect', formData, 'multipart');
  },

  /**
   * 단독 분석
   */
  async analyzeSingle(formData) {
    return apiClient.post('/v7/analyze_single', formData, 'multipart');
  },

  /**
   * SKU 제품 목록 조회
   */
  async getProducts() {
    return apiClient.get('/v7/products');
  },

  /**
   * STD 이력 조회
   */
  async getStdHistory() {
    return apiClient.get('/v7/std_history');
  },

  /**
   * 검사 이력 조회
   */
  async getInspectionHistory(params = {}) {
    return apiClient.get('/inspection/history', params);
  },

  /**
   * 통계 데이터 조회
   */
  async getStats(params = {}) {
    return apiClient.get('/inspection/stats', params);
  },

  /**
   * Plate Gate 추출 (Phase 6 연동)
   */
  async extractPlateGate(formData) {
    return apiClient.post('/v7/plate_gate', formData, 'multipart');
  },

};

// Export both
export { ApiClient };
export default apiClient;

// Global access (for debugging and legacy support)
if (typeof window !== 'undefined') {
  window.apiClient = apiClient;
  window.v7Api = v7Api;
}
