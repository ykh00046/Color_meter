/**
 * Performance Optimization Module for Lens Signature Engine v7
 *
 * Provides client-side performance optimizations:
 * - Lazy loading for heavy modules (Plotly.js, charts)
 * - Client-side caching with TTL and LRU eviction
 * - Progressive image loading (thumbnail → full-res)
 * - Request deduplication
 * - Resource preloading
 *
 * Target: Page load <3s initial, <1s subsequent
 */
(function() {
    const v7 = window.v7 || (window.v7 = {});
    v7.perf = v7.perf || {};

    // Configuration
    const CONFIG = {
        cache: {
            maxEntries: 50,
            defaultTTL: 5 * 60 * 1000,  // 5 minutes
            cleanupInterval: 60 * 1000   // 1 minute
        },
        lazyLoad: {
            rootMargin: '50px',
            threshold: 0.1
        },
        imageLoading: {
            lowResQuality: 0.3,
            highResQuality: 0.9
        }
    };

    // State
    let state = {
        cache: new Map(),
        cacheStats: { hits: 0, misses: 0, evictions: 0 },
        pendingRequests: new Map(),
        loadedModules: new Set(),
        observer: null,
        cleanupTimer: null
    };

    /**
     * Initialize performance optimizations
     */
    v7.perf.init = function() {
        // Start cache cleanup timer
        startCacheCleanup();

        // Initialize lazy loading observer
        initLazyLoadObserver();

        // Add performance marks
        markPerformance('v7.perf.init');

        console.log('[v7.perf] Performance optimizations initialized');
    };

    // ============================================================================
    // CLIENT-SIDE CACHING
    // ============================================================================

    /**
     * Get from cache or fetch with caching
     *
     * @param {string} key - Cache key
     * @param {Function} fetchFn - Function to fetch data if not cached
     * @param {number} ttl - Time to live in milliseconds
     * @returns {Promise} - Cached or fetched data
     */
    v7.perf.cached = async function(key, fetchFn, ttl = CONFIG.cache.defaultTTL) {
        // Check cache
        const cached = state.cache.get(key);

        if (cached && Date.now() < cached.expiry) {
            state.cacheStats.hits++;
            console.log(`[v7.perf] Cache HIT: ${key}`);
            return cached.data;
        }

        // Cache miss
        state.cacheStats.misses++;
        console.log(`[v7.perf] Cache MISS: ${key}`);

        // Check for pending request (deduplication)
        if (state.pendingRequests.has(key)) {
            console.log(`[v7.perf] Request DEDUP: ${key}`);
            return state.pendingRequests.get(key);
        }

        // Fetch data
        const promise = fetchFn().then(data => {
            // Store in cache
            state.cache.set(key, {
                data,
                expiry: Date.now() + ttl,
                size: estimateSize(data)
            });

            // Remove from pending
            state.pendingRequests.delete(key);

            // Evict if over limit
            evictIfNeeded();

            return data;
        }).catch(err => {
            // Remove from pending on error
            state.pendingRequests.delete(key);
            throw err;
        });

        state.pendingRequests.set(key, promise);

        return promise;
    };

    /**
     * Clear cache entry or entire cache
     */
    v7.perf.clearCache = function(key = null) {
        if (key) {
            state.cache.delete(key);
            console.log(`[v7.perf] Cache cleared: ${key}`);
        } else {
            state.cache.clear();
            state.cacheStats = { hits: 0, misses: 0, evictions: 0 };
            console.log('[v7.perf] Entire cache cleared');
        }
    };

    /**
     * Get cache statistics
     */
    v7.perf.getCacheStats = function() {
        const totalRequests = state.cacheStats.hits + state.cacheStats.misses;
        const hitRate = totalRequests > 0 ? (state.cacheStats.hits / totalRequests * 100).toFixed(1) : 0;

        return {
            ...state.cacheStats,
            size: state.cache.size,
            hitRate: hitRate + '%',
            totalRequests
        };
    };

    /**
     * Estimate data size (rough approximation)
     */
    function estimateSize(data) {
        try {
            return new Blob([JSON.stringify(data)]).size;
        } catch {
            return 1024;  // Default 1KB
        }
    }

    /**
     * Evict oldest entries if over limit
     */
    function evictIfNeeded() {
        if (state.cache.size <= CONFIG.cache.maxEntries) return;

        // Sort by expiry (oldest first)
        const entries = Array.from(state.cache.entries()).sort((a, b) => a[1].expiry - b[1].expiry);

        // Remove oldest entries until under limit
        const toRemove = state.cache.size - CONFIG.cache.maxEntries;
        for (let i = 0; i < toRemove; i++) {
            state.cache.delete(entries[i][0]);
            state.cacheStats.evictions++;
        }

        console.log(`[v7.perf] Evicted ${toRemove} cache entries`);
    }

    /**
     * Clean up expired cache entries
     */
    function cleanupExpiredCache() {
        const now = Date.now();
        let removed = 0;

        for (const [key, value] of state.cache.entries()) {
            if (now >= value.expiry) {
                state.cache.delete(key);
                removed++;
            }
        }

        if (removed > 0) {
            console.log(`[v7.perf] Cleaned up ${removed} expired cache entries`);
        }
    }

    /**
     * Start cache cleanup timer
     */
    function startCacheCleanup() {
        state.cleanupTimer = setInterval(() => {
            cleanupExpiredCache();
        }, CONFIG.cache.cleanupInterval);
    }

    // ============================================================================
    // LAZY LOADING
    // ============================================================================

    /**
     * Lazy load heavy module (e.g., Plotly.js)
     *
     * @param {string} moduleName - Module name
     * @param {string} scriptUrl - Script URL
     * @returns {Promise} - Resolves when module is loaded
     */
    v7.perf.lazyLoadModule = function(moduleName, scriptUrl) {
        if (state.loadedModules.has(moduleName)) {
            console.log(`[v7.perf] Module already loaded: ${moduleName}`);
            return Promise.resolve();
        }

        // Check for pending load
        const cacheKey = `module:${moduleName}`;
        if (state.pendingRequests.has(cacheKey)) {
            return state.pendingRequests.get(cacheKey);
        }

        console.log(`[v7.perf] Lazy loading module: ${moduleName}`);

        const promise = new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = scriptUrl;
            script.async = true;

            script.onload = () => {
                state.loadedModules.add(moduleName);
                state.pendingRequests.delete(cacheKey);
                console.log(`[v7.perf] Module loaded: ${moduleName}`);
                resolve();
            };

            script.onerror = () => {
                state.pendingRequests.delete(cacheKey);
                console.error(`[v7.perf] Failed to load module: ${moduleName}`);
                reject(new Error(`Failed to load ${moduleName}`));
            };

            document.head.appendChild(script);
        });

        state.pendingRequests.set(cacheKey, promise);
        return promise;
    };

    /**
     * Lazy load Plotly.js for 3D visualizations
     */
    v7.perf.loadPlotly = function() {
        return v7.perf.lazyLoadModule('plotly', 'https://cdn.plot.ly/plotly-2.27.0.min.js');
    };

    /**
     * Lazy load Chart.js
     */
    v7.perf.loadChartJS = function() {
        return v7.perf.lazyLoadModule('chartjs', 'https://cdn.jsdelivr.net/npm/chart.js');
    };

    /**
     * Initialize Intersection Observer for lazy loading
     */
    function initLazyLoadObserver() {
        if (!('IntersectionObserver' in window)) {
            console.warn('[v7.perf] IntersectionObserver not supported');
            return;
        }

        state.observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const element = entry.target;

                    // Load image
                    if (element.dataset.src) {
                        loadProgressiveImage(element);
                    }

                    // Load module
                    if (element.dataset.module) {
                        const moduleName = element.dataset.module;
                        const scriptUrl = element.dataset.moduleUrl;
                        v7.perf.lazyLoadModule(moduleName, scriptUrl);
                    }

                    // Stop observing
                    state.observer.unobserve(element);
                }
            });
        }, CONFIG.lazyLoad);
    }

    /**
     * Observe element for lazy loading
     */
    v7.perf.observe = function(element) {
        if (state.observer) {
            state.observer.observe(element);
        }
    };

    // ============================================================================
    // PROGRESSIVE IMAGE LOADING
    // ============================================================================

    /**
     * Load image progressively (low-res → high-res)
     *
     * @param {HTMLImageElement} img - Image element
     * @param {string} lowResSrc - Low resolution image URL (optional)
     * @param {string} highResSrc - High resolution image URL
     */
    v7.perf.loadProgressiveImage = function(img, lowResSrc, highResSrc) {
        // Show low-res immediately
        if (lowResSrc) {
            img.src = lowResSrc;
            img.classList.add('loading-lowres');
        }

        // Load high-res in background
        const highResImg = new Image();
        highResImg.onload = () => {
            img.src = highResSrc;
            img.classList.remove('loading-lowres');
            img.classList.add('loading-complete');
        };
        highResImg.src = highResSrc;
    };

    /**
     * Load progressive image from data-src attribute
     */
    function loadProgressiveImage(img) {
        const lowResSrc = img.dataset.srcLowRes;
        const highResSrc = img.dataset.src;

        v7.perf.loadProgressiveImage(img, lowResSrc, highResSrc);
    }

    /**
     * Add progressive loading to all images with data-src
     */
    v7.perf.enableProgressiveImages = function() {
        const images = document.querySelectorAll('img[data-src]');

        images.forEach(img => {
            v7.perf.observe(img);
        });

        console.log(`[v7.perf] Progressive loading enabled for ${images.length} images`);
    };

    // ============================================================================
    // RESOURCE PRELOADING
    // ============================================================================

    /**
     * Preload resources (images, scripts, stylesheets)
     *
     * @param {string} url - Resource URL
     * @param {string} as - Resource type ('image', 'script', 'style')
     */
    v7.perf.preload = function(url, as = 'fetch') {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.href = url;
        link.as = as;

        document.head.appendChild(link);

        console.log(`[v7.perf] Preloading: ${url} (${as})`);
    };

    /**
     * Prefetch resources for later use
     */
    v7.perf.prefetch = function(url) {
        const link = document.createElement('link');
        link.rel = 'prefetch';
        link.href = url;

        document.head.appendChild(link);

        console.log(`[v7.perf] Prefetching: ${url}`);
    };

    // ============================================================================
    // PERFORMANCE MONITORING
    // ============================================================================

    /**
     * Mark performance milestone
     */
    function markPerformance(label) {
        if (performance && performance.mark) {
            performance.mark(label);
        }
    }

    /**
     * Measure performance between marks
     */
    v7.perf.measure = function(name, startMark, endMark) {
        if (performance && performance.measure) {
            try {
                performance.measure(name, startMark, endMark);
                const measures = performance.getEntriesByName(name);
                if (measures.length > 0) {
                    const duration = measures[0].duration;
                    console.log(`[v7.perf] ${name}: ${duration.toFixed(2)}ms`);
                    return duration;
                }
            } catch (e) {
                console.warn(`[v7.perf] Failed to measure ${name}:`, e);
            }
        }
        return null;
    };

    /**
     * Get page load metrics
     */
    v7.perf.getLoadMetrics = function() {
        if (!performance || !performance.timing) {
            return null;
        }

        const timing = performance.timing;

        return {
            dns: timing.domainLookupEnd - timing.domainLookupStart,
            tcp: timing.connectEnd - timing.connectStart,
            ttfb: timing.responseStart - timing.requestStart,
            download: timing.responseEnd - timing.responseStart,
            domReady: timing.domContentLoadedEventEnd - timing.navigationStart,
            loadComplete: timing.loadEventEnd - timing.navigationStart
        };
    };

    /**
     * Log performance summary
     */
    v7.perf.logSummary = function() {
        console.group('[v7.perf] Performance Summary');

        // Cache stats
        const cacheStats = v7.perf.getCacheStats();
        console.log('Cache:', cacheStats);

        // Load metrics
        const loadMetrics = v7.perf.getLoadMetrics();
        if (loadMetrics) {
            console.log('Load Metrics:', loadMetrics);
        }

        // Loaded modules
        console.log('Loaded Modules:', Array.from(state.loadedModules));

        console.groupEnd();
    };

    // ============================================================================
    // DEBOUNCE / THROTTLE UTILITIES
    // ============================================================================

    /**
     * Debounce function calls
     */
    v7.perf.debounce = function(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    };

    /**
     * Throttle function calls
     */
    v7.perf.throttle = function(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    };

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (state.cleanupTimer) {
            clearInterval(state.cleanupTimer);
        }
    });

    console.log('[v7.perf] Performance optimization module loaded');
})();
