// API helpers for v7 UI.
(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });

v7.api.apiCall = async (url, method, body, role = "operator", isJson = true) => {
    const headers = {};
    let payload = body;

    if (isJson && body != null && !(body instanceof FormData)) {
        headers["Content-Type"] = "application/json";
        payload = JSON.stringify(body);
    }
    if (role) headers["X-ROLE"] = role;

    const res = await fetch(url, {
        method,
        headers,
        body: payload
    });

    const text = await res.text();
    let data = null;
    try {
        data = text ? JSON.parse(text) : null;
    } catch {
        data = text;
    }

    if (!res.ok) {
        const msg = data?.detail || data?.error || res.statusText || "request failed";
        throw new Error(msg);
    }
    return data;
};

})();
