// Pan/zoom helpers for analysis viewer.
(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });

v7.viewer.initPanzoom = (viewer, wrap) => {
    if (!viewer || !wrap) return;
    const state = {
        scale: 1,
        x: 0,
        y: 0,
        minScale: 0.3,
        maxScale: 8,
        dragging: false,
        startX: 0,
        startY: 0,
        originX: 0,
        originY: 0
    };
    v7.viewer.state = state;

    const apply = () => {
        wrap.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
    };

    const reset = () => {
        state.scale = 1;
        state.x = 0;
        state.y = 0;
        apply();
    };

    const zoomAt = (delta, clientX, clientY) => {
        const rect = viewer.getBoundingClientRect();
        const sx = clientX - rect.left;
        const sy = clientY - rect.top;
        const next = delta > 0 ? state.scale * 1.1 : state.scale * 0.9;
        const newScale = Math.min(state.maxScale, Math.max(state.minScale, next));
        if (newScale == state.scale) return;
        const px = (sx - state.x) / state.scale;
        const py = (sy - state.y) / state.scale;
        state.scale = newScale;
        state.x = sx - px * newScale;
        state.y = sy - py * newScale;
        apply();
    };

    viewer.addEventListener("wheel", (event) => {
        event.preventDefault();
        zoomAt(-event.deltaY, event.clientX, event.clientY);
    }, { passive: false });

    viewer.addEventListener("mousedown", (event) => {
        if (event.button != 0) return;
        state.dragging = true;
        state.startX = event.clientX;
        state.startY = event.clientY;
        state.originX = state.x;
        state.originY = state.y;
        wrap.classList.add("dragging");
    });

    window.addEventListener("mousemove", (event) => {
        if (!state.dragging) return;
        state.x = state.originX + (event.clientX - state.startX);
        state.y = state.originY + (event.clientY - state.startY);
        apply();
    });

    window.addEventListener("mouseup", () => {
        if (!state.dragging) return;
        state.dragging = false;
        wrap.classList.remove("dragging");
    });

    viewer.addEventListener("dblclick", () => {
        reset();
    });

    v7.viewer.reset = reset;
};

v7.viewer.resetPanzoom = () => {
    if (typeof v7.viewer.reset === "function") {
        v7.viewer.reset();
    }
};

})();
