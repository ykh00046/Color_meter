// Global namespace for v7 UI.
(function() {
const v7 = window.v7 || (window.v7 = {
    api: {},
    utils: {},
    state: {},
    render: {},
    actions: {},
    viewer: {},
    products: {},
    tabs: {},
    copy: {}
});

v7.state.currentArtifacts = { overlay: "", heatmap: "" };
v7.state.productStatus = {};
v7.state.worstCase = null;
v7.state.worstCfg = null;
v7.state.worstHotspot = null;
v7.state.inspectMode = "signature";
v7.state.hotspotBound = false;

window.addEventListener("unhandledrejection", (event) => {
    console.error("Unhandled promise rejection:", event.reason);
});

window.addEventListener("error", (event) => {
    console.error("Unhandled error:", event.error || event.message);
});

})();
