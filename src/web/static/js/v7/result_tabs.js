(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });
v7.tabs.switchResultTab = function switchResultTab(tab) {
    document.querySelectorAll(".result-tab-button").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.resultTab === tab);
    });
    document.querySelectorAll(".result-tab-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `result-${tab}`);
    });
};

v7.actions.initResultTabs = function initResultTabs() {
    document.querySelectorAll(".result-tab-button").forEach(btn => {
        btn.addEventListener("click", () => v7.tabs.switchResultTab(btn.dataset.resultTab || "v7"));
    });
};

})();
